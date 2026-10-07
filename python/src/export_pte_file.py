import argparse
from pathlib import Path

import torch
from torch.export import export

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.llm.export.quantize import quantize_model_

from transformers import AutoModel

MODEL_ID = "vllm-sr/Decision-1.0-Nox-4B"


class NoxExportWrapper(torch.nn.Module):
    """
    ExecuTorch export boundary for Decision-1.0-Nox-4B.

    Inputs:
        input_ids:
            [batch, seq_len], int64

        attention_mask:
            [batch, seq_len], int64

        candidate_positions:
            [batch, num_candidates], int64

        candidate_mask:
            [batch, num_candidates], int64
            (Android Tensor는 bool을 만들기 번거로우므로 int64로 받고
            그래프 안에서 bool로 변환한다)

        query_positions:
            [batch], int64

    Output:
        scores:
            [batch, num_candidates], float32
    """

    def __init__(self, decision_model):
        super().__init__()

        # Decision1Model
        #
        # runtime.model
        #   └── QwenDecision
        #         ├── backbone
        #         └── head
        #
        # The exact object hierarchy is implementation-dependent, so
        # find the QwenDecision module recursively.
        qwen_decision = None

        for module in decision_model.modules():
            if (
                hasattr(module, "backbone")
                and hasattr(module, "head")
                and callable(getattr(module, "forward", None))
            ):
                qwen_decision = module
                break

        if qwen_decision is None:
            raise RuntimeError(
                "Could not find QwenDecision module inside Decision1Model"
            )

        self.model = qwen_decision

    def forward(
        self,
        input_ids,
        attention_mask,
        candidate_positions,
        candidate_mask,
        query_positions,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_positions=candidate_positions,
            candidate_mask=candidate_mask.to(torch.bool),
            query_positions=query_positions,
        )


def make_example_inputs(
    tokenizer,
    *,
    batch_size: int,
    seq_len: int,
    num_candidates: int,
    device: str,
):
    """
    Creates a representative input for torch.export().
    """

    # Use a real tokenizer output rather than arbitrary token IDs.
    text = (
        "Context:\n"
        "Customer requests a refund.\n\n"
        "Task type: choice\n"
        "Question:\n"
        "Which team should handle this?\n"
        "Options:\n"
        "billing: Payments and refunds\n"
        "technical: Product faults\n"
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )

    input_ids = encoded["input_ids"]

    # Force the exact export sequence length.
    if input_ids.shape[1] != seq_len:
        padded = torch.full(
            (batch_size, seq_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )

        length = min(input_ids.shape[1], seq_len)

        padded[:, :length] = input_ids[:, :length]

        input_ids = padded

    else:
        input_ids = input_ids.repeat(batch_size, 1)

    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)

    # Candidate endpoint positions.
    #
    # These must point to the final token of each candidate segment.
    #
    # The graph is exported with a static shape, so the example only
    # needs num_candidates valid positions inside the non-padding
    # tokens (the app pads unused slots and masks them out).
    real_len = int(attention_mask[0].sum().item())

    if real_len < num_candidates + 1:
        raise ValueError(
            f"example prompt has {real_len} tokens, "
            f"needs more than num_candidates={num_candidates}"
        )

    candidate_positions = (
        torch.linspace(1, real_len - 2, num_candidates)
        .round()
        .to(torch.long)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    candidate_mask = torch.ones(
        (batch_size, num_candidates),
        dtype=torch.long,
    )

    # Last non-padding token is the global query position.
    query_positions = (attention_mask.sum(dim=1) - 1).to(torch.long)

    return (
        input_ids.to(device),
        attention_mask.to(device),
        candidate_positions.to(device),
        candidate_mask.to(device),
        query_positions.to(device),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--output",
        default="decision_nox_4b.pte",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Static sequence length used by the exported graph.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=16,
        help="Static candidate count. Must match NoxModelRunner.maxCandidates.",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="Group size for int4 weight quantization.",
    )

    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable 8da4w quantization (fp32, very large).",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available")

    print(f"Loading {args.model}")

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    model.eval()

    print("\n=== Model ===")
    print(type(model))

    print("\n=== Parameters ===")

    dtypes = {}

    for name, param in model.named_parameters():
        dtypes.setdefault(str(param.dtype), 0)
        dtypes[str(param.dtype)] += param.numel()

    for dtype, count in dtypes.items():
        print(f"{dtype}: {count:,} parameters")

    print("\n=== Modules ===")

    for name, module in model.named_modules():
        if (
            "backbone" in name.lower()
            or "head" in name.lower()
            or "decoder" in name.lower()
            or "encoder" in name.lower()
        ):
            print(name, type(module))

    # XNNPACK runs fp32 activations (int8 dynamic for quantized linears).
    model.to(device=device, dtype=torch.float32)

    tokenizer = model.runtime.tokenizer

    wrapper = NoxExportWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    example_inputs = make_example_inputs(
        tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_candidates=args.num_candidates,
        device=str(device),
    )

    print("Running reference inference...")

    with torch.no_grad():
        reference = wrapper(*example_inputs)

    print(
        "Reference output:",
        reference.shape,
        reference.dtype,
    )

    if not args.no_quant:
        print(f"Quantizing (8da4w, group_size={args.group_size})...")

        # Linear: int8 dynamic activation + int4 weight (8da4w) -> XNNPACK
        # Embedding: int8 weight-only (8w) -> quantized embedding_byte kernel
        quantize_model_(
            wrapper,
            qlinear_config="8da4w",
            qlinear_group_size=args.group_size,
            qembedding_config="8w",
            skip_incompatible_shapes=True,
        )

        with torch.no_grad():
            quantized = wrapper(*example_inputs)

        print(
            "Quantized max abs diff:",
            (quantized - reference).abs().max().item(),
        )
        print(
            "Argmax match:",
            torch.equal(quantized.argmax(dim=-1), reference.argmax(dim=-1)),
        )

    print("Running torch.export()...")

    with torch.no_grad():
        exported = export(
            wrapper,
            example_inputs,
        )

    print("torch.export() succeeded")

    # Check exported graph.
    print(exported.graph_module)

    print("Converting to ExecuTorch Edge IR (XNNPACK)...")

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
    )

    edge_manager = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=edge_config,
    )

    print("Converting to ExecuTorch...")

    et_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            # Fuse the quantized embedding into embedding_byte.
            do_quant_fusion_and_const_prop=True,
        )
    )

    output = Path(args.output)

    output.write_bytes(et_program.buffer)

    print()
    print("======================================")
    print("ExecuTorch export completed")
    print("======================================")
    print(f"Output : {output}")
    print(f"Size   : {output.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
