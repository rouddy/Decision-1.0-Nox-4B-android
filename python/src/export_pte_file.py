import argparse
import json
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

MODEL_ID = "vllm-sr/Decision-2.0-Nox-4B"


class NoxExportWrapper(torch.nn.Module):
    """
    ExecuTorch export boundary for Decision-2.0-Nox-4B.

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

        # Decision2Model (transformers remote code)
        #
        # .decision = runtime.backend.model  (DecisionModel, plain nn.Module)
        #   ├── backbone  (Qwen3_5TextModel)
        #   └── head      (CandidateHead)
        decision = getattr(decision_model, "decision", None)

        if not (
            isinstance(decision, torch.nn.Module)
            and hasattr(decision, "backbone")
            and hasattr(decision, "head")
        ):
            raise RuntimeError(
                "Could not find DecisionModel (backbone + head) inside Decision2Model"
            )

        head_variant = decision.metadata.get("head_variant", "shared")

        if head_variant != "shared":
            # Other head variants also need task_type_ids / score_level_indices.
            raise RuntimeError(f"Unsupported head_variant: {head_variant}")

        self.model = decision

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


def canonical(value) -> str:
    """Decision 2.0 data.canonical()."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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

    Follows Decision 2.0 encode() (prompt_version
    "decision2-segmented-options-global-query-v1"), the same format as
    NoxPromptBuilder in the app, so the reference / quantized outputs
    compared below are meaningful.
    """

    prefix = (
        "Context:\n"
        "The order arrived damaged yesterday. The customer has a receipt "
        "and asks for a replacement today.\n\n"
        "Task type: choice\n"
        "Question:\n"
        "Which team should handle this request?\n"
        "Options:"
    )

    teams = [
        ("returns", "Refunds, replacements and damaged deliveries"),
        ("billing", "Payments, invoices and charges"),
        ("technical", "Product setup and faults"),
    ]
    teams += [
        (f"team_{i}", f"Other department number {i}")
        for i in range(len(teams), num_candidates)
    ]
    teams = teams[:num_candidates]

    suffix = (
        "\n\nSelect the single option best supported by the context "
        "and instructions.\nDecision:"
    )

    ids = tokenizer.encode(prefix, add_special_tokens=False)
    endpoints = []

    for key, description in teams:
        option = (
            "\n<option>\n"
            + canonical({"key": key, "description": description})
            + "\n</option>"
        )
        ids.extend(tokenizer.encode(option, add_special_tokens=False))

        # Candidate position = final token of the option.
        endpoints.append(len(ids) - 1)

    ids.extend(tokenizer.encode(suffix, add_special_tokens=False))

    if len(ids) > seq_len:
        raise ValueError(f"example prompt has {len(ids)} tokens, seq_len={seq_len}")

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    input_ids[:, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask[:, : len(ids)] = 1

    candidate_positions = torch.tensor(
        [endpoints] * batch_size,
        dtype=torch.long,
    )

    candidate_mask = torch.ones(
        (batch_size, num_candidates),
        dtype=torch.long,
    )

    # Last non-padding token is the global query position.
    query_positions = torch.full((batch_size,), len(ids) - 1, dtype=torch.long)

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
        default="decision2_nox_4b.pte",
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
        help="Disable 8da4w quantization (fp32, ~17 GB; does not fit on a phone).",
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

    # Decision2Model.to() rejects dtype arguments ("numerics are fixed"),
    # and without `device` the runtime picks cuda:0 when a GPU is visible,
    # so the device is chosen at load time.
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        device=str(device),
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

    tokenizer = model.runtime.backend.tokenizer

    wrapper = NoxExportWrapper(model)
    wrapper.eval()

    # The wrapper holds the plain nn.Module (DecisionModel), which can be cast.
    # XNNPACK runs fp32 activations (int8 dynamic for quantized linears);
    # on CPU Decision 2.0 already loads as fp32, so this is a no-op there
    # (on a GPU the runtime keeps BF16 Linear weights).
    wrapper.to(device=device, dtype=torch.float32)

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
        print(f"Quantizing backbone (8da4w, group_size={args.group_size})...")

        # Backbone only: the candidate head is small and runs in FP32 in
        # the reference runtime, so it is kept FP32 here as well.
        #
        # Linear: int8 dynamic activation + int4 weight (8da4w) -> XNNPACK
        # Embedding: int8 weight-only (8w) -> quantized embedding_byte kernel
        quantize_model_(
            wrapper.model.backbone,
            qlinear_config="8da4w",
            qlinear_group_size=args.group_size,
            qembedding_config="8w",
            skip_incompatible_shapes=True,
        )

        with torch.no_grad():
            quantized = wrapper(*example_inputs)

        print(
            "Reference probabilities:",
            torch.softmax(reference, dim=-1)[0, :4].tolist(),
        )
        print(
            "Quantized probabilities:",
            torch.softmax(quantized, dim=-1)[0, :4].tolist(),
        )
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
