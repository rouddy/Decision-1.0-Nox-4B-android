import argparse
from pathlib import Path

import torch
from torch.export import export

from executorch.exir import EdgeCompileConfig, to_edge

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
            [batch, num_candidates], bool

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
            candidate_mask=candidate_mask,
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
    # This example is deliberately simple. For production export,
    # construct these positions using the exact Decision runtime
    # tokenizer/prompt construction.
    candidate_positions = torch.tensor(
        [
            [20, 25],
        ],
        dtype=torch.long,
    ).repeat(batch_size, 1)

    candidate_mask = torch.ones(
        (batch_size, num_candidates),
        dtype=torch.bool,
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
        default=2,
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

    model.to(device)

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

    print("Running torch.export()...")

    with torch.no_grad():
        exported = export(
            wrapper,
            example_inputs,
        )

    print("torch.export() succeeded")

    # Check exported graph.
    print(exported.graph_module)

    print("Converting to ExecuTorch Edge IR...")

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
    )

    edge_manager = to_edge(
        exported,
        compile_config=edge_config,
    )

    print("Converting to ExecuTorch...")

    et_program = edge_manager.to_executorch()

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
