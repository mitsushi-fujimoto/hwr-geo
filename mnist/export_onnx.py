import os
import sys

import torch
import torch.nn as nn

# Net2 class definition
class Net2(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

def main():
    model_path = os.getenv("MODEL_PATH", "mnist_net2.pth")
    output_path = os.getenv("OUTPUT_PATH", "mnist_net2.onnx")

    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Validate checkpoint
    for key in ("n_input", "n_output", "n_hidden", "state_dict"):
        if key not in ckpt:
            print(f"Error: Checkpoint missing key '{key}'", file=sys.stderr)
            sys.exit(1)

    n_input = int(ckpt["n_input"])
    n_output = int(ckpt["n_output"])
    n_hidden = int(ckpt["n_hidden"])

    print(f"Model architecture: {n_input} -> {n_hidden} -> {n_hidden} -> {n_output}")

    # Create and load model
    model = Net2(n_input, n_output, n_hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Create dummy input for export
    dummy_input = torch.randn(1, n_input)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        external_data=False,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Validate ONNX model
    print("Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    except ImportError:
        print("Warning: onnx package not installed, skipping validation")
    except Exception as e:
        print(f"Warning: ONNX validation failed: {e}")

    # Compare PyTorch and ONNX outputs
    print("Comparing PyTorch and ONNX outputs...")
    try:
        import onnxruntime as ort
        import numpy as np

        # PyTorch inference
        test_input = torch.randn(1, n_input)
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()

        # ONNX Runtime inference
        ort_session = ort.InferenceSession(output_path)
        onnx_output = ort_session.run(None, {"input": test_input.numpy()})[0]

        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        print(f"Max difference between PyTorch and ONNX: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("Outputs match within tolerance")
        else:
            print("Warning: Outputs differ more than expected")

    except ImportError:
        print("Warning: onnxruntime not installed, skipping comparison")
    except Exception as e:
        print(f"Warning: Comparison failed: {e}")

    print(f"\nExport complete: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
