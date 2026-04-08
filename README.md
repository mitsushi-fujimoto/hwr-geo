# HWR-Geo — Handwriting Recognition for Geometry

Handwritten digit recognition in [GeoGebra](https://www.geogebra.org/) using ONNX Runtime Web. Draw digits on a GeoGebra canvas, and the app recognizes them directly in the browser — no server required.

## Demo

Deployed on GitHub Pages: <https://mitsushi-fujimoto.github.io/hwr-geo/>

## Features

- **Digit Recognition** — Draw a digit on the GeoGebra canvas, click **Predict**, and see the top-2 results with confidence scores.
- **Triangle Adjustment** — Draw a triangle, write digits near each side, and click **Adjust** to reshape the triangle to the given side-length ratios.
- **Server Inference** (optional) — Use a FastAPI + PyTorch backend instead of in-browser ONNX.
- **Benchmark Mode** (optional) — Collect 100 handwritten samples (10 per digit) and compare inference latency across FastAPI, ONNX/WASM, and ONNX/WebGPU.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173/` in the browser. Use GeoGebra's **Pen Tool** to draw strokes on the canvas.

## Production Build

```bash
npm run build
npm run preview
```

The production build includes only digit recognition and triangle adjustment modes.

## Server Inference (Optional)

Requires Python 3.12+.

```bash
pip install torch torchvision fastapi uvicorn

# Terminal 1: Start the server
cd mnist
uvicorn server:app --host 0.0.0.0 --port 8001

# Terminal 2: Start the frontend with server mode
VITE_USE_SERVER=true npm run dev
```

## Benchmark Mode (Optional)

Enabled by default in development:

```bash
npm run dev   # benchmark tab is available
```

To enable in a custom build:

```bash
VITE_ENABLE_BENCHMARK=true npm run build
```

> **Note:** To include FastAPI server inference in the benchmark results, the inference server must be running (see [Server Inference](#server-inference-optional)). Without it, only ONNX/WASM and ONNX/WebGPU results are collected.

### Benchmark Data

`data/benchmark_data.json` contains the 100 handwritten digit samples (10 per digit, 0–9) used in the experiments reported in the paper. Each entry has the following fields:

| Field | Description |
|---|---|
| `label` | Ground-truth digit (0–9) |
| `attempt` | Sample index within the digit (1–10) |
| `pixels` | 784 integer values (28×28 grayscale, 0–255) |

To run the benchmark with this data, start the app in development mode (`npm run dev`), switch to the Benchmark tab, and load the JSON file.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_USE_SERVER` | `false` | Use FastAPI server instead of ONNX Runtime Web |
| `VITE_ENABLE_BENCHMARK` | `false` (prod) / `true` (dev) | Show benchmark data collection and runner |

## Model

A fully connected neural network with two hidden layers (128 units each), trained on MNIST. The PyTorch checkpoint (`mnist/mnist_net2.pth`) is exported to ONNX (`public/mnist_net2.onnx`) for browser inference.

- Input: 784-dimensional vector (28×28 grayscale, normalized to [-1, 1])
- Output: 10-class logits (softmax applied in the browser)
- Parameters: 118,282

## Project Structure

```
src/
  App.jsx               # Main component with GeoGebra canvas
  onnxInference.js      # ONNX Runtime Web inference
  fastApiInference.js   # FastAPI client
  mnistUtils.js         # Stroke extraction and 28x28 rasterization
  TriangleAdjust.jsx    # Triangle adjustment UI
  triangleUtils.js      # Triangle geometry and digit-to-edge association
  BenchmarkCollector.jsx  # Benchmark data collection
  BenchmarkRunner.jsx     # Benchmark execution and stats
data/
  benchmark_data.json   # 100 handwritten digit samples for benchmark experiments
mnist/
  server.py             # FastAPI inference server
  export_onnx.py        # PyTorch to ONNX export script
public/
  mnist_net2.onnx       # ONNX model for browser inference
```

## License

MIT License. See [LICENSE](LICENSE).

## References

- M. Fujimoto, "Integrating GeoGebra with React and WebAssembly: A Web-Based Approach for Mathematical Software Development", in *Proc. ICMS 2024*, LNCS, vol.14749, pp. 343-353, 2024.
- M. Fujimoto, "Using a Machine Learning Model for Handwritten Digit Recognition in GeoGebra", preprint, 2026.
