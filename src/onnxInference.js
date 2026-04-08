import * as ort from 'onnxruntime-web';

const MODEL_URL = `${import.meta.env.BASE_URL}mnist_net2.onnx`;
let sessionPromise = null;
let wasmSessionPromise = null;
let webGpuSessionPromise = null;

// Select Execution Provider
export async function initSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu', 'wasm'],
    });
  }
  return sessionPromise;
}

// WASM only session for benchmark
export async function initWasmSession() {
  if (!wasmSessionPromise) {
    wasmSessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
    });
  }
  return wasmSessionPromise;
}

// WebGPU only session for benchmark
export async function initWebGpuSession() {
  if (!webGpuSessionPromise) {
    webGpuSessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu'],
    });
  }
  return webGpuSessionPromise;
}

// normalization: 0..255 -> 0..1 -> -1..1
export function preprocess(pixels) {
  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.min(255, Math.max(0, pixels[i]));
    out[i] = (v / 255.0 - 0.5) / 0.5;
  }
  return out;
}

// softmax: logits -> probability distribution
export function softmax(logits) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - maxLogit);
    exps[i] = v;
    sum += v;
  }

  const probs = new Array(logits.length);
  if (sum === 0) return probs.fill(0);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = exps[i] / sum;
  }
  return probs;
}

// topK: select the k highest probability items
export function topK(probs, k = 2) {
  const items = probs.map((p, i) => ({ i, p }));
  items.sort((a, b) => b.p - a.p);
  return items.slice(0, Math.max(1, Math.min(k, items.length)));
}

// predict by specified ONNX session
export async function onnxPredictWithSession(session, pixels, k = 2) {
  const input = preprocess(pixels);
  const tensor = new ort.Tensor('float32', input, [1, input.length]);
  const feeds = { input: tensor };

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName] || results.output;
  if (!outputTensor) {
    throw new Error('ONNX output not found');
  }

  const probs = softmax(outputTensor.data);
  const topk = topK(probs, k);
  return { pred: topk[0].i, topk };
}

// predict by ONNX model
export async function onnxPredict(pixels, k = 2) {
  const session = await initSession();
  return onnxPredictWithSession(session, pixels, k);
}
