import { useState, lazy, Suspense } from 'react';
import Geogebra from 'react-geogebra';
import { initSession, onnxPredict } from './onnxInference.js';
import TriangleAdjust from './TriangleAdjust.jsx';
import {
  clearAllObjects,
  drawStrokePolylines,
  strokeToMnistPixels,
} from './mnistUtils.js';

const USE_SERVER = import.meta.env.VITE_USE_SERVER === 'true';
const ENABLE_BENCHMARK = import.meta.env.VITE_ENABLE_BENCHMARK === 'true';

// Lazy-load benchmark components (excluded from production build when unused)
const BenchmarkCollector = ENABLE_BENCHMARK
  ? lazy(() => import('./BenchmarkCollector.jsx'))
  : null;
const BenchmarkRunner = ENABLE_BENCHMARK
  ? lazy(() => import('./BenchmarkRunner.jsx'))
  : null;

// Lazy-load server inference module
const serverInference = USE_SERVER
  ? import('./fastApiInference.js')
  : null;

function ensureDebugText(ggbApplet, name = 'err') {
  if (!ggbApplet.exists(name)) {
    ggbApplet.evalCommand(`${name} = ""`);
    ggbApplet.setCoords(name, -6, -4);
  }
}

function ggbOnInit() {
  const ggbApplet = window.ggbApplet;
  ggbApplet.evalCommand('SetPerspective("2")');
  ggbApplet.setAxesVisible(false, false);
  ggbApplet.setGridVisible(false);
  ggbApplet.refreshViews();

  if (!USE_SERVER) {
    initSession()
      .then(() => {
        console.log('ONNX model loaded successfully');
      })
      .catch((err) => {
        console.error('ONNX init failed', err);
        ensureDebugText(ggbApplet, 'err');
        ggbApplet.setTextValue('err', 'ONNX load failed');
      });
  }
}

function stroke() {
  drawStrokePolylines(window.ggbApplet, 'err');
}

function deldata() {
  clearAllObjects(window.ggbApplet);
}

async function predict() {
  const ggbApplet = window.ggbApplet;
  const debug = 'err';
  ensureDebugText(ggbApplet, debug);

  const pixels = strokeToMnistPixels(ggbApplet);
  if (!pixels || pixels.length !== 784) {
    ggbApplet.setTextValue(debug, 'Failed to generate pixels');
    return;
  }

  try {
    let json;
    if (USE_SERVER) {
      const { fastApiPredict } = await serverInference;
      json = await fastApiPredict(pixels);
    } else {
      json = await onnxPredict(pixels, 2);
    }
    const pred = json.pred;
    const t2 = json.topk || [];
    let msg = '';

    if (t2.length >= 2) {
      const p1 = (t2[0].p * 100).toFixed(1);
      const p2 = (t2[1].p * 100).toFixed(1);
      msg = `Top-1:\\, ${pred} \\,(${p1}\\,\\%),\\, Top-2:\\, ${t2[1].i} \\,(${p2}\\,\\%)`;
    } else {
      msg = `Top-1:\\, ${pred}`;
    }

    const tex = `\\text{\\Large ${msg}}`;
    if (ggbApplet.exists('result')) ggbApplet.deleteObject('result');
    ggbApplet.evalCommand(`result = FormulaText("${tex}")`);
    ggbApplet.setCoords('result', -5, -4);
  } catch (e) {
    console.error(e);
    ggbApplet.setTextValue(debug, 'Inference failed (model not loaded)');
  }
}

function App() {
  const [mode, setMode] = useState('normal');
  const [benchmarkData, setBenchmarkData] = useState([]);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 4 }}>
        <h1 style={{ margin: 0 }}>HWR-Geo</h1>
        <span style={{ color: '#888', fontSize: '1.0em' }}>Handwriting Recognition for Geometry</span>
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <button
          type="button"
          onClick={() => setMode('normal')}
          disabled={mode === 'normal'}
        >
          Inference
        </button>
        {ENABLE_BENCHMARK && (
          <button
            type="button"
            onClick={() => setMode('benchmark')}
            disabled={mode === 'benchmark'}
          >
            Benchmark
          </button>
        )}
        <button
          type="button"
          onClick={() => setMode('adjust')}
          disabled={mode === 'adjust'}
        >
          Triangle Adjustment
        </button>
      </div>

      <Geogebra
        width="800"
        height="600"
        showToolBar
        allowStyleBar
        showMenuBar="false"
        showAlgebraInput="false"
        appletOnLoad={ggbOnInit}
      />

      {mode === 'normal' && (
        <div style={{ marginTop: 8 }}>
          <button type="button" onClick={stroke}>
            <big>Stroke</big>
          </button>
          <button type="button" onClick={predict}>
            <big>Predict</big>
          </button>
          <button type="button" onClick={deldata}>
            <big>Clear</big>
          </button>
        </div>
      )}

      {mode === 'benchmark' && ENABLE_BENCHMARK && (
        <Suspense fallback={<div>Loading...</div>}>
          <BenchmarkCollector onStartBenchmark={setBenchmarkData} />
          {benchmarkData.length > 0 && <BenchmarkRunner collectedData={benchmarkData} />}
        </Suspense>
      )}

      {mode === 'adjust' && (
        <TriangleAdjust />
      )}
    </div>
  );
}

export default App;
