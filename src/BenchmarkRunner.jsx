import { useMemo, useState } from 'react';
import { checkFastApiHealth, fastApiPredict } from './fastApiInference.js';
import {
  initWasmSession,
  initWebGpuSession,
  onnxPredictWithSession,
} from './onnxInference.js';

const METHOD_ORDER = ['FastAPI', 'ONNX(WASM)', 'ONNX(WebGPU)'];

function calculateStats(times, correctCount, totalCount) {
  if (!times.length || totalCount === 0) return null;

  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((acc, v) => acc + v, 0) / n;
  const median =
    n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[(n - 1) / 2];
  const min = sorted[0];
  const max = sorted[n - 1];
  const variance = sorted.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const accuracy = (correctCount / totalCount) * 100;

  return { mean, median, min, max, std, accuracy };
}

function formatMs(v) {
  return typeof v === 'number' ? v.toFixed(3) : '-';
}

function formatPct(v) {
  return typeof v === 'number' ? v.toFixed(1) : '-';
}

export default function BenchmarkRunner({ collectedData, serverUrl = 'http://localhost:8001' }) {
  const [isRunning, setIsRunning] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [methodResults, setMethodResults] = useState({});
  const [records, setRecords] = useState([]);

  const canRun = collectedData.length === 100;

  const summaryRows = useMemo(() => {
    return METHOD_ORDER.map((method) => {
      const result = methodResults[method];
      if (!result || result.status !== 'ok' || !result.stats) {
        return {
          method,
          mean: null,
          median: null,
          min: null,
          max: null,
          std: null,
          accuracy: null,
          status: result?.status || 'pending',
          reason: result?.reason || '',
        };
      }

      return {
        method,
        ...result.stats,
        status: 'ok',
        reason: '',
      };
    });
  }, [methodResults]);

  const downloadCsv = () => {
    if (!records.length) return;
    const header = 'digit,attempt,method,inference_time_ms,predicted,correct';
    const lines = records.map((r) =>
      [
        r.digit,
        r.attempt,
        `"${r.method}"`,
        r.inference_time_ms.toFixed(4),
        r.predicted,
        r.correct ? 1 : 0,
      ].join(','),
    );
    const csv = `${header}\n${lines.join('\n')}`;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadStats = () => {
    const okRows = summaryRows.filter((r) => r.status === 'ok');
    if (!okRows.length) return;
    const header = 'Method,Mean(ms),Median(ms),Min(ms),Max(ms),Std.Dev.(ms),Accuracy(%)';
    const lines = okRows.map((r) =>
      [
        `"${r.method}"`,
        formatMs(r.mean),
        formatMs(r.median),
        formatMs(r.min),
        formatMs(r.max),
        formatMs(r.std),
        formatPct(r.accuracy),
      ].join(','),
    );
    const csv = `${header}\n${lines.join('\n')}`;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_stats_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const runBenchmark = async () => {
    if (!canRun || isRunning) return;

    setIsRunning(true);
    setProgressMessage('Initializing...');
    setRecords([]);

    const localResults = {
      FastAPI: { status: 'pending', reason: '' },
      'ONNX(WASM)': { status: 'pending', reason: '' },
      'ONNX(WebGPU)': { status: 'pending', reason: '' },
    };
    const localRecords = [];

    const runMethod = async (method, infer) => {
      try {
        setProgressMessage(`${method}: warming up...`);
        await infer(collectedData[0].pixels);

        const times = [];
        let correctCount = 0;
        for (let i = 0; i < collectedData.length; i++) {
          const sample = collectedData[i];
          setProgressMessage(`${method}: running ${i + 1}/${collectedData.length}`);
          const start = performance.now();
          const json = await infer(sample.pixels);
          const elapsed = performance.now() - start;
          const pred = Number(json?.pred);
          const correct = pred === sample.label;

          times.push(elapsed);
          if (correct) correctCount += 1;
          localRecords.push({
            digit: sample.label,
            attempt: sample.attempt,
            method,
            inference_time_ms: elapsed,
            predicted: pred,
            correct,
          });
        }

        localResults[method] = {
          status: 'ok',
          reason: '',
          stats: calculateStats(times, correctCount, collectedData.length),
        };
      } catch (err) {
        localResults[method] = {
          status: 'error',
          reason: err instanceof Error ? err.message : 'unknown error',
        };
      }
    };

    try {
      setProgressMessage('Checking FastAPI server...');
      let fastApiAvailable = false;
      try {
        const health = await checkFastApiHealth(serverUrl);
        fastApiAvailable = Boolean(health?.ok) && health?.model_loaded !== false;
        if (!fastApiAvailable) {
          localResults.FastAPI = {
            status: 'skipped',
            reason: 'Health responded but model_loaded=false',
          };
        }
      } catch (err) {
        localResults.FastAPI = {
          status: 'skipped',
          reason: err instanceof Error ? err.message : 'health check failed',
        };
      }

      setProgressMessage('Initializing ONNX(WASM) session...');
      try {
        const wasmSession = await initWasmSession();
        await runMethod('ONNX(WASM)', (pixels) =>
          onnxPredictWithSession(wasmSession, pixels, 2),
        );
      } catch (err) {
        localResults['ONNX(WASM)'] = {
          status: 'error',
          reason: err instanceof Error ? err.message : 'WASM init failed',
        };
      }

      setProgressMessage('Initializing ONNX(WebGPU) session...');
      try {
        const webGpuSession = await initWebGpuSession();
        await runMethod('ONNX(WebGPU)', (pixels) =>
          onnxPredictWithSession(webGpuSession, pixels, 2),
        );
      } catch (err) {
        localResults['ONNX(WebGPU)'] = {
          status: 'skipped',
          reason: err instanceof Error ? err.message : 'WebGPU init failed',
        };
      }

      if (fastApiAvailable) {
        await runMethod('FastAPI', (pixels) => fastApiPredict(pixels, serverUrl));
      }
    } finally {
      setMethodResults(localResults);
      setRecords(localRecords);
      setProgressMessage('Benchmark completed.');
      setIsRunning(false);
    }
  };

  return (
    <section style={{ marginTop: 16, padding: 12, border: '1px solid #ccc' }}>
      <h2 style={{ marginTop: 0 }}>Benchmark Results</h2>
      <p>Samples: {collectedData.length}/100</p>
      <button type="button" onClick={runBenchmark} disabled={!canRun || isRunning}>
        {isRunning ? 'Running...' : 'Start Benchmark'}
      </button>
      {progressMessage && <p>{progressMessage}</p>}

      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12 }}>
        <thead>
          <tr>
            <th style={{ border: '1px solid #ccc' }}>Method</th>
            <th style={{ border: '1px solid #ccc' }}>Mean(ms)</th>
            <th style={{ border: '1px solid #ccc' }}>Median(ms)</th>
            <th style={{ border: '1px solid #ccc' }}>Min(ms)</th>
            <th style={{ border: '1px solid #ccc' }}>Max(ms)</th>
            <th style={{ border: '1px solid #ccc' }}>Std.Dev.(ms)</th>
            <th style={{ border: '1px solid #ccc' }}>Accuracy(%)</th>
          </tr>
        </thead>
        <tbody>
          {summaryRows.map((row) => (
            <tr key={row.method}>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{row.method}</td>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{formatMs(row.mean)}</td>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{formatMs(row.median)}</td>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{formatMs(row.min)}</td>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{formatMs(row.max)}</td>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{formatMs(row.std)}</td>
              <td style={{ border: '1px solid #ccc', padding: 4 }}>{formatPct(row.accuracy)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button type="button" onClick={downloadCsv} disabled={!records.length || isRunning}>
          Download CSV
        </button>
        <button
          type="button"
          onClick={downloadStats}
          disabled={!summaryRows.some((r) => r.status === 'ok') || isRunning}
        >
          Download Stats
        </button>
      </div>

      <div style={{ marginTop: 8 }}>
        {summaryRows
          .filter((row) => row.status !== 'ok' && row.reason)
          .map((row) => (
            <p key={`${row.method}-reason`} style={{ margin: 0 }}>
              {row.method}: {row.reason}
            </p>
          ))}
      </div>
    </section>
  );
}
