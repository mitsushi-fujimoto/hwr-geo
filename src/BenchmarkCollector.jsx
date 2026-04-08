import { useRef, useMemo, useState } from 'react';
import { clearAllObjects, strokeToMnistPixels } from './mnistUtils.js';

const DIGITS = 10;
const ATTEMPTS_PER_DIGIT = 10;
const TOTAL_SAMPLES = DIGITS * ATTEMPTS_PER_DIGIT;

export default function BenchmarkCollector({ onStartBenchmark, disabled = false }) {
  const [collectedData, setCollectedData] = useState([]);
  const [message, setMessage] = useState('');
  const fileInputRef = useRef(null);

  const completed = collectedData.length >= TOTAL_SAMPLES;
  const currentIndex = Math.min(collectedData.length, TOTAL_SAMPLES - 1);
  const currentDigit = Math.floor(currentIndex / ATTEMPTS_PER_DIGIT);
  const currentAttempt = (currentIndex % ATTEMPTS_PER_DIGIT) + 1;
  const progressPercent = useMemo(
    () => Math.round((collectedData.length / TOTAL_SAMPLES) * 100),
    [collectedData.length],
  );

  const saveAndNext = () => {
    const pixels = strokeToMnistPixels(window.ggbApplet);
    if (!pixels || pixels.length !== 784) {
      setMessage('No strokes found. Please write a digit before saving.');
      return;
    }

    const sample = { label: currentDigit, attempt: currentAttempt, pixels };
    setCollectedData((prev) => [...prev, sample]);
    clearAllObjects(window.ggbApplet);
    setMessage(`Saved digit ${currentDigit} (${currentAttempt}/10).`);
  };

  const redoCurrentInput = () => {
    clearAllObjects(window.ggbApplet);
    setMessage('Current input cleared.');
  };

  const resetCollection = () => {
    setCollectedData([]);
    clearAllObjects(window.ggbApplet);
    setMessage('Collection data reset.');
  };

  const downloadData = () => {
    const json = JSON.stringify(collectedData);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadData = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (!Array.isArray(data) || data.length !== TOTAL_SAMPLES) {
          setMessage(`Data must be an array of ${TOTAL_SAMPLES} items.`);
          return;
        }
        for (const item of data) {
          if (
            typeof item.label !== 'number' ||
            item.label < 0 ||
            item.label > 9 ||
            typeof item.attempt !== 'number' ||
            item.attempt < 1 ||
            item.attempt > ATTEMPTS_PER_DIGIT ||
            !Array.isArray(item.pixels) ||
            item.pixels.length !== 784
          ) {
            setMessage('Invalid data format (check label/attempt/pixels).');
            return;
          }
        }
        setCollectedData(data);
        setMessage(`Loaded ${data.length} samples.`);
      } catch {
        setMessage('Failed to read file.');
      }
    };
    reader.readAsText(file);
    // Reset so the same file can be re-selected
    event.target.value = '';
  };

  return (
    <section style={{ marginTop: 16, padding: 12, border: '1px solid #ccc' }}>
      <h2 style={{ marginTop: 0 }}>Benchmark Data Collection</h2>
      {!completed && (
        <p>
          Write digit <strong>{currentDigit}</strong> ({currentAttempt}/10)
        </p>
      )}
      {completed && <p>100 samples collected. Ready to run benchmark.</p>}

      <p>Progress: {collectedData.length}/100</p>
      <progress value={collectedData.length} max={TOTAL_SAMPLES} style={{ width: '100%' }} />
      <p>{progressPercent}%</p>

      <input
        type="file"
        accept=".json"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={loadData}
      />

      {!completed && (
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button type="button" onClick={() => fileInputRef.current?.click()} disabled={disabled}>
            Load Data
          </button>
          <button type="button" onClick={saveAndNext} disabled={disabled}>
            Save &amp; Next
          </button>
          <button type="button" onClick={redoCurrentInput} disabled={disabled}>
            Redo
          </button>
        </div>
      )}

      {completed && (
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button type="button" onClick={() => fileInputRef.current?.click()} disabled={disabled}>
            Load Data
          </button>
          <button
            type="button"
            onClick={() => onStartBenchmark(collectedData)}
            disabled={disabled}
          >
            Run Benchmark
          </button>
          <button type="button" onClick={resetCollection} disabled={disabled}>
            Reset
          </button>
          <button type="button" onClick={downloadData} disabled={disabled}>
            Save Data
          </button>
        </div>
      )}

      {message && <p style={{ marginBottom: 0 }}>{message}</p>}
    </section>
  );
}
