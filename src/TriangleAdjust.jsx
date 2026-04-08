import { useEffect, useRef, useState } from 'react';
import { clearAllObjects, strokesToMnistPixels } from './mnistUtils.js';
import { onnxPredict } from './onnxInference.js';
import {
  adjustTriangleToRatios,
  associateDigitsToEdges,
  classifyStrokes,
  collectAllPoints,
  drawTriangleOnCanvas,
  extractTriangleVertices,
  getAllStrokes,
  groupDigitStrokes,
  hasTriangleEdgeCoverage,
} from './triangleUtils.js';

const SINGLE_STROKE_IDLE_MS = 500;
const MULTI_STROKE_IDLE_MS = 1000;
const POLL_MS = 250;
const DETECTED_POINT_NAMES = ['AdjV1', 'AdjV2', 'AdjV3', 'AdjTriangleGuide'];
const EDGE_LABEL_NAMES = ['AdjEdgeText01', 'AdjEdgeText12', 'AdjEdgeText20'];
const EDGE_VALUE_NAMES = ['AdjEdgeVal01', 'AdjEdgeVal12', 'AdjEdgeVal20'];
const RESULT_TRIANGLE_NAMES = ['TA', 'TB', 'TC', 'TriResult'];

function buildStrokeSignature(strokeObjects) {
  if (!Array.isArray(strokeObjects) || strokeObjects.length === 0) return '';
  return strokeObjects
    .map(({ name, subStrokes }) => {
      const sub = (subStrokes || [])
        .map((pts) => {
          if (!pts || pts.length === 0) return '0';
          const first = pts[0];
          const last = pts[pts.length - 1];
          return `${pts.length}@${first[0].toFixed(2)},${first[1].toFixed(2)}->${last[0].toFixed(2)},${last[1].toFixed(2)}`;
        })
        .join('|');
      return `${name}:${sub}`;
    })
    .join('||');
}

function clearDetectedObjects(ggbApplet) {
  if (!ggbApplet) return;
  for (let i = 0; i < DETECTED_POINT_NAMES.length; i++) {
    const name = DETECTED_POINT_NAMES[i];
    if (ggbApplet.exists(name)) ggbApplet.deleteObject(name);
  }
}

function clearEdgeLabelObjects(ggbApplet) {
  if (!ggbApplet) return;
  for (let i = 0; i < EDGE_LABEL_NAMES.length; i++) {
    const name = EDGE_LABEL_NAMES[i];
    if (ggbApplet.exists(name)) ggbApplet.deleteObject(name);
  }
  for (let i = 0; i < EDGE_VALUE_NAMES.length; i++) {
    const name = EDGE_VALUE_NAMES[i];
    if (ggbApplet.exists(name)) ggbApplet.deleteObject(name);
  }
}

function clearResultTriangle(ggbApplet) {
  if (!ggbApplet) return;
  for (let i = 0; i < RESULT_TRIANGLE_NAMES.length; i++) {
    const name = RESULT_TRIANGLE_NAMES[i];
    if (ggbApplet.exists(name)) ggbApplet.deleteObject(name);
  }
}

function showAllStrokeObjects(ggbApplet) {
  if (!ggbApplet) return;
  const count = ggbApplet.getObjectNumber();
  for (let i = 0; i < count; i++) {
    const name = ggbApplet.getObjectName(i);
    ggbApplet.setVisible(name, true);
  }
}

function drawDetectedTriangle(ggbApplet, vertices) {
  if (!ggbApplet || !vertices || vertices.length !== 3) return;
  clearDetectedObjects(ggbApplet);

  const [a, b, c] = vertices;
  const labels = ['A', 'B', 'C'];
  ggbApplet.evalCommand(`AdjV1 = (${a[0]}, ${a[1]})`);
  ggbApplet.evalCommand(`AdjV2 = (${b[0]}, ${b[1]})`);
  ggbApplet.evalCommand(`AdjV3 = (${c[0]}, ${c[1]})`);
  for (let i = 0; i < 3; i++) {
    const name = DETECTED_POINT_NAMES[i];
    ggbApplet.setCaption(name, labels[i]);
    ggbApplet.setLabelStyle(name, 3);
    ggbApplet.setLabelVisible(name, true);
  }
  // Guide shown as PolyLine (no fill) to visually distinguish from the final Polygon result (filled).
  ggbApplet.evalCommand('AdjTriangleGuide = PolyLine(AdjV1, AdjV2, AdjV3, AdjV1)');

  ggbApplet.setColor('AdjTriangleGuide', 34, 139, 34);
  ggbApplet.setLineThickness('AdjTriangleGuide', 5);
  ggbApplet.setFixed('AdjTriangleGuide', true, false);
}

function hideStrokeObjects(ggbApplet, strokeObjects) {
  if (!ggbApplet || !Array.isArray(strokeObjects)) return;
  const names = new Set(strokeObjects.map((s) => s?.name).filter(Boolean));
  names.forEach((name) => {
    if (ggbApplet.exists(name)) ggbApplet.setVisible(name, false);
  });
}

function countSubStrokes(strokeObjects) {
  if (!Array.isArray(strokeObjects)) return 0;
  let count = 0;
  for (let i = 0; i < strokeObjects.length; i++) {
    count += (strokeObjects[i]?.subStrokes || []).length;
  }
  return count;
}

function escapeErrorMessage(error) {
  if (!error) return 'Unknown error';
  if (typeof error.message === 'string' && error.message) return error.message;
  return String(error);
}

function edgeLabelPositionOutward(a, b, centroid, scale) {
  const mx = (a[0] + b[0]) / 2;
  const my = (a[1] + b[1]) / 2;
  const vx = b[0] - a[0];
  const vy = b[1] - a[1];
  const len = Math.hypot(vx, vy);
  if (len < 1e-9) return [mx, my];
  // Compute normal direction; choose the side facing away from the centroid (outward)
  const nx = -vy / len;
  const ny = vx / len;
  const toCenter = (centroid[0] - mx) * nx + (centroid[1] - my) * ny;
  const sign = toCenter <= 0 ? 1 : -1;
  return [mx + nx * scale * sign, my + ny * scale * sign];
}

function drawRecognizedEdgeLabels(ggbApplet, vertices, edgeValues) {
  if (!ggbApplet || !Array.isArray(vertices) || vertices.length !== 3) return;
  if (!Array.isArray(edgeValues) || edgeValues.length !== 3) return;

  clearEdgeLabelObjects(ggbApplet);

  const [v0, v1, v2] = vertices;
  const ctr = [
    (v0[0] + v1[0] + v2[0]) / 3,
    (v0[1] + v1[1] + v2[1]) / 3,
  ];
  const edges = [
    [v0, v1],
    [v1, v2],
    [v2, v0],
  ];

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < vertices.length; i++) {
    const [x, y] = vertices[i];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  const diag = Math.hypot(maxX - minX, maxY - minY);
  const offset = Math.max(0.18, diag * 0.08);

  for (let i = 0; i < edges.length; i++) {
    const [a, b] = edges[i];
    const [lx, ly] = edgeLabelPositionOutward(a, b, ctr, offset);
    const value = Number(edgeValues[i]);
    const safeValue = Number.isFinite(value) ? value : 0;
    const name = EDGE_LABEL_NAMES[i];
    const valueName = EDGE_VALUE_NAMES[i];
    ggbApplet.evalCommand(`${valueName} = ${safeValue}`);
    const tex = `\\Large ${safeValue}`;
    ggbApplet.evalCommand(`${name} = FormulaText("${tex}")`);
    ggbApplet.setCoords(name, lx, ly);
    ggbApplet.setColor(name, 180, 20, 20);
    ggbApplet.setFixed(name, false, true);
  }
}

function formatVertices(vertices) {
  if (!vertices || vertices.length !== 3) return '-';
  return vertices
    .map((v, i) => `${'ABC'[i]}=(${v[0].toFixed(2)}, ${v[1].toFixed(2)})`)
    .join(' / ');
}

export default function TriangleAdjust() {
  const [status, setStatus] = useState('Draw a triangle by hand');
  const [detectedVertices, setDetectedVertices] = useState(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const lastSignatureRef = useRef('');
  const lastChangedAtRef = useRef(0);
  const lastProcessedSignatureRef = useRef('');
  const baseVerticesRef = useRef(null);
  const isTriangleLockedRef = useRef(false);

  const updateResultFromEdges = (v01, v12, v20) => {
    setResult({
      labels: { edge01: v01, edge12: v12, edge20: v20 },
      ratios: { a: v12, b: v20, c: v01 },
      text: `AB : BC : CA = ${v01} : ${v12} : ${v20}`,
    });
  };

  useEffect(() => {
    lastChangedAtRef.current = Date.now();
    const timer = window.setInterval(() => {
      const ggbApplet = window.ggbApplet;
      if (!ggbApplet) {
        setStatus('Waiting for GeoGebra to initialize');
        return;
      }

      const strokeObjects = getAllStrokes(ggbApplet);
      const signature = buildStrokeSignature(strokeObjects);

      // After triangle detection, lock the base shape so digits don't trigger re-detection
      if (isTriangleLockedRef.current) {
        if (!signature) {
          setStatus('Draw a triangle by hand');
          setDetectedVertices(null);
          setResult(null);
          lastProcessedSignatureRef.current = '';
          baseVerticesRef.current = null;
          isTriangleLockedRef.current = false;
          clearDetectedObjects(ggbApplet);
          clearEdgeLabelObjects(ggbApplet);
          lastSignatureRef.current = '';
          return;
        }

        if (signature !== lastSignatureRef.current) {
          lastSignatureRef.current = signature;
          setStatus('Triangle locked. Enter digits near each side and click Adjust.');
        }
        return;
      }

      if (signature !== lastSignatureRef.current) {
        lastSignatureRef.current = signature;
        lastChangedAtRef.current = Date.now();
        if (!signature) {
          setStatus('Draw a triangle by hand');
          setDetectedVertices(null);
          setResult(null);
          lastProcessedSignatureRef.current = '';
          baseVerticesRef.current = null;
          clearDetectedObjects(ggbApplet);
          clearEdgeLabelObjects(ggbApplet);
        } else {
          const subStrokeCount = countSubStrokes(strokeObjects);
          const waitMs = subStrokeCount >= 2 ? MULTI_STROKE_IDLE_MS : SINGLE_STROKE_IDLE_MS;
          setStatus(`Waiting for input to stop (${(waitMs / 1000).toFixed(1)}s)`);
        }
        return;
      }

      if (!signature) return;

      const subStrokeCount = countSubStrokes(strokeObjects);
      const idleThreshold = subStrokeCount >= 2 ? MULTI_STROKE_IDLE_MS : SINGLE_STROKE_IDLE_MS;
      const idle = Date.now() - lastChangedAtRef.current;
      if (idle < idleThreshold) return;
      if (signature === lastProcessedSignatureRef.current) return;

      lastProcessedSignatureRef.current = signature;
      const vertices = extractTriangleVertices(strokeObjects);
      if (!vertices || vertices.length !== 3) {
        setStatus('Could not extract 3 vertices');
        setDetectedVertices(null);
        clearDetectedObjects(ggbApplet);
        return;
      }

      const { flatPoints, bbox } = collectAllPoints(strokeObjects);
      if (!hasTriangleEdgeCoverage(vertices, flatPoints, bbox)) {
        setStatus('Not all 3 edges drawn; detection is pending');
        setDetectedVertices(null);
        clearDetectedObjects(ggbApplet);
        return;
      }

      drawDetectedTriangle(ggbApplet, vertices);
      hideStrokeObjects(ggbApplet, strokeObjects);
      baseVerticesRef.current = vertices;
      isTriangleLockedRef.current = true;
      setDetectedVertices(vertices);
      setResult(null);
      clearEdgeLabelObjects(ggbApplet);
      setStatus('Triangle detected (original strokes hidden)');
    }, POLL_MS);

    return () => {
      window.clearInterval(timer);
      const ggb = window.ggbApplet;
      clearDetectedObjects(ggb);
      clearEdgeLabelObjects(ggb);
      clearResultTriangle(ggb);
      showAllStrokeObjects(ggb);
    };
  }, []);

  const recognizeDigitGroup = async (group) => {
    if (!group || !Array.isArray(group.strokes) || group.strokes.length === 0) {
      throw new Error('Digit group is empty');
    }

    const groupDiag = group?.bbox?.diagonal || 0;
    const noiseThreshold = Math.max(0.04, groupDiag * 0.1);
    const denoised = group.strokes.filter((s) => (s?.bbox?.diagonal || 0) >= noiseThreshold);
    const candidateStrokes = denoised.length > 0 ? denoised : group.strokes;
    if (candidateStrokes.length === 0) {
      throw new Error('Failed to segment digit group');
    }
    const n = candidateStrokes.length;
    if (n > 12) {
      throw new Error(`Too many strokes (${n})`);
    }
    const maxUnitSize = Math.min(3, n);

    const yOverlapRatio = (a, b) => {
      const overlap = Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY);
      if (overlap <= 0) return 0;
      const base = Math.max(1e-9, Math.min(a.height || 0, b.height || 0));
      return overlap / base;
    };

    let avgWidth = 0;
    for (let i = 0; i < n; i++) avgWidth += candidateStrokes[i].bbox.width || 0;
    avgWidth = n > 0 ? avgWidth / n : 0;
    const strongSplitHints = new Set();
    const strongJoinHints = new Set();
    const gapThreshold = Math.max(0.05, avgWidth * 0.22);
    const joinGapThreshold = Math.max(0.015, avgWidth * 0.05);
    for (let i = 0; i < n - 1; i++) {
      const left = candidateStrokes[i].bbox;
      const right = candidateStrokes[i + 1].bbox;
      const xGap = right.minX - left.maxX;
      const yOverlap = yOverlapRatio(left, right);
      if (xGap > gapThreshold) strongSplitHints.add(i);
      if (xGap < joinGapThreshold && yOverlap > 0.25) strongJoinHints.add(i);
    }

    // Pre-infer all unit candidates (O(n * maxUnitSize) times)
    const predictions = new Map();
    for (let start = 0; start < n; start++) {
      for (let size = 1; size <= maxUnitSize; size++) {
        const end = start + size - 1;
        if (end >= n) break;
        const key = `${start}:${end}`;
        const unitStrokes = [];
        for (let i = start; i <= end; i++) unitStrokes.push(candidateStrokes[i].points);
        const pixels = strokesToMnistPixels(unitStrokes);
        if (!pixels || pixels.length !== 784) {
          throw new Error('Failed to rasterize digit strokes');
        }
        const pred = await onnxPredict(pixels, 1);
        const p = pred.topk?.[0]?.p ?? 0.01;
        predictions.set(key, { digit: String(pred.pred), prob: Math.max(1e-6, p) });
      }
    }

    const penaltyPerExtraUnit = 0.18;

    // Per-unit score (log probability + merge penalty + intra-unit hint violations)
    const unitScore = (start, end) => {
      const pred = predictions.get(`${start}:${end}`);
      let s = Math.log(pred.prob);
      const mergeLen = end - start;
      if (mergeLen > 0) s -= mergeLen * 0.45;
      for (let b = start; b < end; b++) {
        if (strongSplitHints.has(b)) s -= 2.8;
        if (strongJoinHints.has(b)) s -= 0.2;
      }
      return s;
    };

    // DP: dp[i].get(numParts) = { score, text }
    // dp[i] = best result for partitioning strokes [0..i-1] into numParts units
    const dp = Array.from({ length: n + 1 }, () => new Map());
    dp[0].set(0, { score: 0, text: '' });

    for (let i = 0; i < n; i++) {
      for (const [numParts, state] of dp[i]) {
        for (let size = 1; size <= maxUnitSize; size++) {
          const end = i + size - 1;
          if (end >= n) break;
          const nextIndex = end + 1;
          const nextNumParts = numParts + 1;

          const pred = predictions.get(`${i}:${end}`);
          let transitionScore = unitScore(i, end);

          // Split boundary penalty (for 2nd unit onward)
          if (numParts > 0) {
            const boundary = i - 1;
            transitionScore -= penaltyPerExtraUnit;
            if (strongSplitHints.has(boundary)) transitionScore += 1.8;
            else if (strongJoinHints.has(boundary)) transitionScore -= 2.4;
            else transitionScore -= 1.0;
          }

          const newScore = state.score + transitionScore;
          const newText = state.text + pred.digit;

          const existing = dp[nextIndex].get(nextNumParts);
          if (!existing || newScore > existing.score) {
            dp[nextIndex].set(nextNumParts, { score: newScore, text: newText });
          }
        }
      }
    }

    // Add global penalty and select the best result
    const expectedMinParts = strongSplitHints.size + 1;
    const widthRatio = (group?.bbox?.width || 0) / Math.max(1e-9, group?.bbox?.height || 1);
    let best = null;
    for (const [numParts, state] of dp[n]) {
      let finalScore = state.score;
      if (numParts < expectedMinParts) {
        finalScore -= (expectedMinParts - numParts) * 1.2;
      }
      if (numParts > 2) {
        finalScore -= (numParts - 2) * (widthRatio < 1.8 ? 1.8 : 0.8);
      }
      if (!best || finalScore > best.score) {
        best = { score: finalScore, text: state.text };
      }
    }

    const text = best?.text || '';
    const value = Number.parseInt(text, 10);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`Invalid recognition value: ${text}`);
    }
    return { ...group, text, value };
  };

  const adjustByDigits = async () => {
    if (isProcessing) return;
    const ggbApplet = window.ggbApplet;
    if (!ggbApplet) {
      setStatus('Waiting for GeoGebra to initialize');
      return;
    }
    const baseVertices = baseVerticesRef.current;
    if (!baseVertices || baseVertices.length !== 3) {
      setStatus('Please draw a triangle first');
      return;
    }

    setIsProcessing(true);
    try {
      const strokeObjects = getAllStrokes(ggbApplet);
      const { digitStrokeObjects } = classifyStrokes(strokeObjects, baseVertices);
      const digitGroups = groupDigitStrokes(digitStrokeObjects);
      if (digitGroups.length !== 3) {
        throw new Error(`Invalid number of digit groups (${digitGroups.length}). Expected 3.`);
      }

      const recognizedGroups = [];
      for (let i = 0; i < digitGroups.length; i++) {
        // Recognize strokes left-to-right within each group to form multi-digit numbers
        recognizedGroups.push(await recognizeDigitGroup(digitGroups[i]));
      }

      const associated = associateDigitsToEdges(recognizedGroups, baseVertices);
      const v01 = associated.edgeValues[0];
      const v12 = associated.edgeValues[1];
      const v20 = associated.edgeValues[2];
      updateResultFromEdges(v01, v12, v20);
      drawRecognizedEdgeLabels(ggbApplet, baseVertices, [v01, v12, v20]);
      if (v01 <= 0 || v12 <= 0 || v20 <= 0) {
        throw new Error('All recognized ratios must be positive');
      }

      // adjustTriangleToRatios args: (a,b,c) = (V1V2, V0V2, V0V1)
      const adjusted = adjustTriangleToRatios(baseVertices, v12, v20, v01);
      clearDetectedObjects(ggbApplet);
      drawTriangleOnCanvas(ggbApplet, adjusted);
      drawRecognizedEdgeLabels(ggbApplet, adjusted, [v01, v12, v20]);
      hideStrokeObjects(ggbApplet, strokeObjects);
      setDetectedVertices(adjusted);
      isTriangleLockedRef.current = true;
      setStatus('Triangle adjusted using ratios from handwritten digits');
    } catch (error) {
      setStatus(`Adjust failed: ${escapeErrorMessage(error)}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const clearCanvas = () => {
    clearAllObjects(window.ggbApplet);
    setDetectedVertices(null);
    setResult(null);
    setStatus('Canvas cleared');
    lastSignatureRef.current = '';
    lastProcessedSignatureRef.current = '';
    baseVerticesRef.current = null;
    isTriangleLockedRef.current = false;
    lastChangedAtRef.current = Date.now();
  };

  return (
    <section style={{ marginTop: 12, padding: 12, border: '1px solid #ccc' }}>
      <h2 style={{ marginTop: 0 }}>Triangle Adjustment</h2>
      <p style={{ marginTop: 0 }}>
        After triangle detection, enter digits near each side and click <strong>Adjust</strong> to reshape the triangle to the given ratios.
      </p>
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="button" onClick={adjustByDigits} disabled={isProcessing}>
          {isProcessing ? 'Adjusting...' : 'Adjust'}
        </button>
        <button type="button" onClick={clearCanvas}>Clear</button>
      </div>
      <p style={{ marginBottom: 4 }}>Status: {status}</p>
      <p style={{ marginTop: 0, marginBottom: 4 }}>
        Result: {result ? result.text : '-'}
      </p>
      <p style={{ marginTop: 0, marginBottom: 0 }}>Vertices: {formatVertices(detectedVertices)}</p>
    </section>
  );
}
