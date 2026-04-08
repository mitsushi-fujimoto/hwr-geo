function distance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.hypot(dx, dy);
}

function triangleArea(a, b, c) {
  return Math.abs(
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
  ) / 2;
}

function calcBBox(points) {
  if (!points || points.length === 0) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < points.length; i++) {
    const [x, y] = points[i];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  const width = maxX - minX;
  const height = maxY - minY;
  return {
    minX,
    minY,
    maxX,
    maxY,
    width,
    height,
    diagonal: Math.hypot(width, height),
  };
}

function parseStrokeCoords(xml) {
  if (!xml) return [];
  const m = xml.match(/<strokeCoords[^>]*val="([^"]*)"/);
  if (!m) return [];

  const nums = m[1].split(',').map((v) => Number.parseFloat(v.trim()));
  const subStrokes = [];
  let current = [];

  for (let i = 0; i + 1 < nums.length; i += 2) {
    const x = nums[i];
    const y = nums[i + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (current.length > 0) subStrokes.push(current);
      current = [];
      continue;
    }
    current.push([x, y]);
  }

  if (current.length > 0) subStrokes.push(current);
  return subStrokes;
}

function clusterPoints(points, threshold) {
  if (!points || points.length === 0) return [];
  const n = points.length;
  const parent = Array.from({ length: n }, (_, i) => i);

  function find(x) {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  function union(a, b) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distance(points[i], points[j]) <= threshold) {
        union(i, j);
      }
    }
  }

  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(points[i]);
  }

  const centroids = [];
  groups.forEach((group) => {
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < group.length; i++) {
      sx += group[i][0];
      sy += group[i][1];
    }
    centroids.push([sx / group.length, sy / group.length]);
  });
  return centroids;
}

function sortVertices(vertices) {
  if (!vertices || vertices.length !== 3) return vertices;
  const cx = (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3;
  const cy = (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3;
  return [...vertices].sort((a, b) => (
    Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx)
  ));
}

function pickMaxAreaTriangle(points) {
  if (!points || points.length < 3) return null;
  let best = null;
  let maxArea = -1;

  for (let i = 0; i < points.length - 2; i++) {
    for (let j = i + 1; j < points.length - 1; j++) {
      for (let k = j + 1; k < points.length; k++) {
        const area = triangleArea(points[i], points[j], points[k]);
        if (area > maxArea) {
          maxArea = area;
          best = [points[i], points[j], points[k]];
        }
      }
    }
  }

  return best && maxArea > 0 ? sortVertices(best) : null;
}

function detectCornersFromStroke(points, lookAhead = 4, angleThresholdDeg = 55) {
  if (!points || points.length < lookAhead * 2 + 1) return [];
  const angleThreshold = (angleThresholdDeg * Math.PI) / 180;
  const candidates = [];

  for (let i = lookAhead; i < points.length - lookAhead; i++) {
    const p0 = points[i - lookAhead];
    const p1 = points[i];
    const p2 = points[i + lookAhead];
    const v1x = p1[0] - p0[0];
    const v1y = p1[1] - p0[1];
    const v2x = p2[0] - p1[0];
    const v2y = p2[1] - p1[1];
    const n1 = Math.hypot(v1x, v1y);
    const n2 = Math.hypot(v2x, v2y);
    if (n1 < 1e-6 || n2 < 1e-6) continue;

    const dot = (v1x * v2x + v1y * v2y) / (n1 * n2);
    const clamped = Math.max(-1, Math.min(1, dot));
    const theta = Math.acos(clamped);
    if (theta >= angleThreshold) {
      candidates.push({
        point: p1,
        score: theta * Math.min(n1, n2),
        index: i,
      });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const selected = [];
  const minIndexGap = lookAhead * 2;
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const tooClose = selected.some((s) => Math.abs(s.index - c.index) < minIndexGap);
    if (!tooClose) selected.push(c);
  }

  return selected.map((c) => c.point);
}

function samplePoints(points, maxSamples = 60) {
  if (!points || points.length <= maxSamples) return points || [];
  const step = (points.length - 1) / (maxSamples - 1);
  const out = [];
  for (let i = 0; i < maxSamples; i++) {
    out.push(points[Math.round(i * step)]);
  }
  return out;
}

function edgeKey(i, j) {
  return i < j ? `${i}-${j}` : `${j}-${i}`;
}

function centroid(vertices) {
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < vertices.length; i++) {
    sx += vertices[i][0];
    sy += vertices[i][1];
  }
  return [sx / vertices.length, sy / vertices.length];
}

function transformCandidateToSource(candidate, sourceVertices, alignEdge) {
  const [si, sj] = alignEdge;
  const srcVec = [
    sourceVertices[sj][0] - sourceVertices[si][0],
    sourceVertices[sj][1] - sourceVertices[si][1],
  ];
  const tarVec = [
    candidate[sj][0] - candidate[si][0],
    candidate[sj][1] - candidate[si][1],
  ];
  const srcAngle = Math.atan2(srcVec[1], srcVec[0]);
  const tarAngle = Math.atan2(tarVec[1], tarVec[0]);
  const rot = srcAngle - tarAngle;
  const cosR = Math.cos(rot);
  const sinR = Math.sin(rot);

  const srcCenter = centroid(sourceVertices);
  const tarCenter = centroid(candidate);
  return candidate.map(([x, y]) => {
    const dx = x - tarCenter[0];
    const dy = y - tarCenter[1];
    const rx = dx * cosR - dy * sinR;
    const ry = dx * sinR + dy * cosR;
    return [rx + srcCenter[0], ry + srcCenter[1]];
  });
}

function alignmentError(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += distance(a[i], b[i]);
  }
  return sum;
}

export function getAllStrokes(ggbApplet = window.ggbApplet) {
  if (!ggbApplet) return [];
  const objectCount = ggbApplet.getObjectNumber();
  const out = [];

  for (let i = 0; i < objectCount; i++) {
    const name = ggbApplet.getObjectName(i);
    const xml = ggbApplet.getXML(name);
    const subStrokes = parseStrokeCoords(xml);
    if (subStrokes.length > 0) {
      out.push({ name, subStrokes });
    }
  }

  return out;
}

export function collectAllPoints(strokeObjects) {
  const flatPoints = [];
  const subStrokeRanges = [];

  if (!Array.isArray(strokeObjects)) {
    return { flatPoints, subStrokeRanges, bbox: null };
  }

  for (let i = 0; i < strokeObjects.length; i++) {
    const strokeObj = strokeObjects[i];
    const subStrokes = strokeObj?.subStrokes || [];
    for (let j = 0; j < subStrokes.length; j++) {
      const sub = subStrokes[j];
      if (!Array.isArray(sub) || sub.length === 0) continue;
      const startIndex = flatPoints.length;
      for (let k = 0; k < sub.length; k++) flatPoints.push(sub[k]);
      const endIndex = flatPoints.length - 1;
      subStrokeRanges.push({
        strokeName: strokeObj.name,
        subStrokeIndex: j,
        startIndex,
        endIndex,
      });
    }
  }

  return {
    flatPoints,
    subStrokeRanges,
    bbox: calcBBox(flatPoints),
  };
}

function distancePointToSegment(p, a, b) {
  const vx = b[0] - a[0];
  const vy = b[1] - a[1];
  const wx = p[0] - a[0];
  const wy = p[1] - a[1];
  const vv = vx * vx + vy * vy;
  if (vv < 1e-12) return distance(p, a);
  const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / vv));
  const px = a[0] + t * vx;
  const py = a[1] + t * vy;
  return Math.hypot(p[0] - px, p[1] - py);
}

function estimateEdgeCoverage(edgeStart, edgeEnd, points, tolerance, samples = 24) {
  if (!points || points.length === 0) return 0;
  let covered = 0;
  for (let i = 0; i < samples; i++) {
    const t = i / (samples - 1);
    const sx = edgeStart[0] + (edgeEnd[0] - edgeStart[0]) * t;
    const sy = edgeStart[1] + (edgeEnd[1] - edgeStart[1]) * t;
    let near = false;
    for (let j = 0; j < points.length; j++) {
      if (distance([sx, sy], points[j]) <= tolerance) {
        near = true;
        break;
      }
    }
    if (near) covered += 1;
  }
  return covered / samples;
}

export function hasTriangleEdgeCoverage(vertices, points, bbox) {
  if (!vertices || vertices.length !== 3 || !points || points.length === 0 || !bbox) return false;

  const [v0, v1, v2] = vertices;
  const area = triangleArea(v0, v1, v2);
  const bboxArea = Math.max(bbox.width * bbox.height, 1e-9);
  const areaRatio = area / bboxArea;
  if (areaRatio < 0.08) return false;

  const edges = [
    [v0, v1],
    [v1, v2],
    [v2, v0],
  ];
  const edgeLens = edges.map(([a, b]) => distance(a, b));
  const maxEdgeLen = Math.max(...edgeLens);
  const altitude = (2 * area) / Math.max(maxEdgeLen, 1e-9);
  if (altitude < bbox.diagonal * 0.06) return false;

  const tolerance = Math.max(0.08, bbox.diagonal * 0.04);
  const minCoverage = 0.55;

  for (let i = 0; i < edges.length; i++) {
    const [a, b] = edges[i];
    if (distance(a, b) < 1e-6) return false;
    const coverage = estimateEdgeCoverage(a, b, points, tolerance);
    if (coverage < minCoverage) return false;
  }

  const inlierTolerance = Math.max(0.1, bbox.diagonal * 0.06);
  let inliers = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const d01 = distancePointToSegment(p, v0, v1);
    const d12 = distancePointToSegment(p, v1, v2);
    const d20 = distancePointToSegment(p, v2, v0);
    if (Math.min(d01, d12, d20) <= inlierTolerance) inliers += 1;
  }
  return inliers / points.length >= 0.65;
}

export function extractTriangleVertices(strokeObjects) {
  if (!Array.isArray(strokeObjects) || strokeObjects.length === 0) return null;

  const endpoints = [];
  for (let i = 0; i < strokeObjects.length; i++) {
    const subStrokes = strokeObjects[i]?.subStrokes || [];
    for (let j = 0; j < subStrokes.length; j++) {
      const points = subStrokes[j];
      if (!points || points.length === 0) continue;
      endpoints.push(points[0]);
      endpoints.push(points[points.length - 1]);
    }
  }

  const { flatPoints, bbox } = collectAllPoints(strokeObjects);
  if (!bbox || flatPoints.length < 3) return null;

  const endpointThreshold = Math.max(0.1, bbox.diagonal * 0.1);
  let vertexCandidates = clusterPoints(endpoints, endpointThreshold);

  if (vertexCandidates.length === 3) {
    return sortVertices(vertexCandidates);
  }
  if (vertexCandidates.length > 3) {
    const best = pickMaxAreaTriangle(vertexCandidates);
    if (best) return best;
  }

  const cornerCandidates = [];
  for (let i = 0; i < strokeObjects.length; i++) {
    const subStrokes = strokeObjects[i]?.subStrokes || [];
    for (let j = 0; j < subStrokes.length; j++) {
      cornerCandidates.push(...detectCornersFromStroke(subStrokes[j]));
    }
  }

  if (cornerCandidates.length > 0) {
    const mergedCorners = clusterPoints(cornerCandidates, endpointThreshold);
    if (mergedCorners.length === 3) {
      return sortVertices(mergedCorners);
    }
    if (mergedCorners.length > 3) {
      const best = pickMaxAreaTriangle(mergedCorners);
      if (best) return best;
    }
  }

  vertexCandidates = samplePoints(flatPoints, 60);
  return pickMaxAreaTriangle(vertexCandidates);
}

function flattenSubStrokes(strokeObjects) {
  const out = [];
  if (!Array.isArray(strokeObjects)) return out;
  for (let i = 0; i < strokeObjects.length; i++) {
    const strokeObj = strokeObjects[i];
    const subs = strokeObj?.subStrokes || [];
    for (let j = 0; j < subs.length; j++) {
      const points = subs[j];
      if (!Array.isArray(points) || points.length === 0) continue;
      const bbox = calcBBox(points);
      if (!bbox) continue;
      out.push({
        strokeName: strokeObj.name,
        subStrokeIndex: j,
        points,
        bbox,
        centroid: [(bbox.minX + bbox.maxX) / 2, (bbox.minY + bbox.maxY) / 2],
      });
    }
  }
  return out;
}

function nearestVertexIndex(point, vertices, threshold) {
  let bestIndex = -1;
  let bestDist = Infinity;
  for (let i = 0; i < vertices.length; i++) {
    const d = distance(point, vertices[i]);
    if (d < bestDist) {
      bestDist = d;
      bestIndex = i;
    }
  }
  return bestDist <= threshold ? bestIndex : -1;
}

export function classifyStrokes(strokeObjects, triangleVertices) {
  const empty = {
    edgeStrokeObjects: [],
    digitStrokeObjects: [],
    edgeSubStrokes: [],
    digitSubStrokes: [],
  };
  if (!Array.isArray(strokeObjects) || strokeObjects.length === 0) return empty;
  if (!Array.isArray(triangleVertices) || triangleVertices.length !== 3) return empty;

  const entries = flattenSubStrokes(strokeObjects);
  const { bbox } = collectAllPoints(strokeObjects);
  if (!bbox || entries.length === 0) return empty;

  const vertexThreshold = Math.max(0.1, bbox.diagonal * 0.08);
  const edgeStrokeNameSet = new Set();

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const pts = entry.points;
    const start = pts[0];
    const end = pts[pts.length - 1];
    const startV = nearestVertexIndex(start, triangleVertices, vertexThreshold);
    const endV = nearestVertexIndex(end, triangleVertices, vertexThreshold);

    let visitedVertices = 0;
    for (let vi = 0; vi < triangleVertices.length; vi++) {
      let near = false;
      for (let pi = 0; pi < pts.length; pi++) {
        if (distance(pts[pi], triangleVertices[vi]) <= vertexThreshold) {
          near = true;
          break;
        }
      }
      if (near) visitedVertices += 1;
    }

    if ((startV >= 0 && endV >= 0 && startV !== endV) || visitedVertices >= 2) {
      edgeStrokeNameSet.add(entry.strokeName);
    }
  }

  const edgeStrokeObjects = [];
  const digitStrokeObjects = [];
  for (let i = 0; i < strokeObjects.length; i++) {
    const obj = strokeObjects[i];
    if (edgeStrokeNameSet.has(obj.name)) edgeStrokeObjects.push(obj);
    else digitStrokeObjects.push(obj);
  }

  const edgeSubStrokes = [];
  const digitSubStrokes = [];
  for (let i = 0; i < entries.length; i++) {
    if (edgeStrokeNameSet.has(entries[i].strokeName)) edgeSubStrokes.push(entries[i]);
    else digitSubStrokes.push(entries[i]);
  }

  return {
    edgeStrokeObjects,
    digitStrokeObjects,
    edgeSubStrokes,
    digitSubStrokes,
  };
}

export function groupDigitStrokes(digitObjects) {
  const entries = flattenSubStrokes(digitObjects);
  if (entries.length === 0) return [];

  let avgDiag = 0;
  for (let i = 0; i < entries.length; i++) avgDiag += entries[i].bbox.diagonal;
  avgDiag /= entries.length;
  const threshold = Math.max(0.15, avgDiag * 1.5);

  const n = entries.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x) {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  function union(a, b) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distance(entries[i].centroid, entries[j].centroid) <= threshold) {
        union(i, j);
      }
    }
  }

  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(entries[i]);
  }

  const out = [];
  groups.forEach((strokes) => {
    const sorted = [...strokes].sort((a, b) => a.centroid[0] - b.centroid[0]);
    const points = [];
    for (let i = 0; i < sorted.length; i++) {
      for (let j = 0; j < sorted[i].points.length; j++) {
        points.push(sorted[i].points[j]);
      }
    }
    const bbox = calcBBox(points);
    out.push({
      strokes: sorted,
      centroid: bbox ? [(bbox.minX + bbox.maxX) / 2, (bbox.minY + bbox.maxY) / 2] : [0, 0],
      bbox,
    });
  });

  out.sort((a, b) => a.centroid[0] - b.centroid[0]);
  return out;
}

export function associateDigitsToEdges(digitGroups, triangleVertices) {
  if (!Array.isArray(triangleVertices) || triangleVertices.length !== 3) {
    throw new Error('Invalid vertices');
  }
  if (!Array.isArray(digitGroups) || digitGroups.length !== 3) {
    throw new Error('Exactly 3 digit groups required');
  }

  const edges = [
    { index: 0, key: edgeKey(0, 1), vertices: [0, 1] },
    { index: 1, key: edgeKey(1, 2), vertices: [1, 2] },
    { index: 2, key: edgeKey(2, 0), vertices: [2, 0] },
  ];
  const edgeMidpoints = edges.map((edge) => {
    const v0 = triangleVertices[edge.vertices[0]];
    const v1 = triangleVertices[edge.vertices[1]];
    return [(v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2];
  });

  const edgeValues = [null, null, null];
  const assignments = [];
  for (let i = 0; i < digitGroups.length; i++) {
    const g = digitGroups[i];
    if (!Number.isFinite(g?.value)) {
      throw new Error('Invalid digit recognition result');
    }
    const gc = g.centroid;
    let bestEdge = -1;
    let bestDist = Infinity;
    for (let e = 0; e < edgeMidpoints.length; e++) {
      const d = distance(gc, edgeMidpoints[e]);
      if (d < bestDist) {
        bestDist = d;
        bestEdge = e;
      }
    }
    if (bestEdge < 0) throw new Error('Failed to associate digit with an edge');
    if (edgeValues[bestEdge] !== null) {
      throw new Error('Multiple digits assigned to the same edge');
    }

    edgeValues[bestEdge] = g.value;
    assignments.push({
      edgeIndex: bestEdge,
      edgeKey: edges[bestEdge].key,
      value: g.value,
      text: g.text || String(g.value),
    });
  }

  if (edgeValues.some((v) => v === null)) {
    throw new Error('Could not assign digits to all 3 edges');
  }

  return { edgeValues, assignments };
}

export function adjustTriangleToRatios(vertices, a, b, c) {
  // a = ratio for edge V1-V2, b = ratio for edge V2-V0, c = ratio for edge V0-V1
  if (!Array.isArray(vertices) || vertices.length !== 3) {
    throw new Error('Exactly 3 vertices required');
  }
  const ratios = [a, b, c].map((v) => Number(v));
  if (ratios.some((v) => !Number.isFinite(v) || v <= 0)) {
    throw new Error('All ratios must be positive');
  }

  // Triangle inequality check
  const sortedRatios = [...ratios].sort((x, y) => y - x);
  if (sortedRatios[0] >= sortedRatios[1] + sortedRatios[2]) {
    throw new Error('Does not satisfy the triangle inequality');
  }

  // Use edge correspondence as given (no re-mapping by length sorting)
  const l12 = Number(a);  // edge V1-V2
  const l20 = Number(b);  // edge V2-V0
  const l01 = Number(c);  // edge V0-V1

  // Original triangle edge lengths
  const srcLen01 = distance(vertices[0], vertices[1]);
  const srcLen12 = distance(vertices[1], vertices[2]);
  const srcLen20 = distance(vertices[2], vertices[0]);
  if (Math.max(srcLen01, srcLen12, srcLen20) < 1e-8) {
    throw new Error('Handwritten triangle is degenerate');
  }

  const sourcePerimeter = srcLen01 + srcLen12 + srcLen20;
  const targetPerimeter = l01 + l12 + l20;
  if (sourcePerimeter < 1e-8 || targetPerimeter < 1e-8) {
    throw new Error('Invalid perimeter');
  }
  const scale = sourcePerimeter / targetPerimeter;

  const x2 = (l20 * l20 + l01 * l01 - l12 * l12) / (2 * l01);
  const y2sq = l20 * l20 - x2 * x2;
  if (y2sq < -1e-8) {
    throw new Error('Cannot form a triangle');
  }
  const y2 = Math.sqrt(Math.max(0, y2sq));
  const baseCandidates = [
    [[0, 0], [l01, 0], [x2, y2]],
    [[0, 0], [l01, 0], [x2, -y2]],
  ];

  const scaledCandidates = baseCandidates.map((candidate) => (
    candidate.map(([x, y]) => [x * scale, y * scale])
  ));
  // Align to the longest original edge direction (for visual stability)
  const srcEdges = [
    { i: 0, j: 1, len: srcLen01 },
    { i: 1, j: 2, len: srcLen12 },
    { i: 2, j: 0, len: srcLen20 },
  ];
  const longestSrc = srcEdges.reduce((ea, eb) => (ea.len > eb.len ? ea : eb));
  const alignEdge = [longestSrc.i, longestSrc.j];
  const transformed = scaledCandidates.map((candidate) => (
    transformCandidateToSource(candidate, vertices, alignEdge)
  ));

  let best = transformed[0];
  let minError = alignmentError(best, vertices);
  for (let i = 1; i < transformed.length; i++) {
    const err = alignmentError(transformed[i], vertices);
    if (err < minError) {
      minError = err;
      best = transformed[i];
    }
  }
  return best;
}

export function drawTriangleOnCanvas(ggbApplet = window.ggbApplet, vertices) {
  if (!ggbApplet || !Array.isArray(vertices) || vertices.length !== 3) return false;
  const [v0, v1, v2] = vertices;
  const names = ['TA', 'TB', 'TC', 'TriResult'];
  const beforeNames = new Set();
  const beforeCount = ggbApplet.getObjectNumber();
  for (let i = 0; i < beforeCount; i++) {
    beforeNames.add(ggbApplet.getObjectName(i));
  }
  for (let i = 0; i < names.length; i++) {
    if (ggbApplet.exists(names[i])) ggbApplet.deleteObject(names[i]);
  }

  ggbApplet.evalCommand(`TA = (${v0[0]}, ${v0[1]})`);
  ggbApplet.evalCommand(`TB = (${v1[0]}, ${v1[1]})`);
  ggbApplet.evalCommand(`TC = (${v2[0]}, ${v2[1]})`);
  ggbApplet.evalCommand('TriResult = Polygon(TA, TB, TC)');
  ggbApplet.setColor('TriResult', 0, 102, 204);
  ggbApplet.setLineThickness('TriResult', 5);
  // Hide labels for TriResult and edge objects internally created by Polygon().
  const afterCount = ggbApplet.getObjectNumber();
  for (let i = 0; i < afterCount; i++) {
    const name = ggbApplet.getObjectName(i);
    if (!beforeNames.has(name)) {
      ggbApplet.setLabelVisible(name, false);
    }
  }
  // Set vertex captions A, B, C after the hide-all loop so they remain visible.
  const vtxLabels = ['A', 'B', 'C'];
  for (let i = 0; i < 3; i++) {
    const name = names[i];
    ggbApplet.setCaption(name, vtxLabels[i]);
    ggbApplet.setLabelStyle(name, 3);
    ggbApplet.setLabelVisible(name, true);
  }
  return true;
}
