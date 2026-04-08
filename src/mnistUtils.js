export function getBBox(strokes) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let s = 0; s < strokes.length; s++) {
    const pts = strokes[s];
    for (let p = 0; p < pts.length; p++) {
      const xx = pts[p][0];
      const yy = pts[p][1];
      if (xx < minX) minX = xx;
      if (yy < minY) minY = yy;
      if (xx > maxX) maxX = xx;
      if (yy > maxY) maxY = yy;
    }
  }

  const bw = (maxX - minX) || 1e-9;
  const bh = (maxY - minY) || 1e-9;
  return { minX, minY, maxX, maxY, bw, bh };
}

export function rasterize28(strokes, opts) {
  const W = 28;
  const H = 28;
  const { target, lineWidth, stepPerPixel } = opts;
  const { minX, minY, maxX, maxY, bw, bh } = getBBox(strokes);

  const scale = target / Math.max(bw, bh);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const outCx = (W - 1) / 2;
  const outCy = (H - 1) / 2;
  const img = Array.from({ length: H }, () => Array(W).fill(0));

  const r = lineWidth / 2;
  function splat(x, y) {
    const x0 = Math.floor(x - r - 1);
    const x1 = Math.ceil(x + r + 1);
    const y0 = Math.floor(y - r - 1);
    const y1 = Math.ceil(y + r + 1);

    for (let j = y0; j <= y1; j++) {
      if (j < 0 || j >= H) continue;
      for (let i2 = x0; i2 <= x1; i2++) {
        if (i2 < 0 || i2 >= W) continue;
        const dx = (i2 + 0.5) - x;
        const dy = (j + 0.5) - y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d > r + 1) continue;
        let val = 0;
        if (d <= r) val = 255;
        else val = Math.max(0, 255 * (1 - (d - r)));
        if (val > img[j][i2]) img[j][i2] = Math.round(val);
      }
    }
  }

  function mapPoint(x, y) {
    return [(x - cx) * scale + outCx, -(y - cy) * scale + outCy];
  }

  for (let si = 0; si < strokes.length; si++) {
    const pts = strokes[si];
    if (pts.length === 1) {
      const q = mapPoint(pts[0][0], pts[0][1]);
      splat(q[0], q[1]);
      continue;
    }

    for (let k = 0; k < pts.length - 1; k++) {
      const a = mapPoint(pts[k][0], pts[k][1]);
      const b = mapPoint(pts[k + 1][0], pts[k + 1][1]);
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const dist = Math.sqrt(dx * dx + dy * dy);
      const steps = Math.max(1, Math.ceil(dist * stepPerPixel));
      for (let t = 0; t <= steps; t++) {
        const u = t / steps;
        splat(a[0] + dx * u, a[1] + dy * u);
      }
    }
  }

  return img;
}

export function shiftToCenterOfMass(img28) {
  const H = 28;
  const W = 28;
  let sum = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = img28[y][x];
      sum += v;
      sumX += x * v;
      sumY += y * v;
    }
  }

  if (sum === 0) return img28;

  const cx = sumX / sum;
  const cy = sumY / sum;
  const target = 13.5;
  const dx = Math.round(target - cx);
  const dy = Math.round(target - cy);

  const out = Array.from({ length: H }, () => Array(W).fill(0));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const ny = y + dy;
      const nx = x + dx;
      if (0 <= ny && ny < H && 0 <= nx && nx < W) {
        out[ny][nx] = img28[y][x];
      }
    }
  }
  return out;
}

export function flatten28(img) {
  const out = [];
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) out.push(img[y][x]);
  }
  return out;
}

export function getStrokes(ggbApplet, strokeName = 'stroke1') {
  if (!ggbApplet || !ggbApplet.exists(strokeName)) return null;

  const xml = ggbApplet.getXML(strokeName);
  const m = xml.match(/<strokeCoords[^>]*val="([^"]*)"/);
  if (!m) return null;

  const arr = m[1].split(',').map((v) => parseFloat(v));
  const strokes = [];
  let cur = [];

  for (let i = 0; i + 1 < arr.length; i += 2) {
    const x = arr[i];
    const y = arr[i + 1];
    if (!isFinite(x) || !isFinite(y)) {
      if (cur.length > 0) strokes.push(cur);
      cur = [];
    } else {
      cur.push([x, y]);
    }
  }
  if (cur.length > 0) strokes.push(cur);
  return strokes.length > 0 ? strokes : null;
}

export function clearAllObjects(ggbApplet = window.ggbApplet) {
  if (!ggbApplet) return;
  const n = ggbApplet.getObjectNumber();
  for (let i = n - 1; i >= 0; i--) {
    const name = ggbApplet.getObjectName(i);
    ggbApplet.deleteObject(name);
  }
}

export function strokeToMnistPixels(ggbApplet = window.ggbApplet) {
  const strokes = getStrokes(ggbApplet, 'stroke1');
  return strokesToMnistPixels(strokes);
}

// Convert an array of strokes to a 784-element pixel array
export function strokesToMnistPixels(strokes) {
  if (!strokes || strokes.length === 0) return null;

  const opts = {
    target: 20,
    lineWidth: 1.8,
    stepPerPixel: 3.0,
  };

  let img = rasterize28(strokes, opts);
  img = shiftToCenterOfMass(img);
  return flatten28(img);
}

export function drawStrokePolylines(ggbApplet = window.ggbApplet, debugName = 'err') {
  if (!ggbApplet) return { ok: false, error: 'GeoGebra not initialized' };

  if (!ggbApplet.exists(debugName)) {
    ggbApplet.evalCommand(`${debugName} = ""`);
  }

  const strokes = getStrokes(ggbApplet, 'stroke1');
  if (!strokes) {
    ggbApplet.setTextValue(debugName, 'No strokes found');
    return { ok: false, error: 'No strokes found' };
  }

  ggbApplet.setTextValue(debugName, JSON.stringify(strokes));
  ggbApplet.setCoords(debugName, -6, -4);

  for (let si = 0; si < strokes.length; si++) {
    const pts = strokes[si];
    let ggList = '{';
    for (let k = 0; k < pts.length; k++) {
      ggList += `(${pts[k][0]},${pts[k][1]})`;
      if (k < pts.length - 1) ggList += ',';
    }
    ggList += '}';

    const lineName = `L${si + 1}`;
    const polyName = `P${si + 1}`;
    ggbApplet.evalCommand(`${lineName} = ${ggList}`);
    ggbApplet.evalCommand(`${polyName} = PolyLine(${lineName})`);
  }

  return { ok: true };
}
