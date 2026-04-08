const DEFAULT_SERVER_URL = 'http://localhost:8001';

function normalizeServerUrl(serverUrl) {
  return (serverUrl || DEFAULT_SERVER_URL).replace(/\/+$/, '');
}

export async function checkFastApiHealth(serverUrl = DEFAULT_SERVER_URL) {
  const base = normalizeServerUrl(serverUrl);
  const res = await fetch(`${base}/health`);
  if (!res.ok) {
    throw new Error(`FastAPI health check failed: ${res.status}`);
  }
  return res.json();
}

export async function fastApiPredict(pixels, serverUrl = DEFAULT_SERVER_URL) {
  const base = normalizeServerUrl(serverUrl);
  const res = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pixels }),
  });

  if (!res.ok) {
    let detail = '';
    try {
      const err = await res.json();
      detail = err?.detail ? `: ${err.detail}` : '';
    } catch {
      detail = '';
    }
    throw new Error(`FastAPI predict failed: ${res.status}${detail}`);
  }

  return res.json();
}
