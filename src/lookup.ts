import type { Dataset, Feature } from './data';

const ALIASES: Record<string, string> = {
  arraybuffer: 'typedarrays', 'typed arrays': 'typedarrays', 'es6 classes': 'es6-class', classes: 'es6-class',
  'css grid': 'css-grid', grid: 'css-grid', flexbox: 'flexbox', 'service worker': 'serviceworkers',
  'web components': 'custom-elementsv1', fetch: 'fetch', promises: 'promises', 'arrow functions': 'arrow-functions',
  webgpu: 'webgpu', webgl: 'webgl', 'web assembly': 'wasm', webassembly: 'wasm', 'container queries': 'css-container-queries',
};

function norm(s: string) { return s.toLowerCase().replace(/[^a-z0-9+ ]/g, ' ').replace(/\s+/g, ' ').trim(); }

/** Fuzzy feature search: returns best matches first. */
export function findFeatures(ds: Dataset, query: string, limit = 5): { index: number; feature: Feature; score: number }[] {
  const q = norm(query);
  const alias = ALIASES[q];
  const words = q.split(' ').filter(Boolean);
  const out: { index: number; feature: Feature; score: number }[] = [];
  ds.features.forEach((f, index) => {
    const title = norm(f.title), id = f.id.toLowerCase();
    let score = 0;
    if (alias && id === alias) score = 100;
    else if (title === q || id === q) score = 90;
    else if (title.includes(q) || id.includes(q.replace(/ /g, '-'))) score = 70 - Math.abs(title.length - q.length) * 0.2;
    else {
      const hits = words.filter((w) => title.includes(w) || id.includes(w)).length;
      if (hits) score = 20 + (40 * hits) / words.length - Math.abs(title.length - q.length) * 0.1;
    }
    if (score > 0) out.push({ index, feature: f, score });
  });
  return out.sort((a, b) => b.score - a.score).slice(0, limit);
}

const CODE = ['n/a', 'no', 'partial', 'yes'];

/** Facts about a feature for the assistant: ship dates per browser, support now and at a given month. */
export function describeFeature(ds: Dataset, fi: number, atMonth: number) {
  const f = ds.features[fi];
  const M = ds.months.length, B = ds.browsers.length;
  const fmt = (m: number) => new Date(ds.months[m]).toISOString().slice(0, 7);
  const browsers = ds.browsers.map((b, bi) => {
    let first = -1, firstPartial = -1;
    for (let m = 0; m < M; m++) {
      const c = ds.browserStatus[(fi * M + m) * B + bi];
      if (firstPartial < 0 && c >= 2) firstPartial = m;
      if (c === 3) { first = m; break; }
    }
    return {
      browser: b.name,
      first_full_support: first >= 0 ? fmt(first) : null,
      first_partial_support: firstPartial >= 0 ? fmt(firstPartial) : null,
      status_at_current_time: CODE[ds.browserStatus[(fi * M + Math.min(M - 1, Math.max(0, Math.floor(atMonth)))) * B + bi]],
      status_today: CODE[ds.browserStatus[(fi * M + (M - 1)) * B + bi]],
    };
  });
  return {
    id: f.id,
    title: f.title,
    category: f.group,
    spec_status: f.status,
    first_shipped: f.firstMonth >= 0 ? fmt(f.firstMonth) : null,
    support_share_today_percent: Math.round((ds.support[fi * M + M - 1] / 200) * 100),
    global_usage_percent: f.usage,
    caniuse_url: `https://caniuse.com/${f.id}`,
    browsers,
  };
}
