export interface Feature {
  id: string;
  title: string;
  group: string;
  status: string;
  firstMonth: number;
  usage: number;
  pos: [number, number, number];
}

export interface Dataset {
  updated: number;
  months: number[];
  groups: string[];
  browsers: { id: string; name: string }[];
  features: Feature[];
  /** Uint8 [feature][month], 0..200 */
  support: Uint8Array;
  /** Uint8 [feature][month][browser]: 0 n/a, 1 no, 2 partial, 3 yes */
  browserStatus: Uint8Array;
}

export async function loadDataset(): Promise<Dataset> {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const [meta, support, browsers] = await Promise.all([
    fetch(`${base}/data/features.json`).then((r) => r.json()),
    fetch(`${base}/data/support.bin`).then((r) => r.arrayBuffer()),
    fetch(`${base}/data/browsers.bin`).then((r) => r.arrayBuffer()),
  ]);
  return { ...meta, support: new Uint8Array(support), browserStatus: new Uint8Array(browsers) };
}

/** Support share (0..1) of a feature at a fractional month index. */
export function supportAt(ds: Dataset, fi: number, m: number): number {
  const M = ds.months.length;
  const m0 = Math.max(0, Math.min(M - 1, Math.floor(m)));
  const m1 = Math.min(M - 1, m0 + 1);
  const t = m - m0;
  const a = ds.support[fi * M + m0];
  const b = ds.support[fi * M + m1];
  return (a + (b - a) * t) / 200;
}

/** Convert a time (ms) to a fractional month index. */
export function monthIndexAt(ds: Dataset, time: number): number {
  const months = ds.months;
  if (time <= months[0]) return 0;
  const last = months.length - 1;
  if (time >= months[last]) return last;
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (months[mid] <= time) lo = mid; else hi = mid;
  }
  return lo + (time - months[lo]) / (months[hi] - months[lo]);
}
