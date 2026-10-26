// Builds public/data/{features.json,support.bin,browsers.bin} from caniuse-db.
// support.bin: Uint8 [feature][month], value 0..200 = support share * 200
// browsers.bin: Uint8 [feature][month][browser], 0 = browser not released yet, 1 = no, 2 = partial, 3 = yes
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const db = JSON.parse(readFileSync(require.resolve('caniuse-db/fulldata-json/data-2.0.json'), 'utf8'));

// Browser lineages used for the support share. IE and Edge are one lineage (Edge replaces IE).
const LINEAGES = [
  { id: 'ms', name: 'IE / Edge', agents: ['ie', 'edge'] },
  { id: 'chrome', name: 'Chrome', agents: ['chrome'] },
  { id: 'firefox', name: 'Firefox', agents: ['firefox'] },
  { id: 'safari', name: 'Safari', agents: ['safari'] },
  { id: 'ios_saf', name: 'Safari on iOS', agents: ['ios_saf'] },
  { id: 'opera', name: 'Opera', agents: ['opera'] },
  { id: 'samsung', name: 'Samsung Internet', agents: ['samsung'] },
];

const CATEGORY_GROUP = {
  CSS: 'CSS', CSS2: 'CSS', CSS3: 'CSS',
  HTML5: 'HTML', Canvas: 'HTML',
  JS: 'JavaScript',
  'JS API': 'Web APIs', DOM: 'Web APIs',
  SVG: 'Graphics & Media', PNG: 'Graphics & Media',
  Security: 'Security',
  Other: 'Other',
};
const GROUPS = ['CSS', 'HTML', 'JavaScript', 'Web APIs', 'Graphics & Media', 'Security', 'Other'];

// Release timeline per lineage: sorted [{ t, agent, version }]
const releases = {};
for (const lin of LINEAGES) {
  const list = [];
  for (const agent of lin.agents) {
    for (const v of db.agents[agent].version_list) {
      if (v.release_date) list.push({ t: v.release_date * 1000, agent, version: v.version });
    }
  }
  list.sort((a, b) => a.t - b.t);
  releases[lin.id] = list;
}

const allDates = Object.values(releases).flat().map((r) => r.t);
const start = new Date(2000, 0, 1);
const endDate = new Date(Math.max(...allDates));
const end = new Date(endDate.getFullYear(), endDate.getMonth() + 1, 1);
const months = [];
for (let d = new Date(start); d < end; d.setMonth(d.getMonth() + 1)) months.push(new Date(d));
const M = months.length;

function statusCode(s) {
  if (!s) return 1;
  const c = s.trim()[0];
  if (c === 'y') return 3;
  if (c === 'a') return 2;
  return 1;
}

// Current version per lineage per month (index into releases[lin])
const currentAt = {};
for (const lin of LINEAGES) {
  const arr = new Int32Array(M).fill(-1);
  const rel = releases[lin.id];
  let i = -1;
  for (let m = 0; m < M; m++) {
    const t = months[m].getTime();
    while (i + 1 < rel.length && rel[i + 1].t <= t) i++;
    arr[m] = i;
  }
  currentAt[lin.id] = arr;
}

const ids = Object.keys(db.data).sort();
const F = ids.length;
const support = new Uint8Array(F * M);
const browsers = new Uint8Array(F * M * LINEAGES.length);
const features = [];

ids.forEach((id, fi) => {
  const f = db.data[id];
  let firstMonth = -1;
  for (let m = 0; m < M; m++) {
    let num = 0, den = 0;
    LINEAGES.forEach((lin, li) => {
      const ri = currentAt[lin.id][m];
      let code = 0;
      if (ri >= 0) {
        const r = releases[lin.id][ri];
        code = statusCode(f.stats[r.agent]?.[r.version]);
        den++;
        num += code === 3 ? 1 : code === 2 ? 0.5 : 0;
      }
      browsers[(fi * M + m) * LINEAGES.length + li] = code;
    });
    const share = den ? num / den : 0;
    support[fi * M + m] = Math.round(share * 200);
    if (firstMonth < 0 && share > 0) firstMonth = m;
  }
  features.push({
    id,
    title: f.title,
    group: CATEGORY_GROUP[f.categories[0]] ?? 'Other',
    status: f.status,
    firstMonth,
    usage: f.usage_perc_y,
  });
});

// ---- Placement on the sphere: compact category blobs ----
function fibonacciSphere(n, offset = 0) {
  const pts = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const th = golden * i + offset;
    pts.push([Math.cos(th) * r, y, Math.sin(th) * r]);
  }
  return pts;
}
const counts = Object.fromEntries(GROUPS.map((g) => [g, 0]));
for (const f of features) counts[f.group]++;
const centres = fibonacciSphere(GROUPS.length, 0.7);
const points = fibonacciSphere(F, 0);
// greedy capacity-constrained assignment
const pairs = [];
points.forEach((p, pi) => centres.forEach((c, ci) => {
  pairs.push({ pi, ci, d: -(p[0] * c[0] + p[1] * c[1] + p[2] * c[2]) });
}));
pairs.sort((a, b) => a.d - b.d);
const cap = GROUPS.map((g) => counts[g]);
const pointGroup = new Int32Array(F).fill(-1);
for (const { pi, ci } of pairs) {
  if (pointGroup[pi] >= 0 || cap[ci] <= 0) continue;
  pointGroup[pi] = ci;
  cap[ci]--;
}
const buckets = GROUPS.map(() => []);
pointGroup.forEach((g, pi) => buckets[g].push(pi));
// assign features (sorted by first support month so neighbours are chronological-ish)
const byGroup = GROUPS.map((g) => features.map((f, i) => [f, i]).filter(([f]) => f.group === g)
  .sort((a, b) => (a[0].firstMonth < 0 ? 1e9 : a[0].firstMonth) - (b[0].firstMonth < 0 ? 1e9 : b[0].firstMonth)));
byGroup.forEach((list, gi) => {
  // shuffle bucket deterministically so chronological order spreads spatially
  const b = buckets[gi];
  let seed = 42 + gi;
  for (let i = b.length - 1; i > 0; i--) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const j = seed % (i + 1);
    [b[i], b[j]] = [b[j], b[i]];
  }
  list.forEach(([f], k) => { f.pos = points[b[k]].map((v) => +v.toFixed(5)); });
});

const meta = {
  updated: db.updated * 1000,
  months: months.map((d) => d.getTime()),
  groups: GROUPS,
  browsers: LINEAGES.map((l) => ({ id: l.id, name: l.name })),
  features,
};
writeFileSync('public/data/features.json', JSON.stringify(meta));
writeFileSync('public/data/support.bin', support);
writeFileSync('public/data/browsers.bin', browsers);
console.log(`features=${F} months=${M} (${months[0].toISOString().slice(0, 7)} .. ${months[M - 1].toISOString().slice(0, 7)})`);
console.log('group counts', counts);
console.log('features with no support ever:', features.filter((f) => f.firstMonth < 0).length);
