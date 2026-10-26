import {
  BufferAttribute, Color, Mesh, MeshStandardMaterial, SphereGeometry, Vector3,
} from 'three/webgpu';
import { supportAt, type Dataset } from './data';

export const GROUP_COLORS: Record<string, string> = {
  CSS: '#3b82f6',
  HTML: '#f97316',
  JavaScript: '#facc15',
  'Web APIs': '#22c55e',
  'Graphics & Media': '#ec4899',
  Security: '#a855f7',
  Other: '#94a3b8',
};

const OCEAN = new Color('#0b2a55');
const OCEAN_DEEP = new Color('#061a3a');
const SNOW = new Color('#f8fafc');

const RADIUS = 1;
const PEAK_RADIUS = 0.095; // chord radius of a feature's mountain
const SHELF_RADIUS = 0.15; // chord radius of its flat base (forms continents with its neighbours)
const PEAK_HEIGHT = 0.15;
const SHELF_HEIGHT = 0.012;
const MAX_HEIGHT = PEAK_HEIGHT + SHELF_HEIGHT;

interface Influence {
  vertex: Int32Array;
  peak: Float32Array;
  shelf: Float32Array;
}

export class Planet {
  readonly mesh: Mesh;
  private readonly geometry: SphereGeometry;
  private readonly positions: Float32Array;
  private readonly base: Float32Array;
  private readonly colors: Float32Array;
  private readonly heights: Float32Array;
  private readonly colAcc: Float32Array;
  private readonly wAcc: Float32Array;
  private readonly influences: Influence[] = [];
  private readonly groupColors: Color[];
  readonly featureShare: Float32Array;

  constructor(private readonly ds: Dataset) {
    this.geometry = new SphereGeometry(RADIUS, 420, 210);
    const posAttr = this.geometry.getAttribute('position') as BufferAttribute;
    this.positions = posAttr.array as Float32Array;
    this.base = this.positions.slice();
    const n = posAttr.count;
    this.colors = new Float32Array(n * 3);
    this.geometry.setAttribute('color', new BufferAttribute(this.colors, 3));
    this.heights = new Float32Array(n);
    this.colAcc = new Float32Array(n * 3);
    this.wAcc = new Float32Array(n);
    this.featureShare = new Float32Array(ds.features.length);
    this.groupColors = ds.features.map((f) => new Color(GROUP_COLORS[f.group] ?? '#888'));

    this.precomputeInfluences();

    const material = new MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.05,
      flatShading: false,
    });
    this.mesh = new Mesh(this.geometry, material);
    this.update(0);
  }

  /** For each feature, find all vertices within its footprint and store a smooth weight. */
  private precomputeInfluences() {
    const n = this.heights.length;
    const base = this.base;
    const r2 = SHELF_RADIUS * SHELF_RADIUS;
    for (const f of this.ds.features) {
      const [fx, fy, fz] = f.pos;
      const idx: number[] = [];
      const peak: number[] = [];
      const shelf: number[] = [];
      for (let i = 0; i < n; i++) {
        const dx = base[i * 3] - fx, dy = base[i * 3 + 1] - fy, dz = base[i * 3 + 2] - fz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) {
          const d = Math.sqrt(d2);
          idx.push(i);
          // pointed peak with flaring sides
          const tp = Math.max(0, 1 - d / PEAK_RADIUS);
          peak.push(Math.pow(tp, 1.35));
          // soft plateau that blends into neighbouring features
          const ts = 1 - d / SHELF_RADIUS;
          shelf.push(ts * ts * (3 - 2 * ts));
        }
      }
      this.influences.push({ vertex: Int32Array.from(idx), peak: Float32Array.from(peak), shelf: Float32Array.from(shelf) });
    }
  }

  /** Rebuild heights and colours for a fractional month index. */
  update(month: number) {
    const { ds, heights, colAcc, wAcc, influences, featureShare, groupColors } = this;
    heights.fill(0);
    colAcc.fill(0);
    wAcc.fill(0);
    for (let fi = 0; fi < influences.length; fi++) {
      const s = supportAt(ds, fi, month);
      featureShare[fi] = s;
      if (s <= 0) continue;
      const inf = influences[fi];
      const col = groupColors[fi];
      // the peak grows steeply with support share; the shelf appears as soon as the feature ships
      const hp = PEAK_HEIGHT * Math.pow(s, 1.6);
      const hs = SHELF_HEIGHT * Math.min(1, s * 6);
      for (let k = 0; k < inf.vertex.length; k++) {
        const v = inf.vertex[k];
        const wp = inf.peak[k], ws = inf.shelf[k];
        heights[v] += hp * wp + hs * ws;
        const cw = ws * 0.5 + wp * (0.5 + s);
        colAcc[v * 3] += col.r * cw;
        colAcc[v * 3 + 1] += col.g * cw;
        colAcc[v * 3 + 2] += col.b * cw;
        wAcc[v] += cw;
      }
    }
    const { positions, base, colors } = this;
    const n = heights.length;
    const tmp = new Color();
    for (let i = 0; i < n; i++) {
      const h = Math.min(heights[i], MAX_HEIGHT * 1.5);
      const scale = RADIUS + h;
      positions[i * 3] = base[i * 3] * scale;
      positions[i * 3 + 1] = base[i * 3 + 1] * scale;
      positions[i * 3 + 2] = base[i * 3 + 2] * scale;

      if (wAcc[i] > 1e-5) {
        const inv = 1 / wAcc[i];
        tmp.setRGB(colAcc[i * 3] * inv, colAcc[i * 3 + 1] * inv, colAcc[i * 3 + 2] * inv);
        // coastline blend into ocean, darker lowlands, snowy caps on tall peaks
        const land = Math.min(1, h / 0.006);
        const rel = h / MAX_HEIGHT;
        tmp.multiplyScalar(0.45 + 0.75 * Math.min(1, rel * 1.4));
        const snow = Math.max(0, Math.min(1, (rel - 0.62) / 0.3));
        tmp.lerp(SNOW, snow * 0.85);
        const ocean = OCEAN.clone().lerp(OCEAN_DEEP, 0.3);
        ocean.lerp(tmp, land);
        colors[i * 3] = ocean.r; colors[i * 3 + 1] = ocean.g; colors[i * 3 + 2] = ocean.b;
      } else {
        colors[i * 3] = OCEAN_DEEP.r; colors[i * 3 + 1] = OCEAN_DEEP.g; colors[i * 3 + 2] = OCEAN_DEEP.b;
      }
    }
    (this.geometry.getAttribute('position') as BufferAttribute).needsUpdate = true;
    (this.geometry.getAttribute('color') as BufferAttribute).needsUpdate = true;
    this.geometry.computeVertexNormals();
  }

  /** Nearest shipped feature to a direction on the unit sphere, or -1. */
  nearestFeature(dir: Vector3): number {
    let best = -1, bestDot = Math.cos(0.1);
    const fs = this.ds.features;
    for (let i = 0; i < fs.length; i++) {
      if (this.featureShare[i] <= 0) continue;
      const p = fs[i].pos;
      const d = dir.x * p[0] + dir.y * p[1] + dir.z * p[2];
      if (d > bestDot) { bestDot = d; best = i; }
    }
    return best;
  }
}
