// Head pose over the W3C Web Neural Network API: YuNet (face box + 5 landmarks) via ONNX Runtime Web's
// WebNN execution provider, preferring the NPU. Yaw / pitch / roll are estimated from the landmarks.
import * as ort from 'onnxruntime-web/all';
import type { FacePose, FaceTracker } from './headpose';

const W = 640, H = 640;
const STRIDES = [8, 16, 32];

export function webnnAvailable() { return 'ml' in navigator; }

async function createSession(allowFallback: boolean) {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  if (import.meta.env.PROD) ort.env.wasm.wasmPaths = `${base}/ort/`;
  ort.env.logLevel = 'error';
  const url = `${base}/models/face_detection_yunet_2023mar.onnx`;

  // The model has symbolic input dims; WebNN needs static shapes, so read the names first.
  const probe = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  const dims = (probe.inputMetadata[0] as { shape?: (number | string)[] }).shape ?? [];
  const freeDimensionOverrides: Record<string, number> = {};
  const wanted = [1, 3, H, W];
  dims.forEach((d, i) => { if (typeof d === 'string') freeDimensionOverrides[d] = wanted[i]; });
  const inputName = probe.inputNames[0];
  const outputNames = probe.outputNames;

  const attempts: { name: string; eps: ort.InferenceSession.ExecutionProviderConfig[] }[] = [];
  if (webnnAvailable()) {
    for (const deviceType of ['npu', 'gpu', 'cpu'] as const) {
      attempts.push({ name: `WebNN · ${deviceType.toUpperCase()}`, eps: [{ name: 'webnn', deviceType } as ort.InferenceSession.ExecutionProviderConfig] });
    }
  }
  if (allowFallback) attempts.push({ name: 'WASM (no WebNN)', eps: ['wasm'] });

  for (const a of attempts) {
    let s: ort.InferenceSession | null = null;
    try {
      s = await ort.InferenceSession.create(url, { executionProviders: a.eps, freeDimensionOverrides });
      await s.run({ [inputName]: new ort.Tensor('float32', new Float32Array(3 * W * H), [1, 3, H, W]) });
      if (s !== probe) await probe.release();
      return { session: s, backend: a.name, inputName, outputNames };
    } catch (e) {
      console.warn(`[webnn-face] ${a.name} unavailable`, e);
      await s?.release().catch(() => {});
    }
  }
  await probe.release();
  throw new Error('WebNN is not available in this browser');
}

export async function startWebNNTracker(onPose: (p: FacePose | null) => void, allowFallback = false): Promise<FaceTracker> {
  const { session, backend, inputName, outputNames } = await createSession(allowFallback);

  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const input = new Float32Array(3 * W * H);
  const plane = W * H;

  let running = true;
  const tracker: FaceTracker = {
    backend, video, pose: null,
    stop() { running = false; stream.getTracks().forEach((t) => t.stop()); session.release().catch(() => {}); },
  };

  const out = (r: ort.InferenceSession.ReturnType, name: string) => r[name].data as Float32Array;
  // Detection runs at most ~25 Hz so rendering keeps its frame budget; the pose is exponentially smoothed.
  const MIN_INTERVAL = 40;
  const SMOOTH = 0.35;
  let smoothed: FacePose | null = null;
  const loop = async () => {
    while (running) {
      const t0 = performance.now();
      // letterbox: draw the 4:3 frame into the 320x320 square, keeping aspect ratio (vertical padding)
      const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
      const scale = W / vw, dh = vh * scale, dy = (H - dh) / 2;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
      ctx.drawImage(video, 0, dy, W, dh);
      const { data } = ctx.getImageData(0, 0, W, H);
      const r = input, g = input.subarray(plane), b = input.subarray(2 * plane);
      for (let i = 0, j = 0; i < plane; i++, j += 4) { // BGR, raw 0..255 as OpenCV feeds it
        r[i] = data[j + 2]; g[i] = data[j + 1]; b[i] = data[j];
      }
      let res: ort.InferenceSession.ReturnType;
      try {
        res = await session.run({ [inputName]: new ort.Tensor('float32', input, [1, 3, H, W]) });
      } catch (e) { console.error('[webnn-face] run failed', e); break; }

      let best = 0.6, bx = 0, by = 0, bw = 0, kps: number[] | null = null;
      for (const s of STRIDES) {
        const cols = W / s, rows = H / s;
        const cls = out(res, `cls_${s}`), obj = out(res, `obj_${s}`), bbox = out(res, `bbox_${s}`), kp = out(res, `kps_${s}`);
        for (let i = 0; i < rows * cols; i++) {
          const score = Math.sqrt(Math.max(0, cls[i]) * Math.max(0, obj[i]));
          if (score <= best) continue;
          const row = Math.floor(i / cols), col = i % cols;
          best = score;
          bx = (col + bbox[i * 4]) * s; by = (row + bbox[i * 4 + 1]) * s; bw = Math.exp(bbox[i * 4 + 2]) * s;
          kps = [];
          for (let k = 0; k < 5; k++) kps.push((col + kp[i * 10 + k * 2]) * s, (row + kp[i * 10 + k * 2 + 1]) * s);
        }
      }
      if (kps) {
        const [rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy] = kps;
        const ex = (rex + lex) / 2, ey = (rey + ley) / 2;
        const mx = (rmx + lmx) / 2, my = (rmy + lmy) / 2;
        const eyeDist = Math.hypot(lex - rex, ley - rey) || 1;
        const faceH = Math.hypot(mx - ex, my - ey) || 1;
        // nose offset from the eye midline -> yaw; nose position between eyes and mouth -> pitch; eye tilt -> roll
        const yaw = Math.atan2(nx - ex, eyeDist * 0.9);
        const pitch = -((ny - ey) / faceH - 0.55) * 2.0; // nose rising toward the eyes = looking up
        const roll = Math.atan2(ley - rey, lex - rex);
        const raw: FacePose = { x: (bx / W) * 2 - 1, y: ((by - dy) / dh) * 2 - 1, size: bw / W, yaw, pitch, roll };
        if (!smoothed) smoothed = { ...raw };
        else for (const k of Object.keys(raw) as (keyof FacePose)[]) smoothed[k] += (raw[k] - smoothed[k]) * SMOOTH;
        tracker.pose = { ...smoothed };
      } else { tracker.pose = null; smoothed = null; }
      onPose(tracker.pose);
      const wait = MIN_INTERVAL - (performance.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      else await new Promise((r) => requestAnimationFrame(r));
    }
  };
  loop();
  return tracker;
}
