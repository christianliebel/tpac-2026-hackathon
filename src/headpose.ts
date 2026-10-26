// Head pose with MediaPipe Face Landmarker: yields the facial transformation matrix,
// from which yaw / pitch / roll are derived.
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Euler, Matrix4 } from 'three/webgpu';
export interface FacePose {
  /** horizontal centre, -1 (left of image) .. 1 (right) */
  x: number;
  /** vertical centre, -1 (top) .. 1 (bottom) */
  y: number;
  /** face width relative to frame width, 0..1 */
  size: number;
  /** head rotation in radians */
  yaw: number;
  pitch: number;
  roll: number;
}

export interface FaceTracker {
  backend: string;
  video: HTMLVideoElement;
  pose: FacePose | null;
  stop(): void;
}

const W = 640, H = 480;

export async function startHeadPoseTracker(onPose: (p: FacePose | null) => void): Promise<FaceTracker> {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const vision = await FilesetResolver.forVisionTasks(`${base}/mediapipe/wasm`);
  let landmarker: FaceLandmarker;
  let backend = 'MediaPipe · GPU';
  const options = {
    runningMode: 'VIDEO' as const,
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
    outputFaceBlendshapes: false,
  };
  try {
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: `${base}/models/face_landmarker.task`, delegate: 'GPU' }, ...options,
    });
  } catch (e) {
    console.warn('[headpose] GPU delegate unavailable, using CPU', e);
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: `${base}/models/face_landmarker.task`, delegate: 'CPU' }, ...options,
    });
    backend = 'MediaPipe · CPU';
  }

  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: W, height: H, facingMode: 'user' }, audio: false });
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  let running = true;
  const tracker: FaceTracker = {
    backend, video, pose: null,
    stop() { running = false; stream.getTracks().forEach((t) => t.stop()); landmarker.close(); },
  };

  const m = new Matrix4();
  const e = new Euler();
  let lastT = -1;
  const loop = () => {
    if (!running) return;
    const t = performance.now();
    if (video.readyState >= 2 && t !== lastT) {
      lastT = t;
      const res = landmarker.detectForVideo(video, t);
      const lm = res.faceLandmarks[0];
      const mat = res.facialTransformationMatrixes?.[0];
      if (lm && mat) {
        m.fromArray(mat.data); // column-major 4x4
        e.setFromRotationMatrix(m, 'YXZ');
        // bounding box from landmarks for the preview
        let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
        for (const p of lm) { if (p.x < x1) x1 = p.x; if (p.x > x2) x2 = p.x; if (p.y < y1) y1 = p.y; if (p.y > y2) y2 = p.y; }
        tracker.pose = { x: (x1 + x2) - 1, y: (y1 + y2) - 1, size: x2 - x1, yaw: e.y, pitch: e.x, roll: e.z };
      } else tracker.pose = null;
      onPose(tracker.pose);
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  return tracker;
}
