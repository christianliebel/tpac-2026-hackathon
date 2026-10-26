import {
  ACESFilmicToneMapping, AdditiveBlending, AmbientLight, BackSide, BufferGeometry, Color, DirectionalLight,
  Float32BufferAttribute, Group, HemisphereLight, Mesh, MeshBasicMaterial, MeshBasicNodeMaterial, PerspectiveCamera, Points,
  PointsMaterial, Quaternion, Raycaster, RingGeometry, Scene, Sphere, SphereGeometry, Vector2, Vector3, WebGPURenderer,
} from 'three/webgpu';
import { color, dot, normalView, positionViewDirection, pow, saturate } from 'three/tsl';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadDataset, type Dataset } from './data';
import { GROUP_COLORS, Planet } from './planet';
import { startHeadPoseTracker, type FacePose, type FaceTracker } from './headpose';
import { startWebNNTracker, webnnAvailable } from './webnn-face';
import { describeFeature, findFeatures } from './lookup';
import { startVoice, type VoiceSession } from './voice';

const LOOP_SECONDS = 75;
const END_HOLD_SECONDS = 4;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main() {
  const ds = await loadDataset();
  const canvas = $<HTMLCanvasElement>('scene');

  const renderer = new WebGPURenderer({ canvas, antialias: true });
  await renderer.init();
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  const backend = (renderer as unknown as { backend: { isWebGPUBackend?: boolean } }).backend;
  const isWebGPU = backend.isWebGPUBackend === true;
  $('renderer-badge').innerHTML = `Rendering with <b>${isWebGPU ? 'WebGPU' : 'WebGL 2'}</b> · three.js`;

  const scene = new Scene();
  scene.background = new Color('#04060f');
  const camera = new PerspectiveCamera(42, innerWidth / innerHeight, 0.1, 100);
  camera.position.set(0, 1.1, 4.0);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 1.6;
  controls.maxDistance = 8;
  controls.enablePan = false;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;

  // Lights
  const sun = new DirectionalLight('#fff4e0', 2.6);
  sun.position.set(4, 2.5, 2);
  scene.add(sun);
  scene.add(new HemisphereLight('#9ec5ff', '#1a0f2e', 0.5));
  scene.add(new AmbientLight('#ffffff', 0.18));

  // Planet
  const world = new Group();
  scene.add(world);
  const planet = new Planet(ds);
  world.add(planet.mesh);

  // Atmosphere glow (TSL fresnel on a back-facing shell)
  const atmoMat = new MeshBasicNodeMaterial();
  atmoMat.colorNode = color('#5fb3ff');
  atmoMat.opacityNode = pow(saturate(dot(normalView, positionViewDirection).mul(1.9)), 5.0).mul(0.55);
  atmoMat.transparent = true;
  atmoMat.side = BackSide;
  atmoMat.blending = AdditiveBlending;
  atmoMat.depthWrite = false;
  world.add(new Mesh(new SphereGeometry(1.14, 96, 48), atmoMat));

  // Focus marker for the voice assistant
  const marker = new Mesh(new RingGeometry(0.05, 0.065, 48), new MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9, depthTest: false }));
  marker.visible = false;
  marker.renderOrder = 10;
  world.add(marker);
  let focusQuat: Quaternion | null = null;
  let focusedFeature = -1;
  let autoRotateResumeAt = 0;

  // Stars
  {
    const n = 2500;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = new Vector3().randomDirection().multiplyScalar(40 + Math.random() * 30);
      pos.set([v.x, v.y, v.z], i * 3);
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(pos, 3));
    scene.add(new Points(g, new PointsMaterial({ color: '#cfe3ff', size: 0.12, sizeAttenuation: true, transparent: true, opacity: 0.8 })));
  }

  // ---- UI ----
  const months = ds.months;
  const M = months.length;
  const scrub = $<HTMLInputElement>('scrub');
  const playBtn = $<HTMLButtonElement>('play');
  const yearEl = $('year');
  const dateEl = $('date');
  const tooltip = $('tooltip');
  const dateFmt = new Intl.DateTimeFormat('en', { month: 'long', year: 'numeric' });

  // legend
  const counts: Record<string, number> = {};
  for (const f of ds.features) counts[f.group] = (counts[f.group] ?? 0) + 1;
  $('legend').innerHTML = ds.groups
    .map((g) => `<li><i style="background:${GROUP_COLORS[g]}"></i>${g}<small>${counts[g] ?? 0}</small></li>`)
    .join('');

  // ticks
  const t0 = months[0], t1 = months[M - 1];
  const ticks = $('ticks');
  for (let y = new Date(t0).getFullYear(); y <= new Date(t1).getFullYear(); y += 2) {
    const t = new Date(y, 0, 1).getTime();
    if (t < t0 || t > t1) continue;
    const el = document.createElement('span');
    el.textContent = String(y);
    el.style.left = `${((t - t0) / (t1 - t0)) * 100}%`;
    ticks.appendChild(el);
  }

  let progress = 0; // 0..1
  let playing = true;
  let holdUntil = 0;
  let lastMonth = -1;

  const setPlaying = (p: boolean) => {
    playing = p;
    playBtn.textContent = p ? '❚❚' : '▶';
    playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
  };
  playBtn.addEventListener('click', () => {
    if (!playing && progress >= 1) progress = 0;
    setPlaying(!playing);
  });
  scrub.addEventListener('input', () => {
    progress = Number(scrub.value) / 10000;
    setPlaying(false);
  });
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); playBtn.click(); }
    if (e.code === 'ArrowLeft') { progress = Math.max(0, progress - 1 / (M - 1)); setPlaying(false); }
    if (e.code === 'ArrowRight') { progress = Math.min(1, progress + 1 / (M - 1)); setPlaying(false); }
    if (e.code === 'KeyC') recalibrate();
  });

  // tooltip / hover
  const raycaster = new Raycaster();
  const ndc = new Vector2();
  const hitSphere = new Sphere(new Vector3(), 1.04);
  const hit = new Vector3();
  let hovered = -1;
  let mouse = { x: 0, y: 0 };
  canvas.addEventListener('pointermove', (e) => {
    mouse = { x: e.clientX, y: e.clientY };
    ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    if (raycaster.ray.intersectSphere(hitSphere, hit)) {
      world.worldToLocal(hit).normalize();
      hovered = planet.nearestFeature(hit);
    } else hovered = -1;
    renderTooltip(ds, planet, hovered, Math.floor(progress * (M - 1)), tooltip, mouse);
  });
  canvas.addEventListener('pointerleave', () => { hovered = -1; tooltip.hidden = true; });

  // ---- face steering (WebNN / ONNX Runtime) ----
  const faceBtn = $<HTMLButtonElement>('face-toggle');
  const faceStatus = $('face-status');
  const cam = $('cam');
  const camBox = $('cam-box');
  let tracker: FaceTracker | null = null;
  let facePose: FacePose | null = null;
  const faceTarget = { yaw: 0, pitch: 0, dist: 0 };
  // neutral head pose, averaged over the first second of tracking (press C to recalibrate)
  const neutral = { yaw: 0, pitch: 0, x: 0, y: 0, size: 0.3, n: 0 };
  const CALIBRATION_SAMPLES = 30;
  const recalibrate = () => { neutral.n = 0; neutral.yaw = neutral.pitch = neutral.x = neutral.y = 0; neutral.size = 0; };
  const onPose = (p: FacePose | null) => {
    facePose = p;
    if (p && neutral.n < CALIBRATION_SAMPLES) {
      const k = neutral.n / (neutral.n + 1), j = 1 / (neutral.n + 1);
      neutral.yaw = neutral.yaw * k + p.yaw * j;
      neutral.pitch = neutral.pitch * k + p.pitch * j;
      neutral.x = neutral.x * k + p.x * j;
      neutral.y = neutral.y * k + p.y * j;
      neutral.size = neutral.size * k + p.size * j;
      neutral.n++;
    }
    (window as unknown as { __pose?: FacePose | null }).__pose = p;
    if (p) {
      camBox.style.display = 'block';
      // preview is mirrored, so flip x
      const cx = (1 - (p.x + 1) / 2) * 100, cy = ((p.y + 1) / 2) * 100;
      const w = p.size * 100, h = (p.size * 160 / 120) * 100;
      camBox.style.left = `${cx - w / 2}%`; camBox.style.top = `${cy - h / 2}%`;
      camBox.style.width = `${w}%`; camBox.style.height = `${h}%`;
    } else camBox.style.display = 'none';
  };
  faceBtn.addEventListener('click', async () => {
    if (tracker) {
      tracker.stop(); tracker = null; facePose = null;
      cam.hidden = true; faceBtn.classList.remove('active'); faceBtn.textContent = 'Steer with your face';
      faceStatus.textContent = 'camera stays on this device';
      controls.autoRotate = true;
      return;
    }
    faceBtn.disabled = true; faceStatus.textContent = 'Loading model, waiting for camera permission…';
    recalibrate();
    try {
      const forceOnnx = new URLSearchParams(location.search).get('engine') === 'onnx';
      if (webnnAvailable() || forceOnnx) {
        try { tracker = await startWebNNTracker(onPose, forceOnnx); }
        catch (e) { console.warn('[face] WebNN tracker failed, using MediaPipe', e); }
      }
      if (!tracker) tracker = await startHeadPoseTracker(onPose);
      cam.prepend(tracker.video);
      cam.hidden = false;
      faceBtn.classList.add('active'); faceBtn.textContent = 'Stop face steering';
      const hint = tracker.backend.startsWith('WebNN') ? '' : ' · enable chrome://flags/#web-machine-learning-neural-network for WebNN on the NPU';
      faceStatus.innerHTML = `Inference on <b>${tracker.backend}</b>${hint}`;
      controls.autoRotate = false;
    } catch (e) {
      console.error(e);
      faceStatus.textContent = `Face tracking failed: ${(e as Error).message}`;
    }
    faceBtn.disabled = false;
  });

  // ---- voice assistant (OpenAI Realtime) ----
  const voiceBtn = $<HTMLButtonElement>('voice-toggle');
  const voiceStatus = $('voice-status');
  const keyInput = $<HTMLInputElement>('openai-key');
  const captions = $('captions');
  try { keyInput.value = localStorage.getItem('openai_api_key') ?? ''; } catch { /* ignore */ }
  keyInput.addEventListener('change', () => { try { localStorage.setItem('openai_api_key', keyInput.value.trim()); } catch { /* ignore */ } });
  let voice: VoiceSession | null = null;
  const captionLines: string[] = [];
  const showCaption = (who: 'you' | 'planet', text: string) => {
    captionLines.push(`<div class="${who}"><b>${who === 'you' ? 'You' : 'Planet'}</b>${text}</div>`);
    while (captionLines.length > 3) captionLines.shift();
    captions.innerHTML = captionLines.join('');
    captions.hidden = false;
  };
  const focusFeature = (fi: number) => {
    focusedFeature = fi;
    const p = ds.features[fi].pos;
    const dir = new Vector3(p[0], p[1], p[2]).normalize();
    const camDir = camera.position.clone().normalize();
    focusQuat = new Quaternion().setFromUnitVectors(dir, camDir);
    marker.position.copy(dir).multiplyScalar(1.02);
    marker.lookAt(dir.clone().multiplyScalar(2));
    marker.visible = true;
    controls.autoRotate = false;
    autoRotateResumeAt = performance.now() + 8000;
  };
  const voiceTools = {
    lookupFeature: (query: string) => {
      const hits = findFeatures(ds, query);
      if (!hits.length) return { error: `No caniuse feature matches "${query}"` };
      return { match: describeFeature(ds, hits[0].index, progress * (M - 1)), other_candidates: hits.slice(1).map((h) => h.feature.title) };
    },
    goToFeature: (query: string) => {
      const hits = findFeatures(ds, query);
      if (!hits.length) return { error: `No caniuse feature matches "${query}"` };
      focusFeature(hits[0].index);
      const fi = hits[0].index;
      // make sure the feature exists at the current time
      if (ds.features[fi].firstMonth >= 0 && progress * (M - 1) < ds.features[fi].firstMonth) {
        progress = Math.min(1, (ds.features[fi].firstMonth + 12) / (M - 1));
        setPlaying(false);
      }
      return { navigated_to: ds.features[fi].title, ...voiceTools.lookupFeature(query) };
    },
    setTime: (year: number, month?: number) => {
      const t = new Date(year, (month ?? 1) - 1, 1).getTime();
      progress = Math.max(0, Math.min(1, (t - months[0]) / (months[M - 1] - months[0])));
      setPlaying(false);
      return { now: dateFmt.format(months[0] + progress * (months[M - 1] - months[0])) };
    },
    setPlayback: (p: boolean) => { if (p && progress >= 1) progress = 0; setPlaying(p); return { playing: p }; },
    getState: () => {
      let shipped = 0, full = 0;
      for (const sh of planet.featureShare) { if (sh > 0) shipped++; if (sh >= 0.999) full++; }
      return { date: dateFmt.format(months[0] + progress * (months[M - 1] - months[0])), playing, features_shipped: shipped, features_supported_everywhere: full, focused_feature: focusedFeature >= 0 ? ds.features[focusedFeature].title : null };
    },
  };
  (window as unknown as { __tools?: unknown }).__tools = voiceTools;
  voiceBtn.addEventListener('click', async () => {
    if (voice) { voice.stop(); voice = null; voiceBtn.classList.remove('active'); voiceBtn.textContent = 'Talk to the planet'; voiceStatus.textContent = 'OpenAI Realtime · WebRTC'; return; }
    const key = keyInput.value.trim();
    if (!key) { voiceStatus.textContent = 'Enter your OpenAI API key first'; keyInput.focus(); return; }
    voiceBtn.disabled = true;
    try {
      voice = await startVoice(key, voiceTools, (st) => { if (st) voiceStatus.textContent = st; }, showCaption);
      voiceBtn.classList.add('active'); voiceBtn.textContent = 'Stop talking';
    } catch (e) {
      console.error(e);
      voiceStatus.textContent = `Voice failed: ${(e as Error).message}`;
    }
    voiceBtn.disabled = false;
  });

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  let prev = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;
    if (playing) {
      if (now < holdUntil) { /* hold at the end */ }
      else if (progress >= 1) { progress = 0; }
      else {
        progress = Math.min(1, progress + dt / LOOP_SECONDS);
        if (progress >= 1) holdUntil = now + END_HOLD_SECONDS * 1000;
      }
    }
    const month = progress * (M - 1);
    if (month !== lastMonth) {
      planet.update(month);
      lastMonth = month;
      const time = months[Math.floor(month)] + (month % 1) * (months[Math.min(M - 1, Math.floor(month) + 1)] - months[Math.floor(month)]);
      yearEl.textContent = String(new Date(time).getFullYear());
      dateEl.textContent = dateFmt.format(time);
      if (document.activeElement !== scrub) scrub.value = String(Math.round(progress * 10000));
      updateStats(planet);
      if (hovered >= 0) renderTooltip(ds, planet, hovered, Math.floor(month), tooltip, mouse);
    }
    if (focusQuat) {
      world.quaternion.slerp(focusQuat, 0.08);
      if (world.quaternion.angleTo(focusQuat) < 0.002) focusQuat = null;
    }
    if (marker.visible) {
      marker.material.opacity = 0.6 + 0.4 * Math.sin(now / 250);
      if (focusedFeature >= 0) renderTooltip(ds, planet, focusedFeature, Math.floor(progress * (M - 1)), tooltip, screenPosOf(marker.getWorldPosition(new Vector3()), camera));
      if (autoRotateResumeAt && now > autoRotateResumeAt) { controls.autoRotate = !tracker; autoRotateResumeAt = 0; marker.visible = false; focusedFeature = -1; tooltip.hidden = true; }
    }
    if (tracker && !focusQuat) {
      if (facePose) {
        const dx = facePose.x - neutral.x, dy = facePose.y - neutral.y, dsize = facePose.size - neutral.size;
        // turn your head -> the planet turns with it (amplified); lean closer -> zoom in
        faceTarget.yaw = -(facePose.yaw - neutral.yaw) * 2.2 + dx * 0.4;
        faceTarget.pitch = (facePose.pitch - neutral.pitch) * 2.2 - dy * 0.3;
        faceTarget.dist = -dsize * 5;
      }
      world.rotation.y += (faceTarget.yaw - world.rotation.y) * 0.08;
      world.rotation.x += (faceTarget.pitch - world.rotation.x) * 0.08;
      const targetLen = Math.min(controls.maxDistance, Math.max(controls.minDistance, 4.1 + faceTarget.dist));
      const len = camera.position.length();
      camera.position.multiplyScalar(1 + ((targetLen - len) * 0.06) / len);
    }
    controls.update();
    renderer.render(scene, camera);
  });
}

function screenPosOf(worldPos: Vector3, camera: PerspectiveCamera) {
  const v = worldPos.project(camera);
  return { x: ((v.x + 1) / 2) * innerWidth, y: ((1 - v.y) / 2) * innerHeight };
}

function updateStats(planet: Planet) {
  let shipped = 0, full = 0, sum = 0;
  for (const s of planet.featureShare) {
    if (s > 0) { shipped++; sum += s; }
    if (s >= 0.999) full++;
  }
  $('stat-shipped').textContent = String(shipped);
  $('stat-full').textContent = String(full);
  $('stat-avg').textContent = shipped ? `${Math.round((sum / shipped) * 100)}%` : '0%';
}

const STATUS: Record<string, string> = {
  rec: 'W3C Recommendation', cr: 'W3C Candidate Recommendation', pr: 'W3C Proposed Recommendation',
  wd: 'W3C Working Draft', ls: 'WHATWG Living Standard', other: 'Other', unoff: 'Unofficial / note',
};
const CODE = ['—', 'no', 'partial', 'yes'];

function renderTooltip(ds: Dataset, planet: Planet, fi: number, month: number, el: HTMLElement, mouse: { x: number; y: number }) {
  if (fi < 0) { el.hidden = true; return; }
  const f = ds.features[fi];
  const share = planet.featureShare[fi];
  const M = ds.months.length;
  const B = ds.browsers.length;
  const rows = ds.browsers.map((b, bi) => {
    const c = ds.browserStatus[(fi * M + month) * B + bi];
    return `<span>${b.name}<b class="b${c}">${CODE[c]}</b></span>`;
  }).join('');
  el.innerHTML = `<div class="cat">${f.group} · ${STATUS[f.status] ?? f.status}</div><h3>${f.title}</h3>
    <div class="share" style="color:${GROUP_COLORS[f.group]}">${Math.round(share * 100)}% browser support</div>
    <div class="browsers">${rows}</div>`;
  el.hidden = false;
  const w = el.offsetWidth, h = el.offsetHeight;
  el.style.left = `${Math.min(innerWidth - w - 12, mouse.x + 16)}px`;
  el.style.top = `${Math.min(innerHeight - h - 12, mouse.y + 16)}px`;
}

main().catch((err) => {
  console.error(err);
  $('renderer-badge').textContent = `Failed to start: ${err?.message ?? err}`;
});
