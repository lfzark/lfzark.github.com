import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const meridianSpecs = [
  { id: 'lung', code: 'LU', name: '手太阴肺经', count: 11, bilateral: true, color: 0x61dafb, axis: 'x', dir: -1, path: [[0.46, 0.8, 0.6], [0.74, 0.56, 0.36], [0.9, 0.36, 0.32]] },
  { id: 'largeIntestine', code: 'LI', name: '手阳明大肠经', count: 20, bilateral: true, color: 0x53e0c1, axis: 'x', dir: 1, path: [[0.92, 0.32, 0.34], [0.7, 0.58, 0.32], [0.54, 0.8, 0.42], [0.3, 0.92, 0.52], [0.08, 0.88, 0.62]] },
  { id: 'stomach', code: 'ST', name: '足阳明胃经', count: 45, bilateral: true, color: 0xffba49, axis: 'z', dir: -1, path: [[0.08, 0.92, 0.62], [0.18, 0.8, 0.6], [0.22, 0.62, 0.6], [0.24, 0.42, 0.56], [0.28, 0.22, 0.52], [0.22, 0.04, 0.48]] },
  { id: 'spleen', code: 'SP', name: '足太阴脾经', count: 21, bilateral: true, color: 0x4dd0e1, axis: 'z', dir: -1, path: [[0.12, 0.04, 0.34], [0.18, 0.22, 0.2], [0.2, 0.44, 0.14], [0.2, 0.64, 0.16], [0.3, 0.8, 0.34]] },
  { id: 'heart', code: 'HT', name: '手少阴心经', count: 9, bilateral: true, color: 0xff7b7b, axis: 'x', dir: -1, path: [[0.28, 0.78, 0.2], [0.56, 0.62, 0.12], [0.82, 0.4, 0.2]] },
  { id: 'smallIntestine', code: 'SI', name: '手太阳小肠经', count: 19, bilateral: true, color: 0xff9e57, axis: 'x', dir: 1, path: [[0.82, 0.38, -0.02], [0.58, 0.62, -0.2], [0.52, 0.82, -0.32], [0.2, 0.92, -0.2], [0.08, 0.9, -0.04]] },
  { id: 'bladder', code: 'BL', name: '足太阳膀胱经', count: 67, bilateral: true, color: 0xff6f91, axis: 'z', dir: 1, path: [[0.1, 0.92, -0.2], [0.18, 0.84, -0.38], [0.2, 0.66, -0.6], [0.18, 0.46, -0.58], [0.16, 0.24, -0.52], [0.12, 0.04, -0.42]] },
  { id: 'kidney', code: 'KI', name: '足少阴肾经', count: 27, bilateral: true, color: 0x8fa8ff, axis: 'z', dir: -1, path: [[0.1, 0.04, -0.08], [0.14, 0.24, -0.1], [0.16, 0.46, -0.04], [0.14, 0.64, 0.04], [0.12, 0.82, 0.22]] },
  { id: 'pericardium', code: 'PC', name: '手厥阴心包经', count: 9, bilateral: true, color: 0xff8fb3, axis: 'x', dir: -1, path: [[0.34, 0.76, 0.24], [0.62, 0.58, 0.18], [0.84, 0.38, 0.16]] },
  { id: 'sanjiao', code: 'SJ', name: '手少阳三焦经', count: 23, bilateral: true, color: 0x8de5ff, axis: 'x', dir: 1, path: [[0.84, 0.36, 0.0], [0.62, 0.6, -0.08], [0.54, 0.82, -0.2], [0.22, 0.92, -0.12], [0.08, 0.9, 0.08]] },
  { id: 'gallbladder', code: 'GB', name: '足少阳胆经', count: 44, bilateral: true, color: 0xd4ff6d, axis: 'x', dir: 1, path: [[0.08, 0.94, 0.1], [0.28, 0.86, -0.02], [0.34, 0.66, -0.08], [0.3, 0.48, -0.12], [0.28, 0.26, -0.06], [0.22, 0.04, 0.04]] },
  { id: 'liver', code: 'LR', name: '足厥阴肝经', count: 14, bilateral: true, color: 0x9affb1, axis: 'z', dir: -1, path: [[0.18, 0.04, 0.22], [0.2, 0.24, 0.08], [0.18, 0.46, 0.02], [0.14, 0.66, 0.08], [0.12, 0.82, 0.24]] },
  { id: 'ren', code: 'RN', name: '任脉', count: 24, bilateral: false, color: 0xffffff, axis: 'z', dir: -1, path: [[0, 0.04, 0.22], [0, 0.24, 0.24], [0, 0.46, 0.26], [0, 0.7, 0.3], [0, 0.9, 0.42]] },
  { id: 'du', code: 'DU', name: '督脉', count: 28, bilateral: false, color: 0xc2c2ff, axis: 'z', dir: 1, path: [[0, 0.04, -0.2], [0, 0.24, -0.3], [0, 0.46, -0.44], [0, 0.7, -0.38], [0, 0.94, -0.12]] },
];
const FAST_INIT_MODE = false;
const SURFACE_CACHE_VERSION = 'v2';
const SURFACE_CACHE_KEY = `jingluo.surfacePoints.${SURFACE_CACHE_VERSION}`;
const PAIN_CACHE_KEY = 'jingluo.painMarks.v1';
const COMMON_ACUPOINT_IDS = new Set([
  'LI4', 'ST36', 'SP6', 'PC6', 'HT7', 'LR3', 'GB20', 'DU20', 'RN4', 'RN6',
  'RN12', 'BL23', 'BL40', 'KI3', 'SJ5', 'LU7', 'LI11', 'ST25', 'GB34', 'DU14',
]);

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function samplePath(path, t) {
  if (path.length === 1) return { x: path[0][0], y: path[0][1], z: path[0][2] };
  const p = Math.max(0, Math.min(1, t)) * (path.length - 1);
  const i = Math.min(path.length - 2, Math.floor(p));
  const k = p - i;
  const a = path[i];
  const b = path[i + 1];
  return { x: lerp(a[0], b[0], k), y: lerp(a[1], b[1], k), z: lerp(a[2], b[2], k) };
}

const meridians = meridianSpecs.map((m) => ({
  id: m.id,
  name: m.name,
  color: m.color,
  acupointIds: Array.from({ length: m.count }, (_, i) => `${m.code}${i + 1}`),
}));

const acupoints = meridianSpecs.flatMap((m) =>
  Array.from({ length: m.count }, (_, i) => {
    const idx = i + 1;
    const pos = samplePath(m.path, m.count === 1 ? 0 : i / (m.count - 1));
    return {
      id: `${m.code}${idx}`,
      name: `${m.code}${idx}`,
      meridianId: m.id,
      effect: `${m.name} 第${idx}穴`,
      anchor: { x: pos.x, y: pos.y, z: pos.z, axis: m.axis, dir: m.dir, mirror: m.bilateral },
    };
  })
);

function parseCsvLine(line) {
  const out = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === ',' && !inQuotes) {
      out.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function normalizeWhoId(nameWho) {
  const m = String(nameWho || '').trim().toUpperCase().match(/^([A-Z]{1,3})-(\d{1,3})$/);
  if (!m) return null;
  const raw = m[1];
  const idx = Number(m[2]);
  const codeMap = { P: 'PC', TB: 'SJ', LIV: 'LR', REN: 'RN', CV: 'RN', GV: 'DU' };
  const code = codeMap[raw] || raw;
  return `${code}${idx}`;
}

async function loadAcupointTranslations() {
  try {
    markPerf('csvStart');
    const res = await fetch('./assets/acupuncture-point-translations.csv');
    if (!res.ok) return;
    const text = await res.text();
    const lines = text.split(/\r?\n/).filter((x) => x.trim());
    if (lines.length < 2) return;

    const header = parseCsvLine(lines[0]);
    const whoIdx = header.indexOf('name_who');
    const zhIdx = header.indexOf('name_chinese');
    const enIdx = header.indexOf('name_english');
    if (whoIdx < 0 || zhIdx < 0) return;

    const map = new Map();
    for (let i = 1; i < lines.length; i += 1) {
      const cols = parseCsvLine(lines[i]);
      const id = normalizeWhoId(cols[whoIdx]);
      if (!id) continue;
      map.set(id, {
        name: (cols[zhIdx] || '').trim() || id,
        effect: (cols[enIdx] || '').trim(),
      });
    }

    acupoints.forEach((p) => {
      const row = map.get(p.id);
      if (!row) return;
      p.name = row.name;
      if (row.effect) p.effect = row.effect;
    });

    renderAcupointList(document.getElementById('search-input').value);
    markPerf('csvDone');
    finalizePerfIfReady();
  } catch {
    // Ignore CSV loading failures and keep fallback labels.
  }
}

const state = {
  painMarks: [],
  selectedMarkId: null,
  selectedAcupointId: null,
  meridianVisible: Object.fromEntries(meridians.map((m) => [m.id, true])),
};
const perf = { t0: performance.now(), marks: {} };

const canvas = document.getElementById('scene');
const perfPanel = document.getElementById('perf-panel');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStage = document.getElementById('loading-stage');
const loadingBar = document.getElementById('loading-bar');
const loadingPercent = document.getElementById('loading-percent');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1620);

const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
camera.position.set(1.8, 1.4, 2.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.9;
controls.minDistance = 1.2;
controls.maxDistance = 5;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(2, 2.5, 2);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x4fa8ff, 0.6);
rimLight.position.set(-2, 1.8, -2);
scene.add(rimLight);

const grid = new THREE.GridHelper(4, 20, 0x2f6a8d, 0x1b3b52);
grid.position.y = -0.02;
scene.add(grid);

const bodyRoot = new THREE.Group();
scene.add(bodyRoot);

const meridianGroup = new THREE.Group();
scene.add(meridianGroup);
const acupointGroup = new THREE.Group();
scene.add(acupointGroup);
const painGroup = new THREE.Group();
scene.add(painGroup);

const acupointMeshes = new Map();
const acupointWorldPos = new Map();
const meridianMeshes = new Map();
const bodyMeshes = [];

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const tempBox = new THREE.Box3();
const glowTexture = createGlowTexture();
let cameraTween = null;
let loadingTarget = 0;
let loadingVisual = 0;
let loadingDone = false;

function stageProgress(stage) {
  const map = {
    modelLoadStart: [8, '开始下载模型...'],
    modelLoaded: [36, '模型下载完成，正在归一化...'],
    modelFitted: [54, '模型归一化完成，重建穴位/经络...'],
    rebuildStart: [58, '重建穴位/经络中...'],
    rebuildDone: [86, '穴位/经络完成，加载中文数据...'],
    csvStart: [90, '解析穴位中文数据...'],
    csvDone: [98, '收尾处理中...'],
    appReady: [100, '加载完成'],
  };
  return map[stage] || null;
}

function setLoadingTargetByStage(stage) {
  const step = stageProgress(stage);
  if (!step) return;
  loadingTarget = Math.max(loadingTarget, step[0]);
  if (loadingStage) loadingStage.textContent = step[1];
}

function setLoadingProgress(percent, text) {
  loadingTarget = Math.max(loadingTarget, Math.min(100, percent));
  if (loadingStage && text) loadingStage.textContent = text;
}

function renderLoading() {
  if (!loadingBar || !loadingPercent) return;
  loadingVisual += (loadingTarget - loadingVisual) * 0.15;
  if (Math.abs(loadingTarget - loadingVisual) < 0.2) loadingVisual = loadingTarget;
  const shown = Math.max(0, Math.min(100, loadingVisual));
  loadingBar.style.width = `${shown.toFixed(1)}%`;
  loadingPercent.textContent = `${Math.round(shown)}%`;
  if (!loadingDone) requestAnimationFrame(renderLoading);
}
requestAnimationFrame(renderLoading);

function markPerf(label) {
  perf.marks[label] = performance.now();
  setLoadingTargetByStage(label);
  renderPerfPanel();
}

function formatMs(v) {
  return `${Math.round(v)} ms`;
}

function renderPerfPanel() {
  if (!perfPanel) return;
  const m = perf.marks;
  const lines = ['加载计时'];
  const push = (name, start, end) => {
    if (m[start] == null || m[end] == null) return;
    lines.push(`${name}: ${formatMs(m[end] - m[start])}`);
  };
  push('模型下载', 'modelLoadStart', 'modelLoaded');
  push('模型归一化', 'modelLoaded', 'modelFitted');
  push('穴位/经络重建', 'rebuildStart', 'rebuildDone');
  push('CSV解析', 'csvStart', 'csvDone');
  if (m.appReady != null) lines.push(`总耗时: ${formatMs(m.appReady - perf.t0)}`);
  perfPanel.textContent = lines.join('\n');
}

function finalizePerfIfReady() {
  if (perf.marks.appReady != null) return;
  if (perf.marks.csvDone == null) return;
  if (perf.marks.rebuildDone == null) return;
  markPerf('appReady');
  loadingDone = true;
  if (loadingOverlay) {
    setTimeout(() => loadingOverlay.classList.add('hidden'), 220);
  }
  console.table(
    Object.entries(perf.marks).map(([stage, t]) => ({
      stage,
      sinceStartMs: Math.round(t - perf.t0),
    }))
  );
}

function createGlowTexture() {
  const size = 128;
  const canvasTex = document.createElement('canvas');
  canvasTex.width = size;
  canvasTex.height = size;
  const ctx = canvasTex.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.4, 'rgba(255,255,255,0.7)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvasTex);
  texture.needsUpdate = true;
  return texture;
}

function collectBodyMeshes(root) {
  bodyMeshes.length = 0;
  root.traverse((obj) => {
    if (obj.isMesh) bodyMeshes.push(obj);
  });
}

function applyTransparentMaterial(root) {
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    obj.material = new THREE.MeshPhysicalMaterial({
      color: 0x9ed6ff,
      transparent: true,
      opacity: 0.26,
      roughness: 0.2,
      metalness: 0,
      transmission: 0.2,
      depthWrite: false,
    });
    obj.castShadow = false;
    obj.receiveShadow = false;
  });
}

function fitModelToScene(model) {
  tempBox.setFromObject(model);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  tempBox.getSize(size);
  tempBox.getCenter(center);

  const targetHeight = 2.0;
  const scale = size.y > 0 ? targetHeight / size.y : 1;
  model.scale.multiplyScalar(scale);
  model.updateMatrixWorld(true);

  // 朝向修正：若深度大于宽度，通常模型侧着，绕Y旋转90度。
  tempBox.setFromObject(model);
  tempBox.getSize(size);
  if (size.z > size.x * 1.15) {
    model.rotation.y += Math.PI / 2;
    model.updateMatrixWorld(true);
  }

  // 居中并让脚底贴地。
  tempBox.setFromObject(model);
  tempBox.getCenter(center);
  model.position.x -= center.x;
  model.position.z -= center.z;
  model.updateMatrixWorld(true);

  tempBox.setFromObject(model);
  model.position.y -= tempBox.min.y;
  model.updateMatrixWorld(true);

  controls.target.set(0, 1.0, 0);
  camera.position.set(1.8, 1.35, 2.3);
  controls.update();
}

function createTransparentHumanPlaceholder() {
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x9ed6ff,
    transparent: true,
    opacity: 0.28,
    roughness: 0.15,
    metalness: 0,
    transmission: 0.18,
    depthWrite: false,
  });

  const parts = [];
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.33, 0.9, 12, 22), mat);
  torso.position.y = 1.1;
  parts.push(torso);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.2, 24, 24), mat);
  head.position.y = 1.85;
  parts.push(head);

  const armGeo = new THREE.CapsuleGeometry(0.09, 0.72, 8, 16);
  const armL = new THREE.Mesh(armGeo, mat);
  armL.position.set(-0.45, 1.15, 0);
  armL.rotation.z = -0.18;
  const armR = armL.clone();
  armR.position.x = 0.45;
  armR.rotation.z = 0.18;
  parts.push(armL, armR);

  const legGeo = new THREE.CapsuleGeometry(0.12, 0.95, 8, 18);
  const legL = new THREE.Mesh(legGeo, mat);
  legL.position.set(-0.2, 0.35, 0);
  const legR = legL.clone();
  legR.position.x = 0.2;
  parts.push(legL, legR);

  bodyRoot.clear();
  parts.forEach((m) => bodyRoot.add(m));
  collectBodyMeshes(bodyRoot);
  rebuildAcupointsAndMeridians();
}

function getIntersections(ev, objects) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(objects, true);
}

function sampleSurfacePoint(anchor, side = 'L') {
  if (!bodyMeshes.length) return null;

  tempBox.setFromObject(bodyRoot);
  const size = tempBox.getSize(new THREE.Vector3());
  const center = tempBox.getCenter(new THREE.Vector3());

  const sideSign = side === 'L' ? 1 : -1;
  const xLocal = anchor.mirror ? anchor.x * sideSign : anchor.x;

  const target = new THREE.Vector3(
    center.x + (xLocal * size.x) / 2,
    tempBox.min.y + anchor.y * size.y,
    center.z + (anchor.z * size.z) / 2
  );

  const axis = anchor.axis || 'z';
  const dirSign = anchor.dir || -1;
  const span = Math.max(size.x, size.y, size.z) * 1.4;

  const from = target.clone();
  if (axis === 'x') from.x += span * dirSign * sideSign;
  else if (axis === 'y') from.y += span * dirSign;
  else from.z += span * dirSign;

  if (FAST_INIT_MODE) return target;

  const candidates = [];
  const primaryDir = target.clone().sub(from).normalize();
  candidates.push({ from: from.clone(), dir: primaryDir });

  // Multi-direction projection improves robustness on varied meshes/orientations.
  const dirs = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0, -1),
  ];
  dirs.forEach((d) => {
    candidates.push({
      from: target.clone().addScaledVector(d, span),
      dir: d.clone().multiplyScalar(-1),
    });
  });

  let bestPoint = null;
  let bestDist = Infinity;
  for (const c of candidates) {
    raycaster.set(c.from, c.dir);
    const hits = raycaster.intersectObjects(bodyMeshes, true);
    if (!hits.length) continue;
    const p = hits[0].point;
    const dist = p.distanceTo(target);
    if (dist < bestDist) {
      bestDist = dist;
      bestPoint = p.clone();
    }
  }

  return bestPoint || target;
}

function mirrorWorldPointAcrossBody(point) {
  tempBox.setFromObject(bodyRoot);
  const center = tempBox.getCenter(new THREE.Vector3());
  return new THREE.Vector3(center.x - (point.x - center.x), point.y, point.z);
}

function clearAcupointsAndMeridians() {
  acupointGroup.clear();
  meridianGroup.clear();
  acupointMeshes.clear();
  acupointWorldPos.clear();
  meridianMeshes.clear();
}

function createAcupointMesh(position, acupoint, side) {
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: glowTexture,
      color: 0xffefb6,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    })
  );
  sprite.scale.setScalar(0.03);
  sprite.position.copy(position);
  sprite.userData = {
    type: 'acupoint',
    id: acupoint.id,
    meta: acupoint,
    side,
    mat: sprite.material,
  };
  return sprite;
}

function getModelCacheSignature() {
  tempBox.setFromObject(bodyRoot);
  const size = tempBox.getSize(new THREE.Vector3());
  return [
    size.x.toFixed(4),
    size.y.toFixed(4),
    size.z.toFixed(4),
    acupoints.length,
    meridians.length,
  ].join('|');
}

function readSurfaceCache() {
  try {
    const raw = localStorage.getItem(SURFACE_CACHE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || data.version !== SURFACE_CACHE_VERSION) return null;
    if (data.signature !== getModelCacheSignature()) return null;
    if (!data.points || typeof data.points !== 'object') return null;
    return data.points;
  } catch {
    return null;
  }
}

function writeSurfaceCache(points) {
  try {
    localStorage.setItem(
      SURFACE_CACHE_KEY,
      JSON.stringify({
        version: SURFACE_CACHE_VERSION,
        signature: getModelCacheSignature(),
        points,
        updatedAt: Date.now(),
      })
    );
  } catch {
    // ignore cache write failures
  }
}

function loadPainMarksFromCache() {
  try {
    const raw = localStorage.getItem(PAIN_CACHE_KEY);
    if (!raw) return;
    const list = JSON.parse(raw);
    if (!Array.isArray(list)) return;
    list.forEach((m) => {
      if (!m || !Array.isArray(m.pos) || m.pos.length !== 3) return;
      const id = String(m.id || `P-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`);
      const level = Number(m.level) >= 1 && Number(m.level) <= 10 ? Number(m.level) : 5;
      const note = String(m.note || '');
      const createdAt = m.createdAt || new Date().toISOString();
      const pos = new THREE.Vector3(Number(m.pos[0]), Number(m.pos[1]), Number(m.pos[2]));

      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.01, 12, 12),
        new THREE.MeshBasicMaterial({
          color: 0xff8a80,
          transparent: true,
          opacity: 0.95,
          depthWrite: false,
        })
      );
      const halo = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: glowTexture,
          color: 0xff6f61,
          transparent: true,
          opacity: 0.45,
          depthWrite: false,
        })
      );
      halo.scale.setScalar(0.06);
      marker.add(halo);
      marker.position.copy(pos);
      marker.userData = { type: 'pain', id, core: marker.material, halo: halo.material };
      painGroup.add(marker);

      state.painMarks.push({ id, level, note, pos: pos.toArray(), createdAt });
    });
  } catch {
    // ignore malformed cache
  }
}

function savePainMarksToCache() {
  try {
    localStorage.setItem(PAIN_CACHE_KEY, JSON.stringify(state.painMarks));
  } catch {
    // ignore storage failure
  }
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function rebuildAcupointsAndMeridians() {
  markPerf('rebuildStart');
  setLoadingProgress(58, '重建穴位中... 0%');
  clearAcupointsAndMeridians();
  const cachedPoints = readSurfaceCache();
  const newCachePoints = {};
  const useCache = !!cachedPoints;

  const totalAcupoints = acupoints.length;
  for (let idx = 0; idx < totalAcupoints; idx += 1) {
    const point = acupoints[idx];
    const leftKey = `${point.id}-L`;
    const rightKey = `${point.id}-R`;
    const leftPos = useCache && cachedPoints[leftKey]
      ? new THREE.Vector3(...cachedPoints[leftKey])
      : sampleSurfacePoint(point.anchor, 'L');
    const pair = [];

    if (leftPos) {
      const leftMesh = createAcupointMesh(leftPos, point, 'L');
      acupointGroup.add(leftMesh);
      pair.push(leftMesh);
      acupointWorldPos.set(leftKey, leftPos.clone());
      newCachePoints[leftKey] = leftPos.toArray();
    }

    if (point.anchor.mirror) {
      const rightPos = useCache && cachedPoints[rightKey]
        ? new THREE.Vector3(...cachedPoints[rightKey])
        : sampleSurfacePoint(point.anchor, 'R');
      if (rightPos) {
        const rightMesh = createAcupointMesh(rightPos, point, 'R');
        acupointGroup.add(rightMesh);
        pair.push(rightMesh);
        acupointWorldPos.set(rightKey, rightPos.clone());
        newCachePoints[rightKey] = rightPos.toArray();
      }
    }

    acupointMeshes.set(point.id, pair);

    // 58% -> 78%
    if (idx % 8 === 0 || idx === totalAcupoints - 1) {
      const p = Math.round(((idx + 1) / totalAcupoints) * 100);
      const prefix = useCache ? '重建穴位中(缓存)... ' : '重建穴位中... ';
      setLoadingProgress(58 + ((idx + 1) / totalAcupoints) * 20, `${prefix}${p}%`);
      await nextFrame();
    }
  }
  if (!useCache) writeSurfaceCache(newCachePoints);

  const totalMeridians = meridians.length;
  for (let idx = 0; idx < totalMeridians; idx += 1) {
    const m = meridians[idx];
    const sides = ['L', 'R'];
    const meshes = [];

    sides.forEach((side) => {
      const pts = m.acupointIds.map((id) => acupointWorldPos.get(`${id}-${side}`)).filter(Boolean);
      if (pts.length < 2) return;

      const curve = new THREE.CatmullRomCurve3(pts);
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(120));
      const material = new THREE.LineBasicMaterial({
        color: m.color,
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
      });
      const line = new THREE.Line(geometry, material);
      line.visible = state.meridianVisible[m.id];
      meridianGroup.add(line);
      meshes.push(line);
    });

    meridianMeshes.set(m.id, meshes);

    // 78% -> 86%
    if (idx % 2 === 0 || idx === totalMeridians - 1) {
      const p = Math.round(((idx + 1) / totalMeridians) * 100);
      setLoadingProgress(78 + ((idx + 1) / totalMeridians) * 8, `重建经络中... ${p}%`);
      await nextFrame();
    }
  }

  for (let idx = 0; idx < totalAcupoints; idx += 1) {
    const p = acupoints[idx];
    const visible = state.meridianVisible[p.meridianId];
    (acupointMeshes.get(p.id) || []).forEach((mesh) => {
      mesh.visible = visible && COMMON_ACUPOINT_IDS.has(p.id);
    });
    if (idx % 24 === 0 || idx === totalAcupoints - 1) await nextFrame();
  }
  setLoadingProgress(86, '穴位/经络重建完成');
  markPerf('rebuildDone');
  finalizePerfIfReady();
}

function loadHumanModel() {
  markPerf('modelLoadStart');
  const loader = new GLTFLoader();
  loader.load(
    './assets/human.glb',
    (gltf) => {
      markPerf('modelLoaded');
      bodyRoot.clear();
      const model = gltf.scene;
      applyTransparentMaterial(model);
      bodyRoot.add(model);

      fitModelToScene(model);
      markPerf('modelFitted');
      collectBodyMeshes(bodyRoot);
      rebuildAcupointsAndMeridians();
    },
    undefined,
    () => {
      createTransparentHumanPlaceholder();
    }
  );
}
loadHumanModel();

canvas.addEventListener('click', (ev) => {
  const painHits = getIntersections(ev, [...painGroup.children]);
  if (painHits.length) {
    let obj = painHits[0].object;
    while (obj && !obj.userData?.id) obj = obj.parent;
    if (obj?.userData?.id) {
      focusPainMark(obj.userData.id);
      return;
    }
  }

  const acupointHits = getIntersections(ev, [...acupointGroup.children]);
  if (acupointHits.length) {
    const hit = acupointHits[0].object.userData.meta;
    focusAcupoint(hit.id);
    return;
  }

  const bodyHits = getIntersections(ev, bodyMeshes);
  if (!bodyHits.length) return;
  createPainMark(bodyHits[0].point);
});

function createPainMark(pos) {
  const id = `P-${Date.now()}`;
  const level = 5;
  const note = '';
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.01, 12, 12),
    new THREE.MeshBasicMaterial({
      color: 0xff8a80,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    })
  );
  const halo = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: glowTexture,
      color: 0xff6f61,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    })
  );
  halo.scale.setScalar(0.06);
  marker.add(halo);

  marker.position.copy(pos);
  marker.userData = { type: 'pain', id, core: marker.material, halo: halo.material };
  painGroup.add(marker);

  state.painMarks.push({ id, level, note, pos: pos.toArray(), createdAt: new Date().toISOString() });
  savePainMarksToCache();
  focusPainMark(id);
}

function updatePainMarkHighlight() {
  painGroup.children.forEach((mesh) => {
    const selected = mesh.userData.id === state.selectedMarkId;
    mesh.scale.setScalar(selected ? 1.45 : 1);
    if (mesh.userData.core) {
      mesh.userData.core.color.set(selected ? 0xfff3c4 : 0xff8a80);
      mesh.userData.core.opacity = selected ? 1 : 0.95;
    }
    if (mesh.userData.halo) {
      mesh.userData.halo.opacity = selected ? 0.85 : 0.45;
      mesh.userData.halo.color.set(selected ? 0xffb347 : 0xff6f61);
    }
  });
}

function tweenCameraTo(targetPos, distance = 0.85) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (!Number.isFinite(dir.length()) || dir.lengthSq() < 1e-6) dir.set(0.55, 0.2, 1).normalize();
  const endTarget = targetPos.clone();
  const endCamera = endTarget.clone().add(dir.multiplyScalar(distance));

  cameraTween = {
    t: 0,
    duration: 0.35,
    startTarget: controls.target.clone(),
    endTarget,
    startCamera: camera.position.clone(),
    endCamera,
  };
}

function focusPainMark(id) {
  const mark = state.painMarks.find((m) => m.id === id);
  if (!mark) return;
  state.selectedMarkId = id;
  state.selectedAcupointId = null;
  updatePainMarkHighlight();
  renderAcupointList(document.getElementById('search-input').value);
  renderPainList();
  renderSelectedPainPanel();
  tweenCameraTo(new THREE.Vector3(...mark.pos), 0.7);
}

function focusAcupoint(id) {
  state.selectedAcupointId = id;
  state.selectedMarkId = null;
  updatePainMarkHighlight();

  acupointMeshes.forEach((pair, key) => {
    const isTarget = key === id;
    pair.forEach((mesh) => {
      if (isTarget) mesh.visible = true;
      mesh.scale.setScalar(isTarget ? 0.05 : 0.03);
      if (mesh.userData.mat) {
        mesh.userData.mat.opacity = isTarget ? 0.95 : 0.65;
        mesh.userData.mat.color.set(isTarget ? 0xffcf6c : 0xffefb6);
      }
    });
  });

  const targetPos = acupointWorldPos.get(`${id}-L`) || acupointWorldPos.get(`${id}-R`);
  if (!targetPos) return;

  renderAcupointList(document.getElementById('search-input').value);
  renderPainList();
  tweenCameraTo(targetPos, 0.78);
}

function renderMeridianFilters() {
  const wrap = document.getElementById('meridian-filters');
  wrap.innerHTML = '';

  meridians.forEach((m) => {
    const row = document.createElement('label');
    row.className = 'checkbox-row';

    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = state.meridianVisible[m.id];
    input.addEventListener('change', () => {
      state.meridianVisible[m.id] = input.checked;
      (meridianMeshes.get(m.id) || []).forEach((mesh) => {
        mesh.visible = input.checked;
      });
      acupoints.forEach((p) => {
        if (p.meridianId !== m.id) return;
        (acupointMeshes.get(p.id) || []).forEach((mesh) => {
          mesh.visible = input.checked && (COMMON_ACUPOINT_IDS.has(p.id) || p.id === state.selectedAcupointId);
        });
      });
    });

    const text = document.createElement('span');
    text.textContent = m.name;

    row.append(input, text);
    wrap.appendChild(row);
  });
}

function renderAcupointList(filter = '') {
  const list = document.getElementById('acupoint-list');
  list.innerHTML = '';

  const lower = filter.trim().toLowerCase();
  acupoints
    .filter((p) => !lower || p.name.toLowerCase().includes(lower) || p.id.toLowerCase().includes(lower))
    .forEach((p) => {
      const li = document.createElement('li');
      if (p.id === state.selectedAcupointId) li.classList.add('selected');
      li.innerHTML = `<strong>${p.id} ${p.name}</strong><br/><small>${p.effect}</small>`;
      li.addEventListener('click', () => focusAcupoint(p.id));
      list.appendChild(li);
    });
}

function renderPainList() {
  const list = document.getElementById('pain-list');
  list.innerHTML = '';

  state.painMarks
    .slice()
    .reverse()
    .forEach((mark) => {
      const li = document.createElement('li');
      if (mark.id === state.selectedMarkId) li.classList.add('selected');
      const when = new Date(mark.createdAt).toLocaleString('zh-CN');
      li.innerHTML = `<strong>${mark.id}</strong><br/>疼痛等级: ${mark.level}<br/><small>${when}</small>`;
      li.addEventListener('click', () => {
        focusPainMark(mark.id);
      });
      list.appendChild(li);
    });
}

function renderSelectedPainPanel() {
  const panel = document.getElementById('selected-point-panel');
  const selected = state.painMarks.find((m) => m.id === state.selectedMarkId);
  if (!selected) {
    panel.className = 'section muted';
    panel.textContent = '点击模型表面创建疼痛点';
    return;
  }

  panel.className = 'section';
  panel.innerHTML = '';

  const levelLabel = document.createElement('label');
  levelLabel.textContent = '疼痛等级 (1-10)';
  const levelInput = document.createElement('input');
  levelInput.type = 'range';
  levelInput.min = '1';
  levelInput.max = '10';
  levelInput.value = String(selected.level);

  const levelText = document.createElement('div');
  levelText.textContent = `当前等级: ${selected.level}`;

  const noteLabel = document.createElement('label');
  noteLabel.textContent = '备注';
  const noteInput = document.createElement('textarea');
  noteInput.rows = 4;
  noteInput.value = selected.note;

  const saveBtn = document.createElement('button');
  saveBtn.textContent = '保存备注';
  saveBtn.addEventListener('click', () => {
    selected.note = noteInput.value.trim();
    selected.level = Number(levelInput.value);
    savePainMarksToCache();
    renderPainList();
  });

  const removeBtn = document.createElement('button');
  removeBtn.textContent = '删除该疼痛点';
  removeBtn.style.marginTop = '8px';
  removeBtn.style.background = 'var(--danger)';
  removeBtn.style.color = '#2b0b07';
  removeBtn.addEventListener('click', () => {
    const idx = state.painMarks.findIndex((m) => m.id === selected.id);
    if (idx >= 0) state.painMarks.splice(idx, 1);
    painGroup.children.forEach((mesh) => {
      if (mesh.userData.id === selected.id) {
        painGroup.remove(mesh);
      }
    });
    state.selectedMarkId = null;
    savePainMarksToCache();
    updatePainMarkHighlight();
    renderPainList();
    renderSelectedPainPanel();
  });

  levelInput.addEventListener('input', () => {
    levelText.textContent = `当前等级: ${levelInput.value}`;
  });

  panel.append(levelLabel, levelInput, levelText, noteLabel, noteInput, saveBtn, removeBtn);
}

document.getElementById('search-input').addEventListener('input', (ev) => {
  renderAcupointList(ev.target.value);
});

renderMeridianFilters();
renderAcupointList('');
loadPainMarksFromCache();
renderPainList();
renderSelectedPainPanel();
loadAcupointTranslations();

window.addEventListener('resize', () => {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
});

function animate() {
  if (cameraTween) {
    cameraTween.t += 1 / 60 / cameraTween.duration;
    const t = Math.min(1, cameraTween.t);
    const ease = 1 - (1 - t) * (1 - t);
    controls.target.lerpVectors(cameraTween.startTarget, cameraTween.endTarget, ease);
    camera.position.lerpVectors(cameraTween.startCamera, cameraTween.endCamera, ease);
    if (t >= 1) cameraTween = null;
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
