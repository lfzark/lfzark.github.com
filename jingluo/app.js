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
const INITIAL_LIGHT_MODE = true;
const SURFACE_CACHE_VERSION = 'v2';
const SURFACE_CACHE_KEY = `jingluo.surfacePoints.${SURFACE_CACHE_VERSION}`;
const PAIN_CACHE_KEY = 'jingluo.painMarks.v1';
const COMMON_ACUPOINT_IDS = new Set([
  // 肺经 LU
  'LU1', 'LU5', 'LU6', 'LU7', 'LU9', 'LU10',
  // 大肠经 LI
  'LI4', 'LI10', 'LI11', 'LI14', 'LI15',
  // 胃经 ST
  'ST25', 'ST31', 'ST34', 'ST36', 'ST40', 'ST41', 'ST44',
  // 脾经 SP
  'SP3', 'SP4', 'SP6', 'SP9', 'SP10',
  // 心经 HT
  'HT3', 'HT5', 'HT6', 'HT7',
  // 小肠经 SI
  'SI3', 'SI9', 'SI10', 'SI11',
  // 膀胱经 BL
  'BL13', 'BL15', 'BL18', 'BL20', 'BL23', 'BL25', 'BL40', 'BL57', 'BL60',
  // 肾经 KI
  'KI3', 'KI6', 'KI7', 'KI10',
  // 心包经 PC
  'PC3', 'PC4', 'PC5', 'PC6', 'PC7',
  // 三焦经 SJ
  'SJ5', 'SJ6', 'SJ10', 'SJ14', 'SJ17',
  // 胆经 GB
  'GB20', 'GB21', 'GB30', 'GB34', 'GB37', 'GB39', 'GB41',
  // 肝经 LR
  'LR2', 'LR3', 'LR8', 'LR13', 'LR14',
  // 任脉 RN
  'RN4', 'RN6', 'RN8', 'RN12', 'RN17',
  // 督脉 DU
  'DU4', 'DU14', 'DU16', 'DU20', 'DU24',
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

const T2S_MAP = {
  '會':'会','陰':'阴','陽':'阳','關':'关','門':'门','風':'风','氣':'气','靈':'灵','臺':'台','腦':'脑','後':'后','頂':'顶','前':'前','囟':'囟',
  '長':'长','強':'强','腰':'腰','兪':'俞','懸':'悬','脊':'脊','樞':'枢','縮':'缩','椎':'椎','瘂':'哑','戶':'户','間':'间','百':'百','實':'实',
  '齦':'龈','兌':'兑','魚':'鱼','際':'际','內':'内','關':'关','沖':'冲','處':'处','竅':'窍','闕':'阙','帶':'带','絡':'络','經':'经','脈':'脉',
  '腎':'肾','膽':'胆','膀':'膀','胱':'胱','臟':'脏','腑':'腑','衛':'卫','營':'营','灸':'灸','針':'针','灘':'滩','雲':'云','俠':'侠','澤':'泽',
  '溝':'沟','穀':'谷','嶺':'岭','濱':'滨','濕':'湿','髎':'髎','髖':'髋','臍':'脐','闌':'阑','谿':'溪','谿':'溪','厥':'厥','藥':'药','圍':'围',
  '懷':'怀','釐':'厘','濟':'济','瀉':'泻','從':'从','來':'来','對':'对','應':'应','邊':'边','側':'侧','點':'点'
};

function toSimplified(text) {
  if (!text) return text;
  let out = '';
  for (const ch of String(text)) out += (T2S_MAP[ch] || ch);
  return out;
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
        name: toSimplified((cols[zhIdx] || '').trim() || id),
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
  showAcupoints: true,
  showPains: true,
  showAllAcupoints: false,
};
const perf = { t0: performance.now(), marks: {}, rebuild: null };

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
let highlightedAcupointId = null;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const tempBox = new THREE.Box3();
const glowTexture = createGlowTexture();
let cameraTween = null;
let loadingTarget = 0;
let loadingVisual = 0;
let loadingDone = false;
let rebuildingInProgress = false;
let adaptivePixelRatio = Math.min(window.devicePixelRatio, 2);
let fpsFrameCount = 0;
let fpsWindowStart = performance.now();

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
  if (perf.rebuild) {
    const r = perf.rebuild;
    lines.push('');
    lines.push('重建细节');
    lines.push(`- 穴位生成(计算): ${formatMs(r.acupointsComputeMs)} (${r.acupointsCount}个, 平均${r.acupointComputeAvgMs.toFixed(2)}ms/个)`);
    lines.push(`- 穴位生成(墙钟): ${formatMs(r.acupointsWallMs)} (${r.acupointsCount}个, 平均${r.acupointWallAvgMs.toFixed(2)}ms/个)`);
    lines.push(`- 经络生成(计算): ${formatMs(r.meridiansComputeMs)} (${r.meridiansCount}条, 平均${r.meridianComputeAvgMs.toFixed(2)}ms/条)`);
    lines.push(`- 经络生成(墙钟): ${formatMs(r.meridiansWallMs)} (${r.meridiansCount}条, 平均${r.meridianWallAvgMs.toFixed(2)}ms/条)`);
    lines.push(`- 可见性刷新(计算): ${formatMs(r.visibilityComputeMs)}`);
    lines.push(`- 可见性刷新(墙钟): ${formatMs(r.visibilityWallMs)}`);
  }
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
      const region = m.region ? String(m.region) : classifyBodyRegion(pos);

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

      state.painMarks.push({ id, level, note, region, pos: pos.toArray(), createdAt });
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

function clearAllPainMarks() {
  state.painMarks.splice(0, state.painMarks.length);
  state.selectedMarkId = null;
  while (painGroup.children.length) {
    painGroup.remove(painGroup.children[0]);
  }
  savePainMarksToCache();
  updatePainMarkHighlight();
  renderPainList();
  renderSelectedPainPanel();
}

function exportPainMarks() {
  const payload = {
    exportedAt: new Date().toISOString(),
    count: state.painMarks.length,
    painMarks: state.painMarks,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  a.href = url;
  a.download = `jingluo-pain-marks-${ts}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function yieldByFrameBudget(frameStart, budgetMs = 6) {
  if (performance.now() - frameStart > budgetMs) {
    await nextFrame();
    return performance.now();
  }
  return frameStart;
}

async function rebuildAcupointsAndMeridians(opts = {}) {
  rebuildingInProgress = true;
  try {
    const lightMode = !!opts.lightMode;
    markPerf('rebuildStart');
    setLoadingProgress(58, '重建穴位中... 0%');
    clearAcupointsAndMeridians();
    const tRebuildStart = performance.now();
    const cachedPoints = readSurfaceCache();
    const newCachePoints = {};
    const useCache = !!cachedPoints;
    const pointsToBuild = lightMode
      ? acupoints.filter((p) => COMMON_ACUPOINT_IDS.has(p.id))
      : acupoints;

    const totalAcupoints = pointsToBuild.length;
    const tAcupointWallStart = performance.now();
    let acupointComputeMs = 0;
    let frameStart = performance.now();
    for (let idx = 0; idx < totalAcupoints; idx += 1) {
      const tOneStart = performance.now();
      const point = pointsToBuild[idx];
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
      }
      acupointComputeMs += performance.now() - tOneStart;
      frameStart = await yieldByFrameBudget(frameStart, 6);
    }
    if (!useCache) writeSurfaceCache(newCachePoints);
    const tAcupointWallDone = performance.now();

    const totalMeridians = meridians.length;
    const tMeridianWallStart = performance.now();
    let meridianComputeMs = 0;
    frameStart = performance.now();
    for (let idx = 0; idx < totalMeridians; idx += 1) {
    const tOneStart = performance.now();
    const m = meridians[idx];
    const sides = ['L', 'R'];
    const meshes = [];

    sides.forEach((side) => {
      const pts = m.acupointIds.map((id) => acupointWorldPos.get(`${id}-${side}`)).filter(Boolean);
      if (pts.length < 2) return;

      const curve = new THREE.CatmullRomCurve3(pts);
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(60));
      const material = new THREE.LineDashedMaterial({
        color: m.color,
        transparent: true,
        opacity: 0.42,
        dashSize: 0.04,
        gapSize: 0.028,
        depthWrite: false,
      });
      const line = new THREE.Line(geometry, material);
      line.computeLineDistances();
      line.visible = state.meridianVisible[m.id];
      meridianGroup.add(line);
      meshes.push(line);
    });

    meridianMeshes.set(m.id, meshes);

    // 78% -> 86%
      if (idx % 2 === 0 || idx === totalMeridians - 1) {
      const p = Math.round(((idx + 1) / totalMeridians) * 100);
      setLoadingProgress(78 + ((idx + 1) / totalMeridians) * 8, `重建经络中... ${p}%`);
      }
      meridianComputeMs += performance.now() - tOneStart;
      frameStart = await yieldByFrameBudget(frameStart, 6);
    }
    const tMeridianWallDone = performance.now();

    const tVisibilityWallStart = performance.now();
    let visibilityComputeMs = 0;
    frameStart = performance.now();
    for (let idx = 0; idx < totalAcupoints; idx += 1) {
    const tOneStart = performance.now();
    const p = pointsToBuild[idx];
    const visible = state.meridianVisible[p.meridianId];
    (acupointMeshes.get(p.id) || []).forEach((mesh) => {
      mesh.visible = state.showAcupoints && visible && (state.showAllAcupoints || COMMON_ACUPOINT_IDS.has(p.id));
    });
      visibilityComputeMs += performance.now() - tOneStart;
      frameStart = await yieldByFrameBudget(frameStart, 6);
    }
    const tVisibilityWallDone = performance.now();
    const tRebuildDone = tVisibilityWallDone;
    const acupointsWallMs = tAcupointWallDone - tAcupointWallStart;
    const meridiansWallMs = tMeridianWallDone - tMeridianWallStart;
    const visibilityWallMs = tVisibilityWallDone - tVisibilityWallStart;
    perf.rebuild = {
      totalMs: tRebuildDone - tRebuildStart,
      acupointsComputeMs: acupointComputeMs,
      meridiansComputeMs: meridianComputeMs,
      visibilityComputeMs,
      acupointsWallMs,
      meridiansWallMs,
      visibilityWallMs,
      acupointsCount: totalAcupoints,
      meridiansCount: totalMeridians,
      acupointComputeAvgMs: totalAcupoints ? acupointComputeMs / totalAcupoints : 0,
      acupointWallAvgMs: totalAcupoints ? acupointsWallMs / totalAcupoints : 0,
      meridianComputeAvgMs: totalMeridians ? meridianComputeMs / totalMeridians : 0,
      meridianWallAvgMs: totalMeridians ? meridiansWallMs / totalMeridians : 0,
      cached: useCache,
      lightMode,
    };
    setLoadingProgress(86, '穴位/经络重建完成');
    markPerf('rebuildDone');
    finalizePerfIfReady();
    renderPerfPanel();
    applyVisibilityStates();
  } finally {
    rebuildingInProgress = false;
  }
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
      rebuildAcupointsAndMeridians({ lightMode: INITIAL_LIGHT_MODE });
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
  const region = classifyBodyRegion(pos);
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
  marker.visible = state.showPains;
  painGroup.add(marker);

  state.painMarks.push({ id, level, note, region, pos: pos.toArray(), createdAt: new Date().toISOString() });
  savePainMarksToCache();
  focusPainMark(id);
}

function classifyBodyRegion(pos) {
  tempBox.setFromObject(bodyRoot);
  const center = tempBox.getCenter(new THREE.Vector3());
  const size = tempBox.getSize(new THREE.Vector3());
  const nx = size.x > 1e-6 ? (pos.x - center.x) / (size.x * 0.5) : 0;
  const ny = size.y > 1e-6 ? (pos.y - tempBox.min.y) / size.y : 0;
  const nz = size.z > 1e-6 ? (pos.z - center.z) / (size.z * 0.5) : 0;
  const side = nx >= 0 ? '左' : '右';

  if (ny >= 0.88) return '头颈';
  if (Math.abs(nx) <= 0.2 && ny >= 0.7) return nz > 0 ? '前颈/胸上' : '后颈/上背';
  if (Math.abs(nx) <= 0.28 && ny >= 0.5) return nz > 0 ? '胸腹' : '背腰';
  if (Math.abs(nx) <= 0.2 && ny >= 0.28) return nz > 0 ? '下腹/骨盆' : '腰骶';
  if (Math.abs(nx) <= 0.24 && ny < 0.28) return '会阴/下肢根部';

  if (Math.abs(nx) >= 0.62) {
    if (ny >= 0.68) return `${side}上臂`;
    if (ny >= 0.52) return `${side}前臂`;
    return `${side}手`;
  }

  if (Math.abs(nx) >= 0.3) {
    if (ny >= 0.58) return `${side}肩部`;
    if (ny >= 0.35) return `${side}胸胁`;
    if (ny >= 0.18) return `${side}髋部`;
    return `${side}大腿根`;
  }

  if (ny >= 0.2) return `${side}大腿`;
  if (ny >= 0.08) return `${side}小腿`;
  return `${side}足部`;
}

function updatePainMarkHighlight() {
  painGroup.children.forEach((mesh) => {
    mesh.visible = state.showPains;
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

function updateToolButtons() {
  const apBtn = document.getElementById('toggle-acupoints-btn');
  const painBtn = document.getElementById('toggle-pains-btn');
  const showAllBtn = document.getElementById('show-all-btn');
  if (apBtn) apBtn.classList.toggle('active', state.showAcupoints);
  if (painBtn) painBtn.classList.toggle('active', state.showPains);
  if (showAllBtn) {
    showAllBtn.textContent = state.showAllAcupoints
      ? '切回常用穴位和经络（更流畅）'
      : '显示全部穴位和经络（可能较卡）';
  }
}

function applyVisibilityStates() {
  acupoints.forEach((p) => {
    const visibleByFilter = !!state.meridianVisible[p.meridianId];
    const visibleByInit = state.showAllAcupoints || COMMON_ACUPOINT_IDS.has(p.id) || p.id === state.selectedAcupointId;
    (acupointMeshes.get(p.id) || []).forEach((mesh) => {
      mesh.visible = state.showAcupoints && visibleByFilter && visibleByInit;
    });
  });
  meridians.forEach((m) => {
    (meridianMeshes.get(m.id) || []).forEach((mesh) => {
      mesh.visible = state.showAcupoints && !!state.meridianVisible[m.id];
    });
  });
  updatePainMarkHighlight();
  updateToolButtons();
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
  renderAcupointList(document.getElementById('search-input').value);
  renderPainList();

  if (highlightedAcupointId && highlightedAcupointId !== id) {
    (acupointMeshes.get(highlightedAcupointId) || []).forEach((mesh) => {
      mesh.scale.setScalar(0.03);
      if (mesh.userData.mat) {
        mesh.userData.mat.opacity = 0.65;
        mesh.userData.mat.color.set(0xffefb6);
      }
      const meta = mesh.userData?.meta;
      if (meta) {
        mesh.visible = state.showAcupoints
          && state.meridianVisible[meta.meridianId]
          && (state.showAllAcupoints || COMMON_ACUPOINT_IDS.has(meta.id));
      }
    });
  }

  (acupointMeshes.get(id) || []).forEach((mesh) => {
    mesh.visible = state.showAcupoints;
    mesh.scale.setScalar(0.05);
    if (mesh.userData.mat) {
      mesh.userData.mat.opacity = 0.95;
      mesh.userData.mat.color.set(0xffcf6c);
    }
  });
  highlightedAcupointId = id;

  let targetPos = acupointWorldPos.get(`${id}-L`) || acupointWorldPos.get(`${id}-R`);
  if (!targetPos) {
    const point = acupoints.find((p) => p.id === id);
    if (point) {
      setLoadingProgress(93, `补点中... ${id}`);
      if (loadingOverlay) loadingOverlay.classList.remove('hidden');
      const pair = acupointMeshes.get(id) || [];
      const leftPos = sampleSurfacePoint(point.anchor, 'L');
      if (leftPos) {
        const leftMesh = createAcupointMesh(leftPos, point, 'L');
        leftMesh.visible = state.showAcupoints;
        leftMesh.scale.setScalar(0.05);
        if (leftMesh.userData.mat) {
          leftMesh.userData.mat.opacity = 0.95;
          leftMesh.userData.mat.color.set(0xffcf6c);
        }
        acupointGroup.add(leftMesh);
        pair.push(leftMesh);
        acupointWorldPos.set(`${id}-L`, leftPos.clone());
        targetPos = leftPos.clone();
      }
      acupointMeshes.set(id, pair);
      setLoadingProgress(96, `补点完成 ${id}`);
      if (loadingOverlay) setTimeout(() => loadingOverlay.classList.add('hidden'), 160);
    }
  }
  if (!targetPos) return;
  tweenCameraTo(targetPos, 0.78);
  applyVisibilityStates();
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
      applyVisibilityStates();
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
      const regionLine = mark.region ? `部位: ${mark.region}<br/>` : '';
      li.innerHTML = `<strong>${mark.id}</strong><br/>${regionLine}疼痛等级: ${mark.level}<br/><small>${when}</small>`;
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
  levelLabel.textContent = `疼痛等级 (1-10)${selected.region ? ` · ${selected.region}` : ''}`;
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

const exportPainBtn = document.getElementById('export-pain-btn');
if (exportPainBtn) {
  exportPainBtn.addEventListener('click', () => exportPainMarks());
}

const clearPainBtn = document.getElementById('clear-pain-btn');
if (clearPainBtn) {
  clearPainBtn.addEventListener('click', () => {
    if (!state.painMarks.length) return;
    const ok = window.confirm(`确认清空 ${state.painMarks.length} 条疼痛点记录？`);
    if (!ok) return;
    clearAllPainMarks();
  });
}

const perfToggleBtn = document.getElementById('toggle-perf-btn');
if (perfToggleBtn && perfPanel) {
  perfPanel.classList.add('hidden');
  perfToggleBtn.addEventListener('click', () => {
    perfPanel.classList.toggle('hidden');
  });
}

const toggleAcupointsBtn = document.getElementById('toggle-acupoints-btn');
if (toggleAcupointsBtn) {
  toggleAcupointsBtn.addEventListener('click', () => {
    state.showAcupoints = !state.showAcupoints;
    applyVisibilityStates();
  });
}

const togglePainsBtn = document.getElementById('toggle-pains-btn');
if (togglePainsBtn) {
  togglePainsBtn.addEventListener('click', () => {
    state.showPains = !state.showPains;
    applyVisibilityStates();
  });
}

const resetViewBtn = document.getElementById('reset-view-btn');
if (resetViewBtn) {
  resetViewBtn.addEventListener('click', () => {
    cameraTween = null;
    controls.target.set(0, 1.0, 0);
    camera.position.set(1.8, 1.35, 2.3);
    controls.update();
  });
}

const showAllBtn = document.getElementById('show-all-btn');
if (showAllBtn) {
  showAllBtn.addEventListener('click', async () => {
    if (rebuildingInProgress) return;
    state.showAllAcupoints = !state.showAllAcupoints;
    setLoadingProgress(
      60,
      state.showAllAcupoints
        ? '正在切换到全部穴位/经络，可能较卡顿，请稍候...'
        : '正在切换到常用穴位/经络...'
    );
    if (loadingOverlay) loadingOverlay.classList.remove('hidden');
    showAllBtn.disabled = true;
    showAllBtn.classList.add('is-loading');
    try {
      await rebuildAcupointsAndMeridians({ lightMode: !state.showAllAcupoints });
      if (state.selectedAcupointId) focusAcupoint(state.selectedAcupointId);
      else applyVisibilityStates();
    } finally {
      showAllBtn.disabled = false;
      showAllBtn.classList.remove('is-loading');
      if (loadingOverlay) setTimeout(() => loadingOverlay.classList.add('hidden'), 160);
    }
  });
}

const leftSidebar = document.querySelector('.sidebar.left');
const rightSidebar = document.querySelector('.sidebar.right');
const toggleLeftBtn = document.getElementById('toggle-left-btn');
const toggleRightBtn = document.getElementById('toggle-right-btn');
const mobileBackdrop = document.getElementById('mobile-backdrop');

function closeMobilePanels() {
  if (leftSidebar) leftSidebar.classList.remove('show');
  if (rightSidebar) rightSidebar.classList.remove('show');
  document.body.classList.remove('panel-open');
}

function syncPanelOpenFlag() {
  const open = (leftSidebar && leftSidebar.classList.contains('show')) ||
    (rightSidebar && rightSidebar.classList.contains('show'));
  document.body.classList.toggle('panel-open', !!open);
}

function applyMobileMode() {
  const isMobile = window.innerWidth <= 1280 || window.matchMedia('(pointer: coarse)').matches;
  document.body.classList.toggle('mobile-mode', isMobile);
  if (!isMobile) closeMobilePanels();
}
applyMobileMode();

if (toggleLeftBtn && leftSidebar && rightSidebar) {
  toggleLeftBtn.addEventListener('click', () => {
    const next = !leftSidebar.classList.contains('show');
    rightSidebar.classList.remove('show');
    leftSidebar.classList.toggle('show', next);
    syncPanelOpenFlag();
  });
}

if (toggleRightBtn && leftSidebar && rightSidebar) {
  toggleRightBtn.addEventListener('click', () => {
    const next = !rightSidebar.classList.contains('show');
    leftSidebar.classList.remove('show');
    rightSidebar.classList.toggle('show', next);
    syncPanelOpenFlag();
  });
}

if (mobileBackdrop) {
  mobileBackdrop.addEventListener('click', () => closeMobilePanels());
}

function bindSwipeClose(panel, side) {
  if (!panel) return;
  let startX = 0;
  let startY = 0;
  let active = false;
  panel.addEventListener('touchstart', (e) => {
    if (!panel.classList.contains('show')) return;
    const t = e.changedTouches[0];
    startX = t.clientX;
    startY = t.clientY;
    active = true;
  }, { passive: true });
  panel.addEventListener('touchend', (e) => {
    if (!active) return;
    active = false;
    const t = e.changedTouches[0];
    const dx = t.clientX - startX;
    const dy = Math.abs(t.clientY - startY);
    if (dy > 40) return;
    if (side === 'left' && dx < -48) closeMobilePanels();
    if (side === 'right' && dx > 48) closeMobilePanels();
  }, { passive: true });
}

bindSwipeClose(leftSidebar, 'left');
bindSwipeClose(rightSidebar, 'right');

window.addEventListener('resize', () => {
  applyMobileMode();
});

renderMeridianFilters();
renderAcupointList('');
loadPainMarksFromCache();
renderPainList();
renderSelectedPainPanel();
loadAcupointTranslations();
applyVisibilityStates();

window.addEventListener('resize', () => {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
});

function animate() {
  fpsFrameCount += 1;
  const now = performance.now();
  const winMs = now - fpsWindowStart;
  if (winMs >= 1000) {
    const fps = (fpsFrameCount * 1000) / winMs;
    const maxDpr = Math.min(window.devicePixelRatio, 2);
    if (fps < 28) adaptivePixelRatio = Math.max(1, adaptivePixelRatio - 0.1);
    else if (fps > 52) adaptivePixelRatio = Math.min(maxDpr, adaptivePixelRatio + 0.05);
    renderer.setPixelRatio(adaptivePixelRatio);
    fpsFrameCount = 0;
    fpsWindowStart = now;
  }
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
