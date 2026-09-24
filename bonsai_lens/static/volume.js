// J-Volume: the per-column top-k maps stacked along the sequence into a 3D volume.
//   x = token position, y = layer (L0 at the bottom, model output on top), z = rank 1..10 (rank 1 in front)
// One voxel per (position, layer, rank); size and brightness scale with the lens probability.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const UI = window.lensUI;
const $ = (id) => document.getElementById(id);
const SX = 1.25, SY = 0.55, SZ = 1.1;   // spacing per position / layer / rank
const K = 10;

const host = $("volwrap");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
host.prepend(renderer.domElement);
const labels = new CSS2DRenderer();
Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
host.prepend(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.6, 1, 0.8);
scene.add(sun);

let mesh = null, labelGroup = null, selBox = null, built = null, dims = null;
const opts = { color: "prob", thr: 0, actual: false };  // thr set from the slider below
const dummy = new THREE.Object3D();
const col = new THREE.Color();

const cssHsl = (v) => {  // "210 85% 62%" -> THREE.Color
  const [h, s, l] = UI.css(v).split(/\s+/).map(parseFloat);
  return new THREE.Color().setHSL(h / 360, s / 100, l / 100);
};
const hashHue = (id) => ((Math.imul(id, 2654435761) >>> 0) % 360) / 360;

function voxelPos(t, li, k) {
  const { T, NL } = dims;
  return [(t - (T - 1) / 2) * SX, (li - (NL - 1) / 2) * SY, ((K - 1) / 2 - k) * SZ];
}

function build() {
  const data = UI.data;
  if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  if (labelGroup) { labelGroup.traverse((o) => o.element?.remove()); scene.remove(labelGroup); }
  const T = data.tokens.length, NL = data.layers.length;
  dims = { T, NL };
  mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }), T * NL * K);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(mesh);

  // Axis labels: tokens along the bottom, layers up the left side, ranks along the depth.
  labelGroup = new THREE.Group();
  const lab = (text, cls, [x, y, z]) => {
    const d = document.createElement("div");
    d.className = "vlab " + cls; d.textContent = text;
    const o = new CSS2DObject(d); o.position.set(x, y, z); labelGroup.add(o);
  };
  const every = Math.max(1, Math.ceil(T / 60));
  for (let t = 0; t < T; t += every) {
    const [x, y, z] = voxelPos(t, 0, 0);
    lab(UI.show(data.tokens[t]), "tok" + (t >= data.n_prompt ? " gen" : ""), [x, y - 1.6, z + 0.8]);
  }
  for (let li = 0; li < NL; li++) {
    const last = li === NL - 1;
    if (li % 8 && !last) continue;
    const [x, y, z] = voxelPos(0, li, 0);
    lab(last ? "out" : "L" + data.layers[li], "lay", [x - 2.2, y, z]);
  }
  for (let k = 0; k < K; k++) {
    const [x, y, z] = voxelPos(T - 1, 0, k);
    lab(String(k + 1), "rank", [x + 1.4, y - 1.0, z]);
  }
  scene.add(labelGroup);
  built = data;
  update();
  resetCamera();
  updateSelection();
}

function update() {
  if (!mesh) return;
  const data = UI.data, { T, NL } = dims;
  const accent = cssHsl("--accent"), heat = cssHsl("--heat"), dim = new THREE.Color(0x3a3a38);
  let i = 0;
  for (let t = 0; t < T; t++) {
    const actual = t + 1 < T ? data.ids[t + 1] : null;
    for (let li = 0; li < NL; li++) {
      const c = data.cells[li][t];
      for (let k = 0; k < K; k++, i++) {
        const p = c.p[k] ?? 0, id = c.ids[k];
        const hit = opts.actual && id === actual;
        const show = p >= opts.thr || hit;
        const s = show ? Math.max(hit ? 0.35 : 0.1, Math.sqrt(p)) : 0;
        dummy.position.set(...voxelPos(t, li, k));
        dummy.scale.set(s * SX * 0.9, s * SY * 0.95, s * SZ * 0.9);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        const b = Math.sqrt(p);
        if (hit) col.copy(heat);
        else if (opts.color === "token") col.setHSL(hashHue(id), 0.7, 0.25 + 0.4 * b);
        else col.copy(accent).multiplyScalar(0.25 + 0.95 * b);
        if (opts.actual && !hit) col.lerp(dim, 0.75);
        mesh.setColorAt(i, col);
      }
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();
}

function updateSelection() {
  if (selBox) { scene.remove(selBox); selBox.geometry.dispose(); selBox = null; }
  const sel = UI.sel;
  if (!sel || !dims || built !== UI.data) return;
  const [li, t] = sel;
  const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(SX, SY, SZ * K));
  selBox = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: new THREE.Color(UI.css("--sel")) }));
  const [x, y] = voxelPos(t, li, 0);
  selBox.position.set(x, y, 0);
  scene.add(selBox);
}

function resetCamera() {
  if (!dims) return;
  const w = dims.T * SX, h = dims.NL * SY;
  // Three-quarter view from the front-right and slightly above, so the rank axis (depth) is visible.
  const dist = Math.max(w, h) * 1.05 + 25;
  camera.position.set(dist * 0.62, h * 0.35, dist * 0.78);
  controls.target.set(0, 0, 0);
  controls.update();
}

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  if (!w || !h) return;
  renderer.setSize(w, h); labels.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);

// ---- hover + click -------------------------------------------------------
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let down = null;
function pick(e) {
  if (!mesh) return null;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(mesh)[0];
  if (!hit) return null;
  const i = hit.instanceId, per = dims.NL * K;
  const t = Math.floor(i / per), li = Math.floor((i % per) / K), k = i % K;
  return { t, li, k };
}
renderer.domElement.addEventListener("pointermove", (e) => {
  const tip = $("tip"), h = pick(e);
  if (!h) { tip.style.display = "none"; return; }
  const data = UI.data, c = data.cells[h.li][h.t], id = c.ids[h.k], tok = data.vocab[id];
  const g = UI.gloss[id] && UI.foreign(tok) ? ` → <b>${UI.esc(UI.gloss[id].en)}</b>` : "";
  const isOut = h.li === dims.NL - 1;
  const actual = h.t + 1 < dims.T && data.ids[h.t + 1] === id ? " · <b>actual next token</b>" : "";
  tip.innerHTML = `<span class="tok">${UI.esc(UI.show(tok))}</span>${g}` +
    `<div class="src">${isOut ? "output" : "L" + data.layers[h.li]} · pos ${h.t} (${UI.esc(UI.show(data.tokens[h.t]))}) · rank ${h.k + 1} · p = ${c.p[h.k]}${actual}</div>`;
  tip.style.display = "block";
  tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
});
renderer.domElement.addEventListener("pointerleave", () => { $("tip").style.display = "none"; });
renderer.domElement.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;  // was a drag
  const h = pick(e);
  if (h) UI.selectCell(h.li, h.t);
});

// ---- controls + wiring ---------------------------------------------------
const thrLabel = () => { $("vthrv").textContent = opts.thr ? `p ≥ ${opts.thr < 0.01 ? opts.thr.toExponential(0) : opts.thr.toFixed(2)}` : "all"; };
// Slider is log scale over p in [1e-4, 0.5]; 0 shows everything.
const thrFromSlider = (v) => (v === 0 ? 0 : Math.pow(10, -4 + (v / 100) * Math.log10(0.5 / 1e-4)));
opts.thr = thrFromSlider(+$("vthr").value);
$("vthr").addEventListener("input", (e) => { opts.thr = thrFromSlider(+e.target.value); thrLabel(); update(); });
$("vcolor").addEventListener("change", (e) => { opts.color = e.target.value; update(); });
$("vactual").addEventListener("change", (e) => { opts.actual = e.target.checked; update(); });
$("vreset").addEventListener("click", resetCamera);
thrLabel();

let active = false;
function setActive(on) {
  active = on;
  renderer.setAnimationLoop(on ? () => { controls.update(); renderer.render(scene, camera); labels.render(scene, camera); } : null);
  if (on) { resize(); if (UI.data && built !== UI.data) build(); }
}
window.addEventListener("lens:view", (e) => setActive(e.detail === "volume"));
window.addEventListener("lens:data", () => { if (active) build(); });
window.addEventListener("lens:select", updateSelection);
