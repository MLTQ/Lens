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
// z-index + isolation: CSS2DRenderer gives every label its own z-index; keep them all in one
// layer below the overlay panels (atom card, legends, controls).
Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "1", isolation: "isolate" });
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
  buildArcs();
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
        if (opts.flowDim) col.lerp(dim, 0.7);
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
  if (arcLines && mesh) {  // set the ray for this pointer position, then try the arcs
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    if (hoverArcs(e)) return;
  }
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

// ---- attention / memory arcs -----------------------------------------------------------
// Arcs run from a source position s to the destination t at the layer where the contribution is
// written, in the front (rank 1) plane, bulging toward the viewer. Colour: head hue (per-head) or
// mechanism (merged: orange = softmax attention, cyan = DeltaNet memory). Alpha/width ~ strength.
const arcMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false,
  vertexShader: `attribute vec3 color; attribute float alpha; varying vec3 vC; varying float vA;
    void main() { vC = color; vA = alpha; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `varying vec3 vC; varying float vA; void main() { if (vA <= 0.0) discard; gl_FragColor = vec4(vC, vA); }`,
});
const arcGroup = new THREE.Group();
let arcLines = null, arcMeta = [];  // one meta record per drawn arc (14 segments each), for hover
scene.add(arcGroup);
const ATTN_COL = new THREE.Color("#ff9d57"), GDN_COL = new THREE.Color("#57d0ff");
function clearArcs() {
  arcGroup.children.slice().forEach((o) => { o.element?.remove(); arcGroup.remove(o); o.geometry?.dispose(); });
}
function arcCurve(s, t, li, push) {  // quadratic curve s -> t at layer li, front plane
  const [x0, y, z0] = voxelPos(s, li, 0), [x1] = voxelPos(t, li, 0);
  const bulge = SZ * 1.5 + Math.abs(x1 - x0) * 0.35;
  let px = x0, pz = z0;
  for (let i = 1; i <= 14; i++) {
    const u = i / 14, xm = (x0 + x1) / 2;
    const x = (1 - u) * (1 - u) * x0 + 2 * (1 - u) * u * xm + u * u * x1;
    const z = z0 + 4 * u * (1 - u) * bulge;
    push(px, y, pz, x, y, z, u);
    px = x; pz = z;
  }
  return [(x0 + x1) / 2, y, z0 + bulge];
}
function buildArcs() {
  clearArcs();
  arcLines = null; arcMeta = [];
  const A = UI.attn, st = UI.atState, data = UI.data;
  if (!A || !st.on || !dims || built !== data) return;
  const v = [], c = [], a = [];
  const add = (s, t, li, color, alpha, meta = {}) => (arcMeta.push({ s, t, li, ...meta }), arcCurve(s, t, li, (x0, y0, z0, x1, y1, z1, u) => {
    v.push(x0, y0, z0, x1, y1, z1); c.push(color.r, color.g, color.b, color.r, color.g, color.b);
    a.push(alpha * (0.35 + 0.65 * (u - 1 / 14)), alpha * (0.35 + 0.65 * u));  // brighter toward the destination
  }));
  const hueCol = (h, H) => new THREE.Color().setHSL(UI.headHue(h, H) / 360, 0.85, 0.6);
  // (1) the selected cell's incoming arcs
  if (UI.sel) {
    const [li, t] = UI.sel;
    for (const arc of UI.attnArcs(li, t)) {
      if (arc.s === t) continue;
      const L = A.layers[data.layers[li]];
      add(arc.s, t, li, arc.merged ? (L.kind === "gdn" ? GDN_COL : ATTN_COL) : hueCol(arc.h, L.heads), Math.min(1, 0.4 + arc.share * 2),
        { h: arc.merged ? -1 : arc.h, share: arc.share, kind: L.kind, sign: arc.sign });
    }
  }
  // (2) flow: strongest cross-position writes of every layer, relative to the residual they land in
  if ($("vflow").checked) {
    // Rank every candidate arc by the share of the destination's residual it writes
    // (share of the layer's incoming contribution x how much the layer writes there), keep the top N.
    const N = Math.max(10, +$("vflowmin").value || 250), T = A.T, cand = [];
    for (let li = 0; li < data.layers.length; li++) {
      const L = A.layers[data.layers[li]];
      if (!L) continue;
      for (let t = 0; t < T; t++) {
        const str = L.strength[t];
        if (st.merged) {
          for (let k = 0; k < A.KM; k++) {
            const s = L.msrcA[t * A.KM + k];
            if (s !== t) cand.push({ s, t, li, w: L.mshareA[t * A.KM + k] * str, share: L.mshareA[t * A.KM + k], h: -1, kind: L.kind, color: L.kind === "gdn" ? GDN_COL : ATTN_COL });
          }
        } else {
          for (let h = 0; h < L.heads; h++) for (let k = 0; k < A.K; k++) {
            const j = (h * T + t) * A.K + k, s = L.srcA[j];
            if (s !== t) cand.push({ s, t, li, w: L.shareA[j] * str, share: L.shareA[j], h, H: L.heads, kind: L.kind, sign: L.signA[j] });
          }
        }
      }
    }
    // Equal quota per layer: early layers write into a still-small residual, so a global ranking
    // would put every arc at the bottom. Within a layer, arcs are ranked and faded by strength.
    const perLayer = Math.max(1, Math.ceil(N / data.layers.length)), byLayer = new Map();
    for (const e of cand) (byLayer.get(e.li) || byLayer.set(e.li, []).get(e.li)).push(e);
    for (const list of byLayer.values()) {
      list.sort((x, y) => y.w - x.w);
      const top = list.slice(0, perLayer), wmax = top[0]?.w || 1;
      for (const e of top) add(e.s, e.t, e.li, e.color || hueCol(e.h, e.H), Math.min(1, 0.25 + 0.75 * Math.sqrt(e.w / wmax)),
        { h: e.h, share: e.share, kind: e.kind, sign: e.sign, flow: true });
    }
    // labels: the strongest decoded messages (merged contributions, decoded with the J-lens)
    let shown = 0;
    for (const f of A.flow) {
      if (shown >= 18) break;
      const li = data.layers.indexOf(f.l);
      if (li < 0) continue;
      const [x0, y] = voxelPos(f.s, li, 0), [x1] = voxelPos(f.t, li, 0);
      const [, , z0] = voxelPos(f.s, li, 0);
      const d = document.createElement("div");
      d.className = "vlab flowlab";
      d.textContent = `L${f.l} ${UI.show(data.tokens[f.s]).trim()}→${UI.show(data.tokens[f.t]).trim()}: ${f.words.slice(0, 2).map((w) => UI.show(w).trim()).join(" ")}`;
      const o = new CSS2DObject(d);
      o.position.set((x0 + x1) / 2, y, z0 + SZ * 1.5 + Math.abs(x1 - x0) * 0.35);
      arcGroup.add(o);
      shown++;
    }
  }
  if (!v.length) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
  g.setAttribute("alpha", new THREE.Float32BufferAttribute(a, 1));
  arcLines = new THREE.LineSegments(g, arcMat);
  arcGroup.add(arcLines);
}

// Hover an arc: its structure at once, then what it carried (decoded on demand, cached).
const msgCache = new Map();
let msgTimer = null, hoverArc = null;
function arcTip(m, msg) {
  const data = UI.data, L = UI.attn?.layers[data.layers[m.li]];
  const tok = (i) => `<span class="tok">${UI.esc(UI.show(data.tokens[i]))}</span>`;
  const mech = m.kind === "gdn" ? "DeltaNet memory" : "softmax attention";
  const who = m.h >= 0 ? `head ${m.h}` : "all heads";
  const carried = msg ? (msg.error ? UI.esc(msg.error) : `carried: <b>${UI.esc(msg.words.map((w) => UI.show(w).trim()).join(" · "))}</b>`)
    : `<span class="src">decoding what it carried…</span>`;
  return `L${data.layers[m.li]} · ${mech} · ${who}${m.sign < 0 ? " · <b>negative</b> (net subtraction after overwrites)" : ""}` +
    `<div>${tok(m.s)} <span class="src">(pos ${m.s})</span> → ${tok(m.t)} <span class="src">(pos ${m.t})</span></div>` +
    `<div class="src">${m.share != null ? `${(m.share * 100).toFixed(1)}% of everything this layer brings into ${UI.esc(UI.show(data.tokens[m.t]))}` : ""}` +
    `${msg?.of_residual != null ? ` · ${(msg.of_residual * 100).toFixed(1)}% of the residual there` : ""}` +
    `${L ? ` · this layer's write there is ${UI.ratio(L.strength[m.t])} the size of the residual it reads` : ""}</div>` +
    `<div>${carried}</div>`;
}
function hoverArcs(e) {
  if (!arcLines) return false;
  ray.params.Line = { threshold: 0.35 };
  const hit = ray.intersectObject(arcLines)[0];
  const tip = $("tip");
  if (!hit) { hoverArc = null; return false; }
  const m = arcMeta[Math.floor(hit.index / 2 / 14)];
  if (!m) return false;
  const key = `${UI.data.layers[m.li]}:${m.t}:${m.s}:${m.h}`;
  hoverArc = key;
  tip.innerHTML = arcTip(m, msgCache.get(key));
  tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
  if (!msgCache.has(key)) {
    clearTimeout(msgTimer);
    msgTimer = setTimeout(async () => {
      const r = await (await fetch(`/api/attn_msg?l=${UI.data.layers[m.li]}&t=${m.t}&s=${m.s}&h=${m.h}`)).json();
      msgCache.set(key, r);
      if (hoverArc === key) tip.innerHTML = arcTip(m, r);
    }, 180);
  }
  return true;
}
$("vflow").addEventListener("change", async (e) => {
  if (e.target.checked) { await UI.ensureAttn(); opts.flowDim = true; } else opts.flowDim = false;
  update(); buildArcs();
});
$("vflowmin").addEventListener("change", buildArcs);
window.addEventListener("lens:attn", () => { if (active) buildArcs(); });
window.addEventListener("lens:data", () => msgCache.clear());  // messages belong to one run
// Screen position of an arc's apex (used for testing hover; harmless otherwise).
window.lensVolume = {
  get arcs() { return arcMeta.length; },
  arcScreen(i) {
    const m = arcMeta[i]; if (!m) return null;
    const [x0, y, z0] = voxelPos(m.s, m.li, 0), [x1] = voxelPos(m.t, m.li, 0);
    const v = new THREE.Vector3((x0 + x1) / 2, y, z0 + SZ * 1.5 + Math.abs(x1 - x0) * 0.35).project(camera);
    const r = renderer.domElement.getBoundingClientRect();
    return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height];
  },
};

let active = false;
function setActive(on) {
  active = on;
  renderer.setAnimationLoop(on ? () => { controls.update(); renderer.render(scene, camera); labels.render(scene, camera); } : null);
  if (on) { resize(); if (UI.data && built !== UI.data) build(); }
}
window.addEventListener("lens:view", (e) => setActive(e.detail === "volume"));
window.addEventListener("lens:data", () => { if (active) build(); });
window.addEventListener("lens:select", () => { updateSelection(); if (active) { UI.atState.on && UI.ensureAttn()?.then?.(buildArcs); buildArcs(); } });
