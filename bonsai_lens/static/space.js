// Canonical token space: every vocabulary token at its UMAP position of the model's readout
// direction u_v = gamma * W_eff[v] (the space all J-lens layers are decoded in).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const UI = window.lensUI;
const $ = (id) => document.getElementById(id);
const R = 60;  // world radius of the normalized [-1, 1] layout

const host = $("spacewrap");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
host.prepend(renderer.domElement);
const labels = new CSS2DRenderer();
Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
host.prepend(labels.domElement);
const labelGroup = new THREE.Group();
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Category palette (index-matched to meta.categories by name).
const PALETTE = {
  "english/latin": "#8fa3b8", "latin (accented)": "#6fc2c9", "code/markup": "#e3c14a", "digits": "#ffffff",
  "punctuation/symbol": "#b88ae6", "whitespace": "#5a5a5a", "cjk": "#e8594f", "japanese": "#f28cb1",
  "korean": "#f0a04b", "cyrillic": "#5fbf6a", "arabic": "#3f8fe0", "hebrew": "#2f6fb0", "greek": "#a1d99b",
  "thai": "#c77d4a", "indic": "#d6a3ff", "other script": "#9e9e6e", "byte fragment": "#ff3d7f",
  "special": "#ffffff", "padding": "#333333",
};

let meta = null, c3 = null, c2 = null, points = null, geom = null;
const st = { dim: 3, color: "cat", hidden: new Set(), query: "", matches: null, focus: null };
let active = false, loaded = false;

const vert = `
attribute vec3 color; attribute float size; attribute float alpha;
varying vec3 vColor; varying float vAlpha; uniform float scale;
void main() {
  vColor = color; vAlpha = alpha;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // Base cloud scales with depth; highlighted points (size >= 10) are a fixed pixel size.
  gl_PointSize = size >= 10.0 ? size : clamp(size * scale / -mv.z, 1.0, 6.0);
  gl_Position = projectionMatrix * mv;
}`;
const frag = `
varying vec3 vColor; varying float vAlpha;
void main() {
  vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
  if (r > 0.25 || vAlpha <= 0.0) discard;
  gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.25, 0.15, r));
}`;
const material = new THREE.ShaderMaterial({
  vertexShader: vert, fragmentShader: frag, transparent: true, depthWrite: false,
  uniforms: { scale: { value: 120 } },
});

async function load() {
  $("spstatus").textContent = "Loading token space…";
  const r = await fetch("/space/meta.json");
  if (!r.ok) { $("spstatus").textContent = "No token space yet: run scripts/embed_vocab.py and copy lenses/space here."; return; }
  meta = await r.json();
  c3 = new Float32Array(await (await fetch("/space/coords3.f32")).arrayBuffer());
  c2 = new Float32Array(await (await fetch("/space/coords2.f32")).arrayBuffer());
  const n = meta.n;
  geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setAttribute("size", new THREE.BufferAttribute(new Float32Array(n), 1));
  geom.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(n), 1));
  points = new THREE.Points(geom, material);
  scene.add(points, labelGroup);
  // log-norm quantiles for the "readout norm" coloring
  const ln = meta.norm.map((x) => Math.log(x + 1e-6)).sort((a, b) => a - b);
  meta.lnLo = ln[Math.floor(ln.length * 0.01)]; meta.lnHi = ln[Math.floor(ln.length * 0.99)];
  buildLegend();
  loaded = true;
  setPositions(); setStyle(); resetCamera(); buildTrace();
  $("spstatus").textContent = `${n.toLocaleString()} tokens · ${meta.space}`;
}

function setPositions() {
  const pos = geom.attributes.position.array, n = meta.n;
  for (let i = 0; i < n; i++) {
    if (st.dim === 3) { pos[3 * i] = c3[3 * i] * R; pos[3 * i + 1] = c3[3 * i + 1] * R; pos[3 * i + 2] = c3[3 * i + 2] * R; }
    else { pos[3 * i] = c2[2 * i] * R; pos[3 * i + 1] = c2[2 * i + 1] * R; pos[3 * i + 2] = 0; }
  }
  geom.attributes.position.needsUpdate = true;
  geom.computeBoundingSphere();
}

const tmp = new THREE.Color();
function ramp(t) {  // dark violet -> teal -> yellow
  return tmp.setHSL(0.78 - 0.62 * t, 0.75, 0.18 + 0.5 * t);
}
function setStyle() {
  const { color, size, alpha } = geom.attributes, n = meta.n, cats = meta.categories;
  const catCol = cats.map((c) => new THREE.Color(PALETTE[c] || "#999"));
  const m = st.matches;
  for (let i = 0; i < n; i++) {
    const cat = cats[meta.cat[i]];
    let c;
    if (st.color === "norm") c = ramp(Math.min(1, Math.max(0, (Math.log(meta.norm[i] + 1e-6) - meta.lnLo) / (meta.lnHi - meta.lnLo))));
    else c = catCol[meta.cat[i]];
    color.array[3 * i] = c.r; color.array[3 * i + 1] = c.g; color.array[3 * i + 2] = c.b;
    const hidden = st.hidden.has(cat);
    if (m) {
      const hit = m.has(i), foc = i === st.focus;
      size.array[i] = foc ? 18 : hit ? 11 : 1.2; alpha.array[i] = hidden && !hit ? 0 : hit || foc ? 1 : 0.06;
      if (foc) { color.array[3 * i] = 1; color.array[3 * i + 1] = 1; color.array[3 * i + 2] = 1; }
      else if (hit) { color.array[3 * i] = 1; color.array[3 * i + 1] = 0.55; color.array[3 * i + 2] = 0.15; }
    } else { size.array[i] = 1.6; alpha.array[i] = hidden ? 0 : tracing() ? 0.05 : 0.75; }
  }
  color.needsUpdate = size.needsUpdate = alpha.needsUpdate = true;
  setLabels();
}

// Text labels for highlighted tokens (search hits or neighbours), capped for legibility.
function setLabels() {
  labelGroup.children.slice().forEach((o) => { o.element.remove(); labelGroup.remove(o); });
  if (!st.matches) return;
  const p = geom.attributes.position.array;
  const ids = [...(st.focus !== null ? [st.focus] : []), ...st.matches].slice(0, 60);
  for (const i of ids) {
    const d = document.createElement("div");
    d.className = "splab" + (i === st.focus ? " foc" : "");
    const g = UI.gloss[i] && UI.foreign(meta.tokens[i]) ? ` (${UI.gloss[i].en})` : "";
    d.textContent = UI.show(meta.tokens[i]) + g;
    const o = new CSS2DObject(d);
    o.position.set(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    labelGroup.add(o);
  }
}

function buildLegend() {
  const cats = meta.categories.map((c, i) => [c, meta.category_counts[c]]).sort((a, b) => b[1] - a[1]);
  $("splegend").innerHTML = cats.map(([c, k]) =>
    `<span class="lg" data-cat="${UI.esc(c)}"><i style="background:${PALETTE[c] || "#999"}"></i>${UI.esc(c)} <em>${k.toLocaleString()}</em></span>`).join("");
}
$("splegend").addEventListener("click", (e) => {
  const el = e.target.closest(".lg"); if (!el || !loaded) return;
  const c = el.dataset.cat;
  st.hidden.has(c) ? st.hidden.delete(c) : st.hidden.add(c);
  el.classList.toggle("off", st.hidden.has(c));
  setStyle();
});

function flyTo(ids) {
  const p = geom.attributes.position.array, c = new THREE.Vector3();
  for (const i of ids) c.add(new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]));
  c.divideScalar(ids.length);
  const off = camera.position.clone().sub(controls.target).setLength(R * 0.6);
  controls.target.copy(c); camera.position.copy(c).add(off);
}

// Click a point: highlight its nearest neighbours in the PCA-128 readout space.
async function showNeighbors(i) {
  const r = await (await fetch(`/api/space_nn?id=${i}&k=30`)).json();
  st.focus = i;
  st.matches = new Set(r.neighbors.map((n) => n.id));
  const foreignIds = [i, ...st.matches].filter((j) => UI.foreign(meta.tokens[j]) && !UI.gloss[j]).slice(0, 8);
  if (foreignIds.length) {
    const g = await fetch("/api/translate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ids: foreignIds }) });
    Object.assign(UI.gloss, await g.json());
  }
  setStyle();
  $("spcount").textContent = `30 nearest to ${UI.show(meta.tokens[i])} (cosine in PCA-128; the layout distorts distances)`;
}

function search(q) {
  st.query = q.trim(); st.focus = null;
  if (!st.query) { st.matches = null; setStyle(); $("spcount").textContent = ""; return; }
  const ql = st.query.toLowerCase(), exact = new Set(), part = new Set();
  for (let i = 0; i < meta.n; i++) {
    const s = meta.tokens[i].trim().toLowerCase();
    if (!s) continue;
    if (s === ql) exact.add(i); else if (part.size < 3000 && s.includes(ql)) part.add(i);
  }
  st.matches = exact.size ? new Set([...exact, ...part]) : part;
  setStyle();
  $("spcount").textContent = `${exact.size} exact · ${st.matches.size} total`;
  const focus = exact.size ? exact : part;
  if (focus.size) flyTo([...focus]);  // fly to the centroid of the matches
}

function resetCamera() {
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, R * 2.4);
  if (st.dim === 3) camera.position.set(R * 1.4, R * 0.8, R * 1.8);
  controls.enableRotate = st.dim === 3;
  controls.update();
}

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  if (!w || !h) return;
  mainMat.resolution.set(w, h);
  renderer.setSize(w, h); labels.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  material.uniforms.scale.value = h * 0.22;  // ~2-3 px points at default distance
}
new ResizeObserver(resize).observe(host);

// ---- hover ---------------------------------------------------------------
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let pend = null, lastHover = -1, moveEvt = null;
renderer.domElement.addEventListener("pointermove", (e) => { moveEvt = e; });
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;  // drag, not click
  if (lastHover >= 0) showNeighbors(lastHover);
});
renderer.domElement.addEventListener("pointerleave", () => { moveEvt = null; $("tip").style.display = "none"; });
function hover() {
  const e = moveEvt; moveEvt = null;
  if (!e || !loaded) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  ray.params.Points.threshold = camera.position.distanceTo(controls.target) * 0.006;
  const tip = $("tip");
  if (tr.nodes) {
    const th = ray.intersectObject(tr.nodes).sort((a, b) => a.distanceToRay - b.distanceToRay)[0];
    if (th) {
      const n = tr.nodeInfo[th.index], tok = meta.tokens[n.id];
      const g = UI.gloss[n.id] && UI.foreign(tok) ? ` → <b>${UI.esc(UI.gloss[n.id].en)}</b>` : "";
      tip.innerHTML = `<span class="tok">${UI.esc(UI.show(tok))}</span>${g}<div class="src">${UI.esc(n.note)}</div>`;
      tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
      lastHover = n.id; return;
    }
  }
  const hits = ray.intersectObject(points).filter((h) => geom.attributes.alpha.array[h.index] > 0.05);
  if (!hits.length) { tip.style.display = "none"; lastHover = -1; return; }
  hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
  const i = hits[0].index, tok = meta.tokens[i];
  lastHover = i;
  const g = UI.gloss[i] && UI.foreign(tok) ? ` → <b>${UI.esc(UI.gloss[i].en)}</b>` : UI.foreign(tok) ? ` → <span class="src">translating…</span>` : "";
  tip.innerHTML = `<span class="tok">${UI.esc(UI.show(tok) || "(none)")}</span>${g}<div class="src">id ${i} · ${UI.esc(meta.categories[meta.cat[i]])} · readout norm ${meta.norm[i].toFixed(3)}</div>`;
  tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
  if (UI.foreign(tok) && !UI.gloss[i]) {
    clearTimeout(pend);
    pend = setTimeout(async () => {
      const res = await fetch("/api/translate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ids: [i] }) });
      Object.assign(UI.gloss, await res.json());
      if (lastHover === i) { moveEvt = e; }
    }, 250);
  }
}

// ---- wiring --------------------------------------------------------------
$("spdim").addEventListener("change", (e) => { st.dim = +e.target.value; if (loaded) { setPositions(); resetCamera(); buildTrace(); } });
$("spcolor").addEventListener("change", (e) => { st.color = e.target.value; if (loaded) setStyle(); });
$("spsearch").addEventListener("keydown", (e) => { if (e.key === "Enter" && loaded) search(e.target.value); });
$("spclear").addEventListener("click", () => { $("spsearch").value = ""; if (loaded) { search(""); resetCamera(); } });

function setActive(on) {
  active = on;
  renderer.setAnimationLoop(on ? (now) => {
    lineMat.uniforms.time.value = now / 1000 * 0.7;   // waves travel toward each segment's end
    mainMat.dashOffset = -now / 1000 * 1.6;          // gaps march in sequence order
    hover(); tickPlayback(now); controls.update(); renderer.render(scene, camera); labels.render(scene, camera); } : null);
  if (on) { resize(); if (!meta) load(); }
}
window.addEventListener("lens:view", (e) => setActive(e.detail === "space"));
// ---- step 3: trace the current run through the space ------------------------------
// Main line: each position's model prediction (top-1 of the output row), joined in
// sequence order and coloured by position. Branches: from that node to every other
// token the lens read out at the same position (top-k per cell, within the layer
// range, including the output row's runners-up). One edge per (position, token):
// alpha = peak probability, colour = probability-weighted mean layer.
const tr = { group: new THREE.Group(), nodes: null, nodeInfo: [], labels: [] };
scene.add(tr.group);
// Lines carry ``dist`` = distance from the segment start; a brightness wave moves toward
// the segment end over time, so every edge shows its direction (readout -> output).
const lineMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, uniforms: { time: { value: 0 } },
  vertexShader: `attribute vec3 color; attribute float alpha; attribute float dist; varying vec3 vC; varying float vA; varying float vD;
    void main() { vC = color; vA = alpha; vD = dist; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform float time; varying vec3 vC; varying float vA; varying float vD;
    void main() {
      if (vA <= 0.0) discard;
      float w = 0.5 + 0.5 * sin(6.2831853 * (vD / 5.0 - time));
      gl_FragColor = vec4(vC * (0.8 + 0.35 * w), vA * (0.5 + 0.5 * w));
    }`,
});
// Per-vertex distance attribute for a flat [x0,y0,z0,x1,y1,z1,...] segment list.
function segDist(v) {
  const d = [];
  for (let i = 0; i < v.length; i += 6) d.push(0, Math.hypot(v[i + 3] - v[i], v[i + 4] - v[i + 1], v[i + 5] - v[i + 2]));
  return d;
}
// Main line: screen-space fat lines (WebGL's native lines are always 1 px).
const mainMat = new LineMaterial({ linewidth: 3, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false,
  dashed: true, dashSize: 2.2, gapSize: 0.7 });  // marching gaps, animated via dashOffset
function mainLine(mv, mc) {
  const g = new LineSegmentsGeometry(); g.setPositions(mv); g.setColors(mc);
  const l = new LineSegments2(g, mainMat); l.computeLineDistances(); return l;
}
const nodeMat = material.clone();
nodeMat.uniforms = { scale: material.uniforms.scale };

const posColor = (f) => new THREE.Color().setHSL(((200 + 215 * f) % 360) / 360, 0.9, 0.62);   // blue→violet→magenta→orange→yellow
const layerColor = (f) => new THREE.Color().setHSL((168 - 22 * f) / 360, 0.75 - 0.35 * f, 0.24 + 0.66 * f); // dark teal→mint→white
const tracing = () => $("trOn").checked && !!UI.data && loaded;
const pThr = () => { const v = +$("trP").value; return v === 0 ? 0 : Math.pow(10, -3 + (v / 100) * 3 * 0.9); };  // 1e-3 .. ~0.5

function clearTrace() {
  tr.group.children.slice().forEach((o) => { tr.group.remove(o); o.geometry?.dispose(); });
  tr.labels.forEach((o) => { o.element.remove(); o.removeFromParent(); });
  tr.labels = []; tr.nodes = null; tr.nodeInfo = [];
  $("trlegend").style.display = "none";
}

function buildTrace() {
  clearTrace();
  $("trPv").textContent = `≥ ${pThr().toPrecision(1)}`;
  if (!tracing()) { setStyle(); renderStrip(-1, 0); return; }
  const data = UI.data, T = data.tokens.length, NL = data.layers.length, out = NL - 1;
  const P = geom.attributes.position.array;
  const at = (id) => new THREE.Vector3(P[3 * id], P[3 * id + 1], P[3 * id + 2]);
  const K = Math.max(1, Math.min(10, +$("trK").value || 3)), thr = pThr();
  const lo = Math.max(0, +$("trLo").value || 0), hi = Math.min(out, +$("trHi").value || out);
  const posF = (t) => (T > 1 ? t / (T - 1) : 0);
  const pred = [...Array(T)].map((_, t) => data.cells[out][t].ids[0]);

  // main line
  const mv = [], mc = [], ma = [];
  for (let t = 0; t + 1 < T; t++) {
    const a = at(pred[t]), b = at(pred[t + 1]), ca = posColor(posF(t)), cb = posColor(posF(t + 1));
    mv.push(a.x, a.y, a.z, b.x, b.y, b.z); mc.push(ca.r, ca.g, ca.b, cb.r, cb.g, cb.b); ma.push(1, 1);
  }
  // branches, aggregated per (position, token)
  const agg = new Map();
  for (let t = 0; t < T; t++) for (let li = lo; li <= hi; li++) {
    const c = data.cells[li][t];
    for (let k = 0; k < K; k++) {
      const p = c.p[k], id = c.ids[k];
      if (p < thr || id === pred[t]) continue;
      const key = t * 1e6 + id;
      const g = agg.get(key) || { t, id, pmax: 0, w: 0, lw: 0, layers: [] };
      g.pmax = Math.max(g.pmax, p); g.w += p; g.lw += p * li; g.layers.push(li === out ? "out" : "L" + data.layers[li]);
      agg.set(key, g);
    }
  }
  const bv = [], bc = [], ba = [];
  const nodes = new Map();  // id -> strongest appearance, for points + hover
  const addNode = (id, info) => { const n = nodes.get(id); if (!n || info.strength > n.strength) nodes.set(id, info); };
  for (const g of agg.values()) {
    const a = at(pred[g.t]), b = at(g.id), col = layerColor((g.lw / g.w) / out), al = Math.min(1, 0.08 + 0.92 * Math.sqrt(g.pmax));
    bv.push(b.x, b.y, b.z, a.x, a.y, a.z); bc.push(col.r, col.g, col.b, col.r, col.g, col.b); ba.push(al, al * 0.35);  // readout -> prediction
    addNode(g.id, { id: g.id, main: false, strength: g.pmax, color: col, alpha: al, size: 3 + 6 * Math.sqrt(g.pmax),
      note: `branch from pos ${g.t} (${UI.show(data.tokens[g.t])}) · peak p ${g.pmax.toFixed(3)} · ${g.layers.join(" ")}` });
  }
  pred.forEach((id, t) => addNode(id, { id, main: true, strength: 2 + t, color: posColor(posF(t)), alpha: 1, size: 13,
    note: `prediction at pos ${t} (after ${UI.show(data.tokens[t])})` + (pred.indexOf(id) !== t ? ` · also pos ${pred.indexOf(id)}` : "") }));

  const seg = (v, c, a) => {
    const gm = new THREE.BufferGeometry();
    gm.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    gm.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
    gm.setAttribute("alpha", new THREE.Float32BufferAttribute(a, 1));
    gm.setAttribute("dist", new THREE.Float32BufferAttribute(segDist(v), 1));
    return new THREE.LineSegments(gm, lineMat);
  };
  tr.group.add(seg(bv, bc, ba));
  if (mv.length) tr.group.add(mainLine(mv, mc));

  const list = [...nodes.values()].sort((a, b) => a.main - b.main);  // main nodes drawn last (on top)
  const ng = new THREE.BufferGeometry(), pv = [], pc = [], ps = [], pa = [];
  for (const n of list) { const p = at(n.id); pv.push(p.x, p.y, p.z); pc.push(n.color.r, n.color.g, n.color.b); ps.push(n.size); pa.push(n.alpha); }
  ng.setAttribute("position", new THREE.Float32BufferAttribute(pv, 3));
  ng.setAttribute("color", new THREE.Float32BufferAttribute(pc, 3));
  ng.setAttribute("size", new THREE.Float32BufferAttribute(ps, 1));
  ng.setAttribute("alpha", new THREE.Float32BufferAttribute(pa, 1));
  tr.nodes = new THREE.Points(ng, nodeMat); tr.nodeInfo = list;
  tr.group.add(tr.nodes);

  if ($("trLab").checked) for (const n of list.filter((n) => n.main)) {
    const d = document.createElement("div");
    d.className = "splab main"; d.style.color = "#" + n.color.getHexString();
    const tok = meta.tokens[n.id], g = UI.gloss[n.id] && UI.foreign(tok) ? ` (${UI.gloss[n.id].en})` : "";
    d.textContent = UI.show(tok) + g;
    const o = new CSS2DObject(d); o.position.copy(at(n.id)); tr.group.add(o); tr.labels.push(o);
  }
  const grad = (fn) => `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((f) => "#" + fn(f).getHexString()).join(",")})`;
  $("trlegend").innerHTML = `main line = model prediction, by position<div class="gr" style="background:${grad(posColor)}"></div>
    <div class="ends"><span>${UI.esc(UI.show(data.tokens[0]))}</span><span>${UI.esc(UI.show(data.tokens[T - 1]))}</span></div>
    <div style="margin-top:6px">branches = other readouts, by mean layer · alpha = confidence</div><div class="gr" style="background:${grad(layerColor)}"></div>
    <div class="ends"><span>L0</span><span>out</span></div>
    <div style="margin-top:4px">${pred.length} predictions · ${agg.size} branches</div>`;
  $("trlegend").style.display = "block";
  setStyle();
  if (tr.frameNext) { tr.frameNext = false; frameTrace(list.map((n) => n.id)); }
  if (!pb.plan) renderStrip(-1, T);
}

// Point the camera at the trace (only when a new run arrives, not on every control tweak).
function frameTrace(ids) {
  if (!ids.length) return;
  const P = geom.attributes.position.array, box = new THREE.Box3();
  for (const i of ids) box.expandByPoint(new THREE.Vector3(P[3 * i], P[3 * i + 1], P[3 * i + 2]));
  const c = box.getCenter(new THREE.Vector3()), r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 8);
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(c); camera.position.copy(c).addScaledVector(dir, r * 2.2);
}
["trOn", "trK", "trP", "trLo", "trHi", "trLab"].forEach((id) => $(id).addEventListener(id === "trP" ? "input" : "change", () => loaded && buildTrace()));
tr.frameNext = true;
window.addEventListener("lens:data", () => { tr.frameNext = true; if (loaded) buildTrace(); });

// ---- playback: the "4th dimension" -------------------------------------------------
// Per position t: (A) the input token lights up; (B) each layer's readouts pop in, in
// layer order; (C) rays grow from those nodes to the output prediction, which lights up
// and stays, extending the main line; (D) nodes and rays fade. Then t + 1.
const pb = { group: new THREE.Group(), playing: false, t: 0, start: 0, stopAt: Infinity, plan: null, labels: [] };
scene.add(pb.group);
const PH = { A: 150, POP: 12, C: 300, HOLD: 80, D: 300 };  // ms at 1×
const speed = () => +$("pbSpeed").value;

function planPlayback() {
  const data = UI.data, T = data.tokens.length, NL = data.layers.length, out = NL - 1;
  const K = Math.max(1, Math.min(10, +$("trK").value || 3)), thr = pThr();
  const lo = Math.max(0, +$("trLo").value || 0), hi = Math.min(out - 1, +$("trHi").value || out);
  const steps = [];
  for (let t = 0; t < T; t++) {
    const events = [];  // [{li, id, p}] in layer order
    for (let li = lo; li <= hi; li++) {
      const c = data.cells[li][t];
      for (let k = 0; k < K; k++) if (c.p[k] >= thr) events.push({ li, id: c.ids[k], p: c.p[k] });
    }
    const nLayers = hi - lo + 1;
    const durB = nLayers * PH.POP;
    steps.push({ t, input: data.ids[t], pred: data.cells[out][t].ids[0], events, lo, nLayers, durB,
      dur: PH.A + durB + PH.C + PH.HOLD + PH.D });
  }
  return { T, out, steps };
}

function clearPlayback() {
  pb.group.children.slice().forEach((o) => { pb.group.remove(o); o.geometry?.dispose(); });
  pb.labels.forEach((o) => { o.element.remove(); o.removeFromParent(); });
  pb.labels = [];
}

function addLabel(id, cls, color) {
  const P = geom.attributes.position.array, d = document.createElement("div");
  d.className = "splab " + cls; if (color) d.style.color = color;
  const tok = meta.tokens[id], g = UI.gloss[id] && UI.foreign(tok) ? ` (${UI.gloss[id].en})` : "";
  d.textContent = UI.show(tok) + g;
  const o = new CSS2DObject(d); o.position.set(P[3 * id], P[3 * id + 1], P[3 * id + 2]);
  pb.group.add(o); pb.labels.push(o); return o;
}

// Follow-along strip: input token (position colour) over its prediction. ``cur`` = position
// being played (-1: none), ``done`` = number of positions whose prediction has been revealed.
const strip = { key: null, spans: [], state: "" };
function renderStrip(cur, done) {
  const data = UI.data, el = $("pbstrip");
  if (!data || !tracing()) { el.style.display = "none"; placeLegends(); return; }
  const T = data.tokens.length, out = data.layers.length - 1, posF = (t) => (T > 1 ? t / (T - 1) : 0);
  if (strip.key !== data) {
    el.innerHTML = data.tokens.map((tok, t) => {
      const c = "#" + posColor(posF(t)).getHexString(), pred = data.vocab[data.cells[out][t].ids[0]];
      return `<span class="pbt${t >= data.n_prompt ? " gen" : ""}" data-t="${t}" title="position ${t}: reads ${UI.esc(JSON.stringify(tok))}, predicts ${UI.esc(JSON.stringify(pred))}">` +
        `<b style="color:${c}">${UI.esc(UI.show(tok))}</b><i style="background:${c}33; border-bottom:2px solid ${c}">${UI.esc(UI.show(pred))}</i></span>`;
    }).join("");
    strip.key = data; strip.spans = [...el.children]; strip.state = "";
  }
  const state = cur + "|" + done;
  if (state !== strip.state) {
    strip.state = state;
    strip.spans.forEach((sp, t) => {
      sp.classList.toggle("future", cur >= 0 && t > cur);
      sp.classList.toggle("nopred", t >= done);
      sp.classList.toggle("cur", t === cur);
    });
    if (cur >= 0) strip.spans[cur]?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  el.style.display = "flex";
  placeLegends();
}
// Keep the two legends above the strip.
function placeLegends() {
  const el = $("pbstrip"), h = el.style.display === "none" ? 0 : el.offsetHeight + 8;
  $("trlegend").style.bottom = $("splegend").style.bottom = 30 + h + "px";
}
$("pbstrip").addEventListener("click", (e) => {  // jump playback to a position
  const sp = e.target.closest(".pbt"); if (!sp || !loaded || !UI.data) return;
  enterPlayback(); pb.playing = false; pb.t = +sp.dataset.t; pbClock = 0; pb.stopAt = Infinity;
});

// Draw the state of step ``s`` at local time ``ms`` (plus the finished main line before it).
function drawPlayback(stepIdx, ms) {
  clearPlayback();
  const { steps, T, out } = pb.plan, P = geom.attributes.position.array;
  const at = (id) => new THREE.Vector3(P[3 * id], P[3 * id + 1], P[3 * id + 2]);
  const posF = (t) => (T > 1 ? t / (T - 1) : 0);
  const pts = { v: [], c: [], s: [], a: [] };
  const point = (id, col, size, alpha) => { const p = at(id); pts.v.push(p.x, p.y, p.z); pts.c.push(col.r, col.g, col.b); pts.s.push(size); pts.a.push(alpha); };
  const lines = { v: [], c: [], a: [] };
  const line = (a, b, col, alA, alB) => { lines.v.push(a.x, a.y, a.z, b.x, b.y, b.z); lines.c.push(col.r, col.g, col.b, col.r, col.g, col.b); lines.a.push(alA, alB); };

  // finished part of the main line (predictions 0..done-1 stay lit)
  const s = steps[stepIdx];
  const doneCount = s ? stepIdx + (ms >= PH.A + s.durB + PH.C ? 1 : 0) : steps.length;
  const mv = [], mc = [];
  for (let t = 0; t < doneCount; t++) {
    point(steps[t].pred, posColor(posF(t)), 12, 1);
    if (t > 0) { const a = at(steps[t - 1].pred), b = at(steps[t].pred), ca = posColor(posF(t - 1)), cb = posColor(posF(t));
      mv.push(a.x, a.y, a.z, b.x, b.y, b.z); mc.push(ca.r, ca.g, ca.b, cb.r, cb.g, cb.b); }
  }
  const lastLabel = new Map();
  for (let t = 0; t < doneCount; t++) lastLabel.set(steps[t].pred, t);
  if ($("trLab").checked) for (const [id, t] of lastLabel) addLabel(id, "main", "#" + posColor(posF(t)).getHexString());

  if (s) {
    const tB = ms - PH.A, tC = tB - s.durB, tD = tC - PH.C - PH.HOLD;
    const fade = tD > 0 ? Math.max(0, 1 - tD / PH.D) : 1;
    // (A) the input token being processed
    if (fade > 0) { point(s.input, new THREE.Color(1, 1, 1), 16, fade); addLabel(s.input, "cur"); }
    // (B) layer readouts pop in, in order; one node per token, latest layer sets its colour
    const nodes = new Map();
    const layerNow = tB < 0 ? -1 : Math.min(s.nLayers - 1, Math.floor(tB / PH.POP));
    for (const e of s.events) {
      const k = e.li - s.lo;
      if (k > layerNow) break;
      const n = nodes.get(e.id) || { id: e.id, pmax: 0, li: e.li, since: 0 };
      n.pmax = Math.max(n.pmax, e.p); n.li = e.li; n.since = tB - k * PH.POP;
      nodes.set(e.id, n);
    }
    const outP = at(s.pred), rayU = tC <= 0 ? 0 : Math.min(1, tC / PH.C);
    for (const n of nodes.values()) {
      const col = layerColor(n.li / out), al = Math.min(1, 0.15 + 0.85 * Math.sqrt(n.pmax)) * fade;
      const pulse = Math.max(0, 1 - n.since / 120);  // pops bigger for 120 ms after (re)activation
      point(n.id, col, 10 + 6 * Math.sqrt(n.pmax) + 6 * pulse, al);
      if (rayU > 0 && n.id !== s.pred) {  // (C) rays grow toward the output token
        const a = at(n.id), b = a.clone().lerp(outP, rayU);
        line(a, b, col, al * 0.8, al);
      }
    }
    if (rayU >= 1 && fade > 0) point(s.pred, posColor(posF(s.t)), 16 + 6 * fade, 1);  // output lights up
    const layerTxt = layerNow < 0 ? "input" : rayU > 0 ? "→ output" : "L" + UI.data.layers[s.lo + layerNow];
    $("pbStatus").textContent = `pos ${s.t + 1}/${T} · ${UI.show(UI.data.tokens[s.t])} · ${layerTxt}`;
  } else $("pbStatus").textContent = `done · ${T} predictions`;
  renderStrip(s ? s.t : -1, doneCount);

  if (mv.length) pb.group.add(mainLine(mv, mc));
  if (lines.v.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(lines.v, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(lines.c, 3));
    g.setAttribute("alpha", new THREE.Float32BufferAttribute(lines.a, 1));
    g.setAttribute("dist", new THREE.Float32BufferAttribute(segDist(lines.v), 1));
    pb.group.add(new THREE.LineSegments(g, lineMat));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts.v, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(pts.c, 3));
  g.setAttribute("size", new THREE.Float32BufferAttribute(pts.s, 1));
  g.setAttribute("alpha", new THREE.Float32BufferAttribute(pts.a, 1));
  pb.group.add(new THREE.Points(g, nodeMat));
}

let pbClock = 0, pbLast = 0;
function tickPlayback(now) {
  if (!pb.plan) return;
  if (pb.playing) {
    pbClock += (now - pbLast) * speed();
    let s = pb.plan.steps[pb.t];
    while (s && pbClock >= s.dur) {  // advance to the next token
      pbClock -= s.dur; pb.t++;
      if (pb.t >= pb.stopAt) { pb.playing = false; pbClock = 0; break; }
      s = pb.plan.steps[pb.t];
    }
    if (pb.t >= pb.plan.steps.length) pb.playing = false;
  }
  $("pbPlay").textContent = pb.playing ? "❚❚ pause" : "▶ play";
  pbLast = now;
  drawPlayback(pb.t, pbClock);
}

function enterPlayback() {
  if (!tracing()) { $("trOn").checked = true; buildTrace(); }
  if (!pb.plan) { pb.plan = planPlayback(); pb.t = 0; pbClock = 0; }
  tr.group.visible = false;
  tr.labels.forEach((o) => (o.element.style.display = "none"));
}
$("pbPlay").addEventListener("click", () => {
  if (!loaded || !UI.data) return;
  enterPlayback();
  if (pb.t >= pb.plan.steps.length) { pb.t = 0; pbClock = 0; }  // replay from the start
  pb.stopAt = Infinity; pb.playing = !pb.playing; pbLast = performance.now();
});
$("pbStep").addEventListener("click", () => {
  if (!loaded || !UI.data) return;
  enterPlayback();
  if (pb.t >= pb.plan.steps.length) { pb.t = 0; }
  pbClock = 0; pb.stopAt = pb.t + 1; pb.playing = true; pbLast = performance.now();
});
function resetPlayback() {
  pb.playing = false; pb.plan = null; pb.t = 0; pbClock = 0;
  clearPlayback(); $("pbStatus").textContent = ""; $("pbPlay").textContent = "▶ play";
  tr.group.visible = true; tr.labels.forEach((o) => (o.element.style.display = ""));
  if (UI.data && loaded) renderStrip(-1, UI.data.tokens.length);
}
$("pbReset").addEventListener("click", resetPlayback);
// Any change to the run or the trace controls invalidates the plan.
window.addEventListener("lens:data", resetPlayback);
["trK", "trP", "trLo", "trHi", "trOn"].forEach((id) => $(id).addEventListener(id === "trP" ? "input" : "change", resetPlayback));

// Handle for other views (step 3 will project runs into this space).
window.lensSpace = { showNeighbors: (i) => { showNeighbors(i); flyTo([i]); }, get meta() { return meta; }, get trace() { return tr.nodeInfo; }, get playback() { return { t: pb.t, playing: pb.playing, n: pb.plan?.steps.length }; } };
