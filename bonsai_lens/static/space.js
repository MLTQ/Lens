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
// z-index + isolation: CSS2DRenderer gives every label its own z-index; keep them all in one
// layer below the overlay panels (atom card, legends, controls).
Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "1", isolation: "isolate" });
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
  "non-verbal atom": "#d4ff00", "privileged dim": "#ff2020",
};

let meta = null, c3 = null, c3L = null, c2 = null, points = null, geom = null;
let joint = null;  // {b3, d3, b2, dist, info}: joint token + atom layouts, when available
const isTok = (i) => i < meta.NT;
const st = { view: "A3", dim: 3, color: "cat", hidden: new Set(), query: "", matches: null, focus: null };
let active = false, loaded = false;

// Depth fog: points fade and desaturate with distance from the camera. Transparent point clouds
// have no occlusion, so the eye can flip which side is near (a depth reversal: correct perspective
// then looks like the far side "expanding"); fog gives an unambiguous depth cue.
const vert = `
attribute vec3 color; attribute float size; attribute float alpha;
varying vec3 vColor; varying float vAlpha; uniform float scale; uniform float uNear; uniform float uFar; uniform float uFog;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float fog = uFog * smoothstep(uNear, uFar, -mv.z);
  float lum = dot(color, vec3(0.299, 0.587, 0.114));
  vColor = mix(color, vec3(lum), fog * 0.6) * (1.0 - 0.45 * fog);
  vAlpha = alpha * (1.0 - 0.6 * fog);
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
  uniforms: { scale: { value: 120 }, uNear: { value: 50 }, uFar: { value: 200 }, uFog: { value: 1 } },
});

let loading = null;
const ensureLoaded = () => (loading ??= load());  // one load, however many triggers

async function load() {
  $("spstatus").textContent = "Loading token space…";
  const r = await fetch("/space/meta.json");
  if (!r.ok) { $("spstatus").textContent = "No token space yet: run scripts/embed_vocab.py and copy lenses/space here."; return; }
  meta = await r.json();
  c3 = new Float32Array(await (await fetch("/space/coords3.f32")).arrayBuffer());
  // Best local-fidelity layout from scripts/layout_bakeoff.py (UMAP k=15, min_dist=0); optional
  const c3lr = await fetch("/space/coords3_local.f32");
  c3L = c3lr.ok ? new Float32Array(await c3lr.arrayBuffer()) : null;
  if (!c3L) $("spdim").querySelector('option[value="A3L"]').disabled = true;
  c2 = new Float32Array(await (await fetch("/space/coords2.f32")).arrayBuffer());
  meta.NT = meta.n;
  // log-norm quantiles for the "readout norm" coloring (tokens only)
  const ln = meta.norm.map((x) => Math.log(x + 1e-6)).sort((a, b) => a - b);
  meta.lnLo = ln[Math.floor(ln.length * 0.01)]; meta.lnHi = ln[Math.floor(ln.length * 0.99)];
  const jr = await fetch("/joint/meta_joint.json");
  if (jr.ok) {
    const info = await jr.json();
    const f32 = async (name) => new Float32Array(await (await fetch(`/joint/${name}`)).arrayBuffer());
    joint = { info, b3: await f32("coords_b3.f32"), d3: await f32("coords_d3.f32"), b2: await f32("coords_b2.f32"), dist: await f32("dist_word.f32") };
    const ca = meta.categories.push("non-verbal atom") - 1, cp = meta.categories.push("privileged dim") - 1;
    meta.category_counts["non-verbal atom"] = info.n_atoms; meta.category_counts["privileged dim"] = info.n_priv;
    for (const a of info.atoms) { meta.tokens.push(a.id); meta.cat.push(ca); meta.norm.push(1); }
    for (const p of info.privileged) { meta.tokens.push(p.id); meta.cat.push(cp); meta.norm.push(1); }
    meta.n = meta.NT + info.n_atoms + info.n_priv;
    ["B", "D", "C"].forEach((v) => ($("spdim").querySelector(`option[value="${v}"]`).disabled = false));
    const tryJson = async (path) => { const r = await fetch(path); return r.ok ? r.json() : null; };
    const tryBin = async (path, T) => { const r = await fetch(path); return r.ok ? new T(await r.arrayBuffer()) : null; };
    joint.quality = await tryJson("/joint/quality/summary.json");
    joint.sweep = await tryJson("/joint/quality/sweep_results.json");
    joint.interp = await tryJson("/joint/autointerp.json");
    joint.control = await tryJson("/joint/autointerp_control.json");
    joint.iso = await tryBin("/joint/isolation.f32", Float32Array);
    joint.isoStats = await tryJson("/joint/isolation.json");
    joint.rescore = await tryJson("/joint/autointerp_rescore.json");
    joint.fid = {};
    for (const v of ["A3", "A3L", "A2", "B", "D", "C"]) joint.fid[v] = await tryBin(`/joint/quality/fid_${v}.u8`, Uint8Array);
    const rj = await tryJson("/joint/quality/regions.json");
    const rof = await tryBin("/joint/quality/region_of.u16", Uint16Array);
    if (rj && rof) {
      const byId = new Map(rj.regions.map((r) => [r.id, { ...r, members: [] }]));
      for (let i = 0; i < rof.length; i++) byId.get(rof[i])?.members.push(i);
      joint.regions = [...byId.values()];
      for (const r of joint.regions) r.name = joint.interp?.regions?.[String(r.id)] || null;
    }
    renderStats();
    renderCandidates();
  }
  const n = meta.n;
  geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geom.setAttribute("size", new THREE.BufferAttribute(new Float32Array(n), 1));
  geom.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(n), 1));
  points = new THREE.Points(geom, material);
  scene.add(points, labelGroup);
  buildLegend();
  loaded = true;
  setPositions(); setStyle(); resetCamera(); buildTrace(); buildRegions();
  $("spstatus").textContent = `${meta.NT.toLocaleString()} tokens${joint ? ` + ${joint.info.n_atoms} non-verbal atoms` : ""} · ${meta.space}`;
}

function setPositions() {
  const pos = geom.attributes.position.array, n = meta.n, v = st.view;
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0, z = 0;
    if (v === "A3" && isTok(i)) { x = c3[3 * i]; y = c3[3 * i + 1]; z = c3[3 * i + 2]; }
    else if (v === "A3L" && isTok(i)) { x = c3L[3 * i]; y = c3L[3 * i + 1]; z = c3L[3 * i + 2]; }
    else if (v === "A2" && isTok(i)) { x = c2[2 * i]; y = c2[2 * i + 1]; }
    else if (v === "B" || v === "D") { const c = v === "B" ? joint.b3 : joint.d3; x = c[3 * i]; y = c[3 * i + 1]; z = c[3 * i + 2]; }
    else if (v === "C") { x = joint.b2[2 * i]; z = -joint.b2[2 * i + 1]; y = joint.dist[i] * 1.1; }  // floor = UMAP 2D, up = distance to nearest word
    pos[3 * i] = x * R; pos[3 * i + 1] = y * R; pos[3 * i + 2] = z * R;
  }
  geom.attributes.position.needsUpdate = true;
  geom.computeBoundingSphere();
}

const tmp = new THREE.Color();
const GREY = new THREE.Color(0x4a4a48);
// Layout fidelity: share of a point's 15 true neighbours found among its 150 nearest on screen.
// Scale saturates at 60% (few points exceed it); red = the picture misplaces this point's neighbourhood.
function fidColor(i) {
  const f = joint?.fid?.[st.view];
  if (!f) return GREY;
  const v = Math.min(1, f[i] / 15 / 0.6);
  return tmp.setHSL(0.0 + 0.33 * v, 0.85, 0.35 + 0.2 * v);
}
// Isolation: 1 - cosine to the nearest other point, measured in the full space (not the layout).
// Scale 0.3 (crowded) .. 0.9 (alone); words median ~0.58, atoms ~0.79.
function isoColor(i) {
  const v = joint?.iso;
  if (!v) return GREY;
  const t = Math.min(1, Math.max(0, (v[i] - 0.3) / 0.6));
  return tmp.setHSL(0.62 - 0.62 * t, 0.8, 0.3 + 0.35 * t);
}
// Atom describability: AUROC of Bonsai's own description on held-out contexts (0.5 = chance).
function descColor(i) {
  if (isTok(i)) return GREY;
  const r = describe(i);
  if (!r) return tmp.setHSL(0, 0, 0.55);
  const v = Math.min(1, Math.max(0, (r.auroc - 0.5) / 0.5));
  return tmp.setHSL(0.0 + 0.33 * v, 0.9, 0.5);
}
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
    if (st.color === "norm") c = isTok(i) ? ramp(Math.min(1, Math.max(0, (Math.log(meta.norm[i] + 1e-6) - meta.lnLo) / (meta.lnHi - meta.lnLo)))) : catCol[meta.cat[i]];
    else if (st.color === "fid") c = fidColor(i);
    else if (st.color === "desc") c = descColor(i);
    else if (st.color === "iso") c = isoColor(i);
    else c = catCol[meta.cat[i]];
    color.array[3 * i] = c.r; color.array[3 * i + 1] = c.g; color.array[3 * i + 2] = c.b;
    const hidden = st.hidden.has(cat) || (!isTok(i) && st.view.startsWith("A"));
    if (!isTok(i) && !m) {  // atoms and privileged dims: always legible in the joint views
      const pr = cat === "privileged dim";
      size.array[i] = pr ? 16 : 2.8; alpha.array[i] = hidden ? 0 : pr ? 1 : tracing() ? 0.25 : 0.95;
      continue;
    }
    if (m) {
      const hit = m.has(i), foc = i === st.focus;
      size.array[i] = foc ? 18 : hit ? 11 : 1.2; alpha.array[i] = hidden && !hit ? 0 : hit || foc ? 1 : 0.06;
      if (foc) { color.array[3 * i] = 1; color.array[3 * i + 1] = 1; color.array[3 * i + 2] = 1; }
      else if (hit) { color.array[3 * i] = 1; color.array[3 * i + 1] = 0.55; color.array[3 * i + 2] = 0.15; }
    } else { size.array[i] = 1.6; alpha.array[i] = hidden ? 0 : tracing() ? 0.05 : st.view === "C" ? 0.35 : 0.75; }
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
    const g = isTok(i) && UI.gloss[i] && UI.foreign(meta.tokens[i]) ? ` (${UI.gloss[i].en})` : "";
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
  if (ql === "candidates" && joint?.rescore) {  // the robust undescribable atoms
    for (let i = meta.NT; i < meta.n; i++) if (isCandidate(i)) exact.add(i);
    st.matches = exact; setStyle();
    $("spcount").textContent = `${exact.size} xeno-candidates (undescribable by Bonsai on re-test)`;
    if (exact.size) flyTo([...exact]);
    return;
  }
  for (let i = 0; i < meta.n; i++) {
    const s = meta.tokens[i].trim().toLowerCase();
    if (!s || (!isTok(i) && st.view.startsWith("A"))) continue;
    if (!isTok(i) && part.size < 3000 && describe(i)?.desc.toLowerCase().includes(ql)) { part.add(i); continue; }
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
  if (st.view === "C") {  // frame the floor (words) and the atom sheet above it from their actual extent
    geom.computeBoundingBox();
    const bb = geom.boundingBox, c = bb.getCenter(new THREE.Vector3()), size = bb.getSize(new THREE.Vector3()).length();
    controls.target.copy(c);
    camera.position.set(c.x + size * 0.35, c.y + size * 0.45, c.z + size * 0.95);
  }
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
  if (lastHover >= 0) (isTok(lastHover) ? showNeighbors(lastHover) : showAtom(lastHover));
  else clearSelection();  // click on empty space
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
  if (!isTok(i)) {
    tip.innerHTML = atomSummary(i);
    tip.style.display = "block"; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px";
    return;
  }
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

// ---- non-verbal atoms ------------------------------------------------------
const SPARK = "▁▂▃▄▅▆▇█";
function atomInfo(i) {
  const j = i - meta.NT, na = joint.info.n_atoms;
  return j < na ? { kind: "atom", ...joint.info.atoms[j] } : { kind: "priv", ...joint.info.privileged[j - na] };
}
function spark(profile) {
  const mx = Math.max(...profile, 1e-9);
  return profile.map((v) => SPARK[Math.min(7, Math.floor((v / mx) * 7.99))]).join("");
}
const tokStr = (id) => UI.show(meta.tokens[id]);
function nearestHtml(a, k) {
  return a.nearest.slice(0, k).map((n) => `${UI.esc(tokStr(n.id))} <span class="src">${n.cos.toFixed(2)}</span>`).join(" · ");
}
function ctxHtml(c) {  // "left[[tok]]right" -> highlighted
  const m = c.match(/^(.*)\[\[(.*)\]\](.*)$/s);
  return m ? `${UI.esc(m[1])}<mark>${UI.esc(m[2])}</mark>${UI.esc(m[3])}` : UI.esc(c);
}
function exHtml(x) {
  return `<div class="ex">L${x.layer} · ${x.act.toFixed(2)} · ${UI.esc(x.left.replace(/\n/g, "↵"))}<mark>${UI.esc(x.tok.replace(/\n/g, "↵"))}</mark>${UI.esc(x.right.replace(/\n/g, "↵"))}</div>`;
}
function describe(i) {
  const r = joint?.interp?.atoms?.[String(i - meta.NT)];
  if (!(r && r.desc && r.auroc != null)) return null;
  const rs = joint?.rescore?.[String(i - meta.NT)];
  return rs?.auroc15 != null ? { ...r, auroc: rs.auroc15, rescored: rs } : r;
}
// Robust "Bonsai cannot describe it" atoms: at chance on 15+15 fresh contexts and no better than
// a random other atom's description. Candidates for the xeno hunt, pending a stronger describer.
function isCandidate(i) {
  const rs = joint?.rescore?.[String(i - meta.NT)];
  return !!rs && rs.auroc15 != null && rs.auroc15 < 0.62 && rs.auroc15 - rs.shuffled15 < 0.1;
}
function atomSummary(i) {
  const a = atomInfo(i);
  if (a.kind === "priv")
    return `<span class="tok">${a.id}</span> privileged residual coordinate<div class="src">best cos to any word ${a.best_cos.toFixed(3)} · nearest: ${nearestHtml(a, 4)}</div>`;
  const d = describe(i);
  return `<span class="tok">${a.id}</span> non-verbal atom · fires on ${(a.freq * 100).toFixed(2)}% of activations` +
    (d ? `<div><b>“${UI.esc(d.desc)}”</b> <span class="src">describability ${d.auroc.toFixed(2)}</span></div>` : "") +
    `<div class="src">layers ${joint.info.layers[0]}→${joint.info.layers.at(-1)} <span class="spark">${spark(a.layer_profile)}</span></div>` +
    `<div class="src">best cos to any word ${a.best_cos.toFixed(3)} · ${nearestHtml(a, 3)}</div>` +
    (a.exemplars[0] ? exHtml(a.exemplars[0]) : "") + `<div class="src">click for the full card</div>`;
}
// Drop neighbour / search highlights, their labels, and the atom card.
function clearSelection() {
  if (st.focus === null && !st.matches) return;
  st.focus = null; st.matches = null;
  $("atomcard").style.display = "none";
  $("spcount").textContent = "";
  setStyle();
}

function showAtom(i) {
  const a = atomInfo(i);
  st.focus = i;
  st.matches = new Set(a.nearest.map((n) => n.id));
  setStyle();
  const card = $("atomcard");
  card.innerHTML = `<button id="acClose">close</button><h3>${a.id}</h3>` + (a.kind === "priv"
    ? `<div>Privileged residual coordinate (a single axis the model treats specially).</div>`
    : `<div>Non-verbal atom · fires on ${(a.freq * 100).toFixed(2)}% of held-out activations</div>
       <div class="src">where it fires, layers ${joint.info.layers.join(" ")}:</div><div class="spark">${spark(a.layer_profile)}</div>`) +
    (() => {
      const d = describe(i), r = joint?.interp?.atoms?.[String(i - meta.NT)];
      if (d) return `<div style="margin-top:6px">Bonsai's description: <b>“${UI.esc(d.desc)}”</b></div>
        <div class="src">describability ${d.auroc.toFixed(2)} ${d.rescored
          ? `(re-scored on 15 fresh firing vs 15 silent contexts; a random other atom's description scores ${d.rescored.shuffled15.toFixed(2)} on the same contexts; first pass on 5 + 5: ${d.rescored.first_auroc.toFixed(2)})`
          : "(AUROC on 5 held-out firing vs 5 silent contexts; 0.5 = chance, 1 = perfect; single scores are noisy)"}</div>
        ${isCandidate(i) ? `<div style="color:#ff7a45;margin-top:3px">xeno-candidate: Bonsai's own description predicts this atom no better than chance. Read the contexts — is there a pattern a human can name?</div>` : ""}
        <details><summary class="src">held-out test contexts</summary>${d.held_out.map((c) => `<div class="ex">✓ ${ctxHtml(c)}</div>`).join("")}${d.negatives.map((c) => `<div class="ex">✗ ${ctxHtml(c)}</div>`).join("")}</details>`;
      return r?.note ? `<div class="src" style="margin-top:6px">${UI.esc(r.note)}</div>` : "";
    })() +
    `<div style="margin-top:6px">best cosine to any of ${meta.NT.toLocaleString()} words: <b>${a.best_cos.toFixed(3)}</b>` +
    ` <span class="src">(words to their nearest word: median ${joint.info.stats.best_cos_to_any_token["tokens (nearest other token)"].median.toFixed(2)})</span></div>` +
    `<div class="src" style="margin:4px 0">least-unlike words (highlighted in the map): ${nearestHtml(a, 10)}</div>` +
    (a.exemplars?.length ? `<div style="margin-top:6px">strongest contexts (activating token highlighted):</div>${a.exemplars.map(exHtml).join("")}` : "");
  card.style.display = "block";
  $("acClose").onclick = () => { card.style.display = "none"; st.focus = null; st.matches = null; setStyle(); };
  $("spcount").textContent = `${a.id}: its 10 least-unlike words are highlighted`;
}
// The popover is position:fixed (the view clips overflow); place it under the badge, clamped on screen.
function placeInfo() {
  const b = $("spinfo").getBoundingClientRect(), pop = $("spstats"), w = 330;
  pop.style.left = Math.max(8, Math.min(b.left, innerWidth - w - 16)) + "px";
  pop.style.top = b.bottom + 6 + "px";
}
$("spinfo").addEventListener("mouseenter", placeInfo);
$("spinfo").addEventListener("focus", placeInfo);

// ---- list of atoms Bonsai cannot describe (side panel) -------------------------------
function renderCandidates() {
  if (!joint?.rescore || !joint?.interp) return;
  const row = (a, score, sub, ctx) => {
    const d = joint.interp.atoms[String(a)], info = joint.info.atoms[a];
    return `<div class="cand" data-atom="${a}"><span class="id">ξ${a}</span><span class="sc">${sub}</span>
      <div class="d">“${UI.esc(d?.desc ?? "—")}”</div>
      <div class="ctx">fires on ${(info.freq * 100).toFixed(2)}%${ctx ? ` · ${ctxHtml(ctx.slice(-120))}` : ""}</div></div>`;
  };
  const robust = Object.entries(joint.rescore)
    .filter(([a]) => isCandidate(meta.NT + +a))
    .sort((x, y) => x[1].auroc15 - y[1].auroc15);
  $("candlist").innerHTML = robust.map(([a, v]) =>
    row(+a, v.auroc15, `${v.auroc15.toFixed(2)} vs random ${v.shuffled15.toFixed(2)}`, v.pos?.[0])).join("");
  $("candN").textContent = `(${robust.length})`;
  const retested = new Set(Object.keys(joint.rescore));
  const noisy = Object.entries(joint.interp.atoms)
    .filter(([a, v]) => v.auroc != null && v.auroc <= 0.5 && !retested.has(a))
    .sort((x, y) => x[1].auroc - y[1].auroc);
  $("candmorelist").innerHTML = noisy.map(([a, v]) => row(+a, v.auroc, `first pass ${v.auroc.toFixed(2)}`, v.held_out?.[0])).join("");
  $("candMoreN").textContent = noisy.length;
  $("cands").style.display = "";
}
$("cands").addEventListener("click", (e) => {
  const el = e.target.closest(".cand"); if (!el || !loaded) return;
  const a = +el.dataset.atom;
  if ($("spacewrap").style.display !== "block") $("vSpace").click();   // show the Space view
  if (st.view.startsWith("A")) { $("spdim").value = "B"; $("spdim").dispatchEvent(new Event("change")); }  // atoms live in B / densMAP / C
  showAtom(meta.NT + a);
  flyTo([meta.NT + a]);
});

function renderStats() {
  const b = joint.info.stats.best_cos_to_any_token, p = joint.info.stats.purity;
  const raw = p["atoms: share of neighbours that are atoms, raw full space"];
  const q = joint.quality?.fidelity?.[st.view];
  const pct = (x) => (x * 100).toFixed(0) + "%";
  const acc = q ? `<h4>How accurate is this view?</h4>
    Of each word's 15 true nearest neighbours, <b>${pct(q.strict_tokens)}</b> are among its 15 nearest on screen and
    <b>${pct(q.loose_tokens)}</b> among its 150 nearest${q.strict_atoms != null ? ` (atoms: ${pct(q.strict_atoms)} / ${pct(q.loose_atoms)})` : ""}.
    A random layout scores ~0.01%. So regions are real, fine neighbourhoods are not: click a point for its true neighbours.
    Colour → <i>layout fidelity</i> shows where this picture is trustworthy.` : "";
  const bo = joint.quality?.bakeoff ? `<h4>Could a different projection do better?</h4>Measured on the word map (exact / neighbourhood / global rank-correlation of distances):
    <table>${Object.entries(joint.quality.bakeoff).filter(([, r]) => r.strict != null).map(([n, r]) =>
      `<tr><td>${UI.esc(n)}</td><td>${pct(r.strict)}</td><td>${pct(r.loose)}</td><td>${r.global_spearman.toFixed(2)}</td></tr>`).join("")}</table>
    Local fidelity improves by ~60% at best (the "local-faithful" view) at a small global cost; no 3D layout comes close to faithful.
    The vocabulary is too high-dimensional for any picture: trust regions, not exact neighbours.` : "";
  const sw = joint.sweep ? (() => {
    const rows = joint.sweep.sae.map((r) => `<tr><td>${r.n.toLocaleString()} atoms, ${r.k} active</td><td><b>${pct(1 - r.eval_fvu)}</b></td><td>${pct(1 - r.train_fvu)}</td><td>${pct(r.dead_on_eval)}</td></tr>`).join("");
    const pc = Object.entries(joint.sweep.pca_eval_fvu).map(([m, f]) => `${m}: ${pct(1 - f)}`).join(" · ");
    return `<h4>How many non-verbal atoms are there?</h4>
      Share of the non-verbal remainder explained on held-out documents (training docs; atoms never firing on held-out):
      <table>${rows}</table>
      Linear reference, plain principal directions: ${pc}.`;
  })() : "";
  const di = joint.interp?.atoms ? (() => {
    const sc = Object.values(joint.interp.atoms).filter((r) => r.auroc != null).map((r) => r.auroc).sort((a, b) => a - b);
    if (!sc.length) return "";
    const med = sc[sc.length >> 1], low = sc.filter((x) => x < 0.7).length;
    const c = joint.control, rs = joint.rescore ? Object.values(joint.rescore).filter((v) => v.auroc15 != null) : [];
    const medOf = (xs) => { const a = xs.slice().sort((x, y) => x - y); return a[a.length >> 1]; };
    const g = (grp, k) => medOf(rs.filter((v) => v.group === grp).map((v) => v[k]));
    const nCand = joint.rescore ? Object.keys(joint.rescore).filter((k) => isCandidate(meta.NT + +k)).length : 0;
    return `<h4>Can Bonsai describe its own atoms?</h4>${sc.length} atoms described from 12 contexts and tested on 5 + 5 held-out ones:
      median describability <b>${med.toFixed(2)}</b> (0.5 = chance) · ${low} below 0.7.
      ${c ? `<br>Control: re-scored with a <i>different</i> atom's description, the median drops to <b>${c.shuffled_median.toFixed(2)}</b>, so the score measures real fit — but ${(c["shuffled_ge_0.8"] * 100).toFixed(0)}% of mismatched descriptions still reach 0.8, so single first-pass scores are noisy.` : ""}
      ${rs.length ? `<br>Re-test on 15 + 15 fresh contexts: the 50 lowest scorers <b>${g("lowest", "auroc15").toFixed(2)}</b> (shuffled ${g("lowest", "shuffled15").toFixed(2)}), 50 random atoms <b>${g("random", "auroc15").toFixed(2)}</b> (shuffled ${g("random", "shuffled15").toFixed(2)}).
        <b>${nCand}</b> atoms stay at chance: search <i>candidates</i>. They are what Bonsai cannot describe — not yet what no human can.` : ""}
      Colour → <i>atom describability</i>.`;
  })() : "";
  const is = joint.isoStats ? `<h4>Are the atoms clustered or diffuse?</h4>Diffuse. Distance to the nearest other point (1 − cosine, full space):
    words median <b>${joint.isoStats.words_median.toFixed(2)}</b>, atoms <b>${joint.isoStats.atoms_median.toFixed(2)}</b>; only ${(joint.isoStats.words_above_atom_median * 100).toFixed(1)}% of words are as isolated as the typical atom.
    Layouts draw them as one lobe because each atom's nearest neighbours are other atoms — neighbour-based projections show <i>who is nearest</i>, not <i>how far</i>. Colour → <i>isolation</i> shows it.` : "";
  $("spstats").innerHTML = acc + is + bo + sw + di + `<h4>Separation (measured, not from the picture)</h4>`;
  $("spstats").innerHTML += `
    best cosine to any word, full 5120-d (median): words→nearest other word <b>${b["tokens (nearest other token)"].median.toFixed(3)}</b> ·
    atoms <b>${b.atoms.median.toFixed(3)}</b> · random directions ${b["random directions"].median.toFixed(3)}<br>
    atoms' 29 nearest neighbours that are atoms: <b>${(p["atoms: share of 29 nearest neighbours that are atoms"] * 100).toFixed(0)}%</b> in the layout's graph
    ${raw !== undefined ? `(${(raw * 100).toFixed(0)}% in the raw, hub-distorted space)` : ""} · chance ${(p["chance (atoms / all points)"] * 100).toFixed(1)}% ·
    atoms with no word among them: ${(p["atoms with no token among nearest 29"] * 100).toFixed(0)}%<br>
    ${joint.info.n_atoms} atoms from a TopK SAE on the non-verbal remainder (FVU ${joint.info.fvu.toFixed(2)})`;
}

// ---- region labels -----------------------------------------------------------
// A label sits at the median position of its region's tokens in the current view and is shown
// only if the region is compact there (median member distance small), fading as it scatters:
// the map never claims a coherent region that the layout has spread across the picture.
const regionGroup = new THREE.Group();
scene.add(regionGroup);
function buildRegions() {
  regionGroup.children.slice().forEach((o) => { o.element?.remove(); regionGroup.remove(o); });
  if (!joint?.regions || !$("spRegions").checked) return;
  const P = geom.attributes.position.array, SAMPLE = 300;
  const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };
  const placed = [];
  for (const r of joint.regions) {
    const m = r.members, step = Math.max(1, Math.floor(m.length / SAMPLE)), xs = [], ys = [], zs = [];
    for (let k = 0; k < m.length; k += step) { const i = m[k]; xs.push(P[3 * i]); ys.push(P[3 * i + 1]); zs.push(P[3 * i + 2]); }
    const c = [med(xs), med(ys), med(zs)];
    const d = xs.map((x, k) => Math.hypot(x - c[0], ys[k] - c[1], zs[k] - c[2]));
    const spread = med(d) / R;  // median member distance from the label, relative to the map radius
    placed.push({ r, c, spread });
  }
  placed.sort((a, b) => a.spread - b.spread);
  let shown = 0;
  for (const { r, c, spread } of placed) {
    if (spread > 0.12 || r.size < 300) continue;
    if (++shown > 40) break;  // the 40 most compact regions; more is clutter
    const el = document.createElement("div");
    el.className = "reglab";
    el.textContent = r.name || r.top.slice(0, 3).map((i) => UI.show(meta.tokens[i]).trim()).join(" · ");
    el.style.opacity = ((1 - spread / 0.12) * 0.85 + 0.15) * (tracing() ? 0.35 : 1);  // recede behind an active trace
    el.style.fontSize = 10 + Math.min(4, Math.log10(r.size) * 1.2) + "px";
    el.title = `${r.size} tokens · coherence ${r.coherence.toFixed(2)} · on-screen spread ${spread.toFixed(3)}`;
    const o = new CSS2DObject(el);
    o.position.set(...c);
    regionGroup.add(o);
  }
}
$("spRegions").addEventListener("change", () => loaded && buildRegions());

// ---- wiring --------------------------------------------------------------
$("spdim").addEventListener("change", (e) => {
  st.view = e.target.value; st.dim = st.view === "A2" ? 2 : 3;
  $("spstatus").textContent = st.view.startsWith("A")
    ? `${meta.NT.toLocaleString()} tokens · ${meta.space}`
    : `${meta.NT.toLocaleString()} tokens + ${joint.info.n_atoms} atoms · ${joint.info.space}`;
  $("spinfo").style.display = joint ? "" : "none";
  if (st.view.startsWith("A")) $("atomcard").style.display = "none";
  if (loaded) { setPositions(); setStyle(); resetCamera(); buildTrace(); buildRegions(); renderStats(); }
});
$("spcolor").addEventListener("change", (e) => { st.color = e.target.value; if (loaded) setStyle(); });
$("spsearch").addEventListener("keydown", (e) => { if (e.key === "Enter" && loaded) search(e.target.value); });
$("spclear").addEventListener("click", () => { $("spsearch").value = ""; if (loaded) { search(""); resetCamera(); } });

function setActive(on) {
  active = on;
  renderer.setAnimationLoop(on ? (now) => {
    lineMat.uniforms.time.value = now / 1000 * 0.7;   // waves travel toward each segment's end
    { // fog range: from just in front of the cloud's near side to its far side
      const d = camera.position.distanceTo(controls.target), r = geom?.boundingSphere?.radius ?? R;
      material.uniforms.uNear.value = Math.max(0.1, d - r * 0.6); material.uniforms.uFar.value = d + r * 0.9;
    }
    mainMat.dashOffset = -now / 1000 * 1.6;          // gaps march in sequence order
    hover(); tickPlayback(now); controls.update(); renderer.render(scene, camera); labels.render(scene, camera); } : null);
  if (on) { resize(); ensureLoaded(); }
}
window.addEventListener("lens:view", (e) => setActive(e.detail === "space"));
// Load the token space in the background too, so the side-panel atom list is available from any tab.
setTimeout(ensureLoaded, 1500);
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
nodeMat.uniforms = { scale: material.uniforms.scale, uNear: material.uniforms.uNear, uFar: material.uniforms.uFar, uFog: { value: 0.35 } };

const posColor = (f) => new THREE.Color().setHSL(((200 + 215 * f) % 360) / 360, 0.9, 0.62);   // blue→violet→magenta→orange→yellow
const layerColor = (f) => new THREE.Color().setHSL((168 - 22 * f) / 360, 0.75 - 0.35 * f, 0.24 + 0.66 * f); // dark teal→mint→white
const tracing = () => $("trOn").checked && !!UI.data && loaded;
const pThr = () => { const v = +$("trP").value; return v === 0 ? 0 : Math.pow(10, -3 + (v / 100) * 3 * 0.9); };  // 1e-3 .. ~0.5

// ---- non-verbal atoms in the trace --------------------------------------------
const LIME = new THREE.Color("#d4ff00");
const ALWAYS_ON = 0.1;  // atoms firing on >10% of activations are background machinery; hidden in traces
const atomsOn = () => $("trAtoms").checked && !!joint && !st.view.startsWith("A");
const atomThr = () => Math.max(0.001, (+$("trAthr").value || 1) / 100);
const nv = { key: null, data: null, pending: null };
function nonverbalFor(data) {  // cached per run; triggers a rebuild when it arrives
  if (nv.key === data) return nv.data;
  if (nv.pending !== data) {
    nv.pending = data;
    $("trlegend").dataset.note = "decomposing activations into verbal / non-verbal…";
    fetch("/api/nonverbal").then((r) => r.json()).then((d) => {
      if (nv.pending !== data) return;
      nv.key = data; nv.data = d.cells ? d : null; nv.pending = null;
      buildTrace(); if (pb.plan) { pb.plan = planPlayback(); }
    });
  }
  return null;
}
function atomEvents(data, t, li) {  // strongest non-background atoms of cell (li, t) above threshold
  const layer = data.layers[li], cells = nv.data?.cells?.[String(layer)];
  if (!cells) return [];
  const out = [];
  for (const [a, act, share] of cells[t].atoms) {
    if (share < atomThr() || (joint.info.atoms[a]?.freq ?? 0) > ALWAYS_ON) continue;
    out.push({ li, id: meta.NT + a, p: share, atom: a });
  }
  return out;
}

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
  // non-verbal atoms, aggregated per (position, atom) like the word branches
  const aagg = new Map();
  let nvNote = "";
  if (atomsOn()) {
    if (nonverbalFor(data)) {
      for (let t = 0; t < T; t++) for (let li = lo; li <= hi; li++) for (const e of atomEvents(data, t, li)) {
        const key = t * 1e6 + e.atom;
        const g = aagg.get(key) || { t, id: e.id, atom: e.atom, pmax: 0, w: 0, lw: 0, layers: [] };
        g.pmax = Math.max(g.pmax, e.p); g.w += e.p; g.lw += e.p * li; g.layers.push("L" + data.layers[li]);
        aagg.set(key, g);
      }
      nvNote = ` · ${aagg.size} atom branches (always-on atoms hidden)`;
    } else nvNote = " · decomposing activations…";
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
  for (const g of aagg.values()) {  // lime branches: non-verbal atom -> this position's prediction
    const a = at(pred[g.t]), b = at(g.id), vis = Math.min(1, Math.sqrt(g.pmax / 0.05));
    const al = 0.1 + 0.8 * vis;
    bv.push(b.x, b.y, b.z, a.x, a.y, a.z); bc.push(LIME.r, LIME.g, LIME.b, LIME.r, LIME.g, LIME.b); ba.push(al, al * 0.3);
    const d = joint.interp?.atoms?.[String(g.atom)];
    addNode(g.id, { id: g.id, main: false, strength: g.pmax, color: LIME, alpha: al, size: 3 + 7 * vis,
      note: `non-verbal ξ${g.atom} at pos ${g.t} (${UI.show(data.tokens[g.t])}) · peak ${(g.pmax * 100).toFixed(1)}% of the activation · ${g.layers.join(" ")}` +
        (d?.desc ? ` · “${d.desc}” (describability ${d.auroc.toFixed(2)})` : "") });
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
    <div style="margin-top:4px">${pred.length} predictions · ${agg.size} word branches${nvNote}</div>` +
    (atomsOn() ? `<div style="margin-top:4px"><span style="color:#d4ff00">■</span> lime = non-verbal atom · alpha = its share of the activation</div>` : "");
  $("trlegend").style.display = "block";
  setStyle();
  if (tr.frameNext) { tr.frameNext = false; frameTrace(list.map((n) => n.id)); }
  buildRegions();
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
["trOn", "trK", "trP", "trLo", "trHi", "trLab", "trAtoms", "trAthr"].forEach((id) => $(id).addEventListener(id === "trP" ? "input" : "change", () => loaded && buildTrace()));
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
    const events = [];  // [{li, id, p, atom?}] in layer order
    const withAtoms = atomsOn() && nv.key === data;
    for (let li = lo; li <= hi; li++) {
      const c = data.cells[li][t];
      for (let k = 0; k < K; k++) if (c.p[k] >= thr) events.push({ li, id: c.ids[k], p: c.p[k] });
      if (withAtoms) for (const e of atomEvents(data, t, li)) events.push({ ...e, p: Math.min(1, e.p * 10) });  // shares are small; ×10 for visual parity
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
      const n = nodes.get(e.id) || { id: e.id, pmax: 0, li: e.li, since: 0, atom: e.atom != null };
      n.pmax = Math.max(n.pmax, e.p); n.li = e.li; n.since = tB - k * PH.POP;
      nodes.set(e.id, n);
    }
    const outP = at(s.pred), rayU = tC <= 0 ? 0 : Math.min(1, tC / PH.C);
    for (const n of nodes.values()) {
      const col = n.atom ? LIME : layerColor(n.li / out), al = Math.min(1, 0.15 + 0.85 * Math.sqrt(n.pmax)) * fade;
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
["trK", "trP", "trLo", "trHi", "trOn", "trAtoms", "trAthr"].forEach((id) => $(id).addEventListener(id === "trP" ? "input" : "change", resetPlayback));

// Handle for other views (step 3 will project runs into this space).
window.lensSpace = { showNeighbors: (i) => { (isTok(i) ? showNeighbors(i) : showAtom(i)); flyTo([i]); }, get meta() { return meta; }, get trace() { return tr.nodeInfo; }, get playback() { return { t: pb.t, playing: pb.playing, n: pb.plan?.steps.length }; } };
