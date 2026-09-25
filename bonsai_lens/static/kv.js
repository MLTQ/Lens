// KV space. Default: every head of the layer as a hyperbolic disk (see drawDisks). One head:
// its keys seen from the query of the selected cell.
//   x = match with the query (q . k / |q|; for softmax heads x * |q| * scale is exactly the logit)
//   y, z = the keys' two main directions of variation orthogonal to the query
//   size = the weight the head actually used (softmax, or DeltaNet's effective weight after decay
//   and overwriting), colour = position in the prompt (same gradient as the Space playback).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const UI = window.lensUI;
const $ = (id) => document.getElementById(id);
const R = 30;

const host = $("kvwrap");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
host.prepend(renderer.domElement);
const labels = new CSS2DRenderer();
Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "1", isolation: "isolate" });
host.prepend(labels.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 1.3));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(0.5, 1, 0.8);
scene.add(sun);
const group = new THREE.Group();
scene.add(group);

const posColor = (f) => new THREE.Color().setHSL(((200 + 215 * f) % 360) / 360, 0.9, 0.62);
let active = false, head = null, cellKey = null, req = 0;

function clear() {
  group.children.slice().forEach((o) => { o.element?.remove(); group.remove(o); o.geometry?.dispose(); });
  track = null;
}
const label = (text, cls, pos) => {
  const d = document.createElement("div");
  d.className = "vlab " + cls; d.innerHTML = text;
  const o = new CSS2DObject(d); o.position.copy(pos); group.add(o); return o;
};

// Centred status in the canvas area (waiting, errors, "select a cell").
const status = document.createElement("div");
status.className = "kvstatus";
host.appendChild(status);
let statusTimer = null;
function setStatus(html, waiting = false) {
  clearInterval(statusTimer);
  status.style.display = html ? "flex" : "none";
  if (!html) return;
  if (!waiting) { status.innerHTML = html; return; }
  const t0 = performance.now();
  const tick = () => { status.innerHTML = `${html}<span class="muted">${((performance.now() - t0) / 1000).toFixed(0)} s</span>`; };
  tick(); statusTimer = setInterval(tick, 1000);
}

async function refresh() {
  if (!active) return;
  const data = UI.data, sel = UI.sel;
  if (!data || !sel) {
    clear(); $("kvtitle").innerHTML = "";
    setStatus("Select a cell (a layer and a position) in the Grid or J-Volume,<br>then come back here to see one head's keys from that cell's query.");
    return;
  }
  const [li, t] = sel, l = data.layers[li];
  const key = `${l}:${t}`;
  stopPlay();
  player.t = t; $("kvt").max = data.tokens.length - 1; $("kvt").value = t;
  $("kvtlab").innerHTML = `${UI.esc(UI.show(data.tokens[t]))} <span class="muted">${t + 1}/${data.tokens.length}</span>`;
  if (disksMode()) return refreshDisks(data, l, t);
  const keep = $("kvkeep").checked && head !== null;
  if (key !== cellKey) cellKey = key;
  const my = ++req;
  setStatus(`Asking the model for L${l}${keep ? ` head ${head}` : ""} into “${UI.esc(UI.show(data.tokens[t]))}”… (waits behind any running job)`, true);
  const d = await (await fetch(`/api/kv?l=${l}&t=${t}&h=${keep ? head : -1}`)).json();
  if (my !== req) return;
  if (d.error || d.detail) { setStatus(UI.esc(d.error || JSON.stringify(d.detail))); return; }
  head = d.h;
  $("kvhead").innerHTML = d.ranking.map(([h, sh]) => `<option value="${h}">head ${h} · ${(sh * 100).toFixed(1)}%</option>`).join("");
  $("kvhead").value = String(head);
  setStatus("");
  draw(d, data, t);
}

// ---------------------------------------------------------------- all heads as hyperbolic disks
// Looking down the query: centre = its tip. Radius is a Poincare-disk radius r = tanh(rho/2) with
// hyperbolic distance rho = -ln(w)/2, i.e. r = (1 - sqrt w) / (1 + sqrt w): a key taking all the
// attention sits at the centre, each e-fold less is an equal hyperbolic step out, zero is the rim.
// For softmax heads w is monotonic in the match, so the ordering is exactly the query-axis order.
const disksEl = $("kvdisks");
const disksMode = () => $("kvcam").value === "disks";
const rOf = (w) => { const q = Math.sqrt(Math.max(w, 1e-12)); return (1 - q) / (1 + q); };
const hsl = (f) => `hsl(${(200 + 215 * f) % 360} 90% 62%)`;

// Frames of the all-heads view, fetched per (layer, query position) and cached for playback.
let runId = 0;
const diskCache = new Map();
function fetchDisk(l, t) {
  const k = `${runId}:${l}:${t}`;
  if (!diskCache.has(k)) diskCache.set(k, fetch(`/api/kv_heads?l=${l}&t=${t}`).then((r) => r.json()).catch((e) => ({ error: String(e) })));
  return diskCache.get(k);
}

async function refreshDisks(data, l, t) {
  const my = ++req;
  setStatus(`Asking the model for every head of L${l} into “${UI.esc(UI.show(data.tokens[t]))}”… (waits behind any running job)`, true);
  const d = await fetchDisk(l, t);
  if (my !== req) return;
  if (d.error || d.detail) { diskCache.delete(`${runId}:${l}:${t}`); setStatus(UI.esc(d.error || JSON.stringify(d.detail))); return; }
  setStatus("");
  drawDisks(d, data, t);
}

// The disks are persistent DOM (one <g> per source key), so moving from one query position to the
// next only changes transforms and CSS transitions animate the keys to their new places.
let shell = null;
const S_ = 90;  // disk radius in svg units
const RINGS = [0.5, 0.1, 0.01, 0.001];
function buildShell(d, data, order) {
  const T = data.tokens.length;
  const tok = (i) => UI.esc(UI.show(data.tokens[i]).trim() || "·");
  const rings = RINGS.map((w) => `<circle r="${(rOf(w) * S_).toFixed(1)}" fill="none" stroke="var(--line)" stroke-width="0.7"/>`).join("");
  const keys = [...Array(T).keys()].map((i) =>
    `<g class="k" style="opacity:0"><circle r="2" fill="${hsl(T > 1 ? i / (T - 1) : 0)}" stroke-width="1.2"><title></title></circle><text x="7" y="3">${tok(i)}</text></g>`).join("");
  disksEl.innerHTML = order.map((h) => `<div class="kvdisk" data-h="${h}" title="Click for head ${h}'s number line">
      <div class="hd"><b>head ${h}</b><span class="sh"></span></div>
      <svg viewBox="${-S_ - 4} ${-S_ - 4} ${2 * S_ + 8} ${2 * S_ + 8}"><circle r="${S_}" fill="none" stroke="var(--line)" stroke-width="1.5"/>${rings}
        <circle class="zero" r="${S_}" fill="none" stroke="var(--muted)" stroke-width="0.8" stroke-dasharray="3 3" style="display:${d.kind === "attn" ? "" : "none"}"><title>zero match</title></circle>
        ${keys}<circle r="2" fill="var(--ink)"/></svg>
      <div class="top"></div></div>`).join("");
  const heads = new Map();
  for (const el of disksEl.querySelectorAll(".kvdisk"))
    heads.set(+el.dataset.h, { el, sh: el.querySelector(".sh"), top: el.querySelector(".top"), zero: el.querySelector(".zero"),
      keys: [...el.querySelectorAll("g.k")].map((g) => ({ g, c: g.querySelector("circle"), tip: g.querySelector("title"), txt: g.querySelector("text") })) });
  shell = { l: d.l, T, run: runId, kind: d.kind, order, heads, tok };
}

function drawDisks(d, data, t, reorder = true) {
  clear();
  const order = d.heads.map((x) => x.h);
  // Rebuild when the layer or run changes, or when a newly selected cell changes the strength order;
  // during playback and scrubbing the order is frozen so each head stays in its place.
  if (!shell || shell.l !== d.l || shell.T !== data.tokens.length || shell.run !== runId ||
      (reorder && order.join() !== shell.order.join())) buildShell(d, data, order);
  const tok = shell.tok;
  for (const hd of d.heads) {
    const ref = shell.heads.get(hd.h);
    // softmax weights already sum to 1; memory heads: share of the head's absolute effective weight
    const tot = d.kind === "attn" ? 1 : hd.w.reduce((a, b) => a + Math.abs(b), 0) || 1;
    const share = hd.w.map((w) => Math.abs(w) / tot);
    if (d.kind === "attn") {  // where a key with zero match would sit: w0 = exp(0 - lse)
      const m = Math.max(...hd.logit), lse = m + Math.log(hd.logit.reduce((a, b) => a + Math.exp(b - m), 0));
      ref.zero.style.r = (rOf(Math.exp(-lse)) * S_).toFixed(1) + "px";
      ref.zero.querySelector("title").textContent = `zero match (a key here would get ${(Math.exp(-lse) * 100).toPrecision(2)}%)`;
    }
    ref.keys.forEach((k, i) => {
      if (i >= hd.w.length) {  // not written yet at this query position: wait just outside the rim
        k.g.style.opacity = "0"; k.g.style.transform = `translate(0px, ${-S_ - 12}px)`; return;
      }
      const ang = Math.atan2(hd.z[i], hd.y[i]), r = rOf(share[i]) * S_;
      const x = r * Math.cos(ang), y = -r * Math.sin(ang), neg = hd.w[i] < 0;
      const dot = 1.6 + 5 * Math.sqrt(share[i]);
      if (k.g.style.opacity === "0") {  // newly written key: start from the rim in its direction
        k.g.style.transition = "none"; k.g.style.transform = `translate(${(S_ * Math.cos(ang)).toFixed(1)}px, ${(-S_ * Math.sin(ang)).toFixed(1)}px)`;
        k.g.getBoundingClientRect(); k.g.style.transition = "";
      }
      k.g.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
      k.g.style.opacity = neg ? "1" : String(0.35 + 0.65 * Math.min(1, share[i] * 5));
      k.c.style.r = dot.toFixed(1) + "px";
      k.c.setAttribute("fill", neg ? "none" : k.c.dataset.col || (k.c.dataset.col = k.c.getAttribute("fill")));
      k.c.setAttribute("stroke", neg ? "#ff5555" : "none");
      k.txt.setAttribute("x", (dot + 2).toFixed(1));
      k.txt.style.display = share[i] >= 0.08 ? "" : "none";
      k.tip.textContent = `${UI.show(data.tokens[i]).trim() || "·"}${i === hd.w.length - 1 ? " (itself)" : ""} · pos ${i} · ${(share[i] * 100).toPrecision(3)}%${neg ? " (net negative)" : ""} · match ${(hd.x[i] * hd.q_norm).toFixed(2)}`;
    });
    const best = share.indexOf(Math.max(...share));
    ref.sh.textContent = `${(hd.share * 100).toFixed(1)}% of layer`;
    ref.top.innerHTML = `top: <b>${tok(best)}</b> ${(share[best] * 100).toFixed(0)}%`;
  }
  const kind = d.kind === "gdn" ? `<span class="kindtag gdn">DeltaNet memory</span> ${d.heads.length} heads` : `<span class="kindtag attn">softmax attention</span> ${d.heads.length} heads`;
  $("kvtitle").innerHTML = `<b>L${d.l}</b> ${kind} · query of <b>${UI.esc(UI.show(data.tokens[t]))}</b> (pos ${t}) · ` +
    (reorder ? "sorted by share of what the layer brings here" : "order kept from the selected cell while playing / scrubbing") +
    `<div class="muted">centre = the query's tip (all attention) · each ring outward = less attention (50 / 10 / 1 / 0.1%) · rim = none · angle = how a key misses the query</div>`;
  layoutMode();
}

function layoutMode() {
  const on = disksMode();
  disksEl.style.display = on ? "grid" : "none";
  disksEl.style.top = host.querySelector(".vctl").offsetHeight + 4 + "px";
  $("kvhead").style.display = on ? "none" : "";
  $("kvkeep").parentElement.style.display = on ? "none" : "";
  $("kvreset").style.display = on ? "none" : "";
  $("kvhelpdisk").style.display = on ? "" : "none";
  $("kvhelpone").style.display = on ? "none" : "";
}
disksEl.addEventListener("click", (e) => {
  const el = e.target.closest(".kvdisk");
  if (!el) return;
  head = +el.dataset.h; $("kvkeep").checked = true; $("kvcam").value = lastCam = "front";
  layoutMode(); refresh();
});

// ---------------------------------------------------------------- playback
// Steps the query through the prompt at the selected layer: the disks (all heads) or one head's
// number line / 3D view, with keys gliding to their new places as each token's query takes over.
const player = { on: false, t: 0, timer: null };
const dur = () => 900 / +$("kvspeed").value;
const playLayer = () => UI.data.layers[UI.sel[0]];

function setPlaying(on) {
  player.on = on;
  clearTimeout(player.timer);
  $("kvplay").textContent = on ? "⏸" : "▶";
  $("kvplay").title = on ? "Pause" : "Play: step the query through every token of the prompt at this layer";
}
function stopPlay() { if (player.on) setPlaying(false); }

async function showFrame(t) {
  const data = UI.data, l = playLayer(), T = data.tokens.length;
  player.t = t;
  $("kvt").max = T - 1; $("kvt").value = t;
  $("kvtlab").innerHTML = `${UI.esc(UI.show(data.tokens[t]))} <span class="muted">${t + 1}/${T}</span>`;
  document.documentElement.style.setProperty("--kvdur", `${Math.min(0.7, dur() / 1000 * 0.7)}s`);
  if (disksMode()) {
    const my = ++req;
    const d = await fetchDisk(l, t);
    for (let k = 1; k <= 3 && t + k < T; k++) fetchDisk(l, t + k);  // prefetch the next frames
    if (my !== req) return false;
    if (d.error || d.detail) { setStatus(UI.esc(d.error || JSON.stringify(d.detail))); return false; }
    setStatus(""); drawDisks(d, data, t, false);
  } else {
    if (head === null) return false;
    const tr = await ensureTrack(l, head);
    if (!tr) return false;
    setTrackFrame(t);
  }
  return true;
}

async function playStep() {
  if (!player.on) return;
  const ok = await showFrame(player.t);
  if (!player.on) return;
  if (!ok || player.t >= UI.data.tokens.length - 1) { setPlaying(false); return; }
  player.timer = setTimeout(() => { player.t++; playStep(); }, dur());
}

// One head over all query positions, for the number line / 3D view. Scales are fixed over the whole
// run (unlike the still view) so movement between frames is real movement.
let track = null;
async function ensureTrack(l, h) {
  if (track && track.l === l && track.h === h && track.run === runId) return track;
  const my = ++req;
  setStatus(`Loading head ${h} of L${l} for every query position…`, true);
  const tr = await (await fetch(`/api/kv_track?l=${l}&h=${h}`)).json();
  if (my !== req) return null;
  if (tr.error || tr.detail) { setStatus(UI.esc(tr.error || JSON.stringify(tr.detail))); return null; }
  setStatus("");
  buildTrack(tr, UI.data);
  return track;
}

function buildTrack(tr, data) {
  clear();
  const T = data.tokens.length, F = tr.frames;
  const amax = (key) => Math.max(1e-6, ...F.flatMap((f) => f[key].map(Math.abs)));
  const sx = R / amax("x"), sy = (R * 0.7) / amax("y"), sz = (R * 0.7) / amax("z");
  const wmax = tr.kind === "attn" ? 1 : amax("weight");
  const qlen = R * 1.15;
  group.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), qlen, 0xffffff, 2.2, 1.2));
  const qlab = label("", "kvq", new THREE.Vector3(qlen + 2, 0, 0));
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(R * 2.2, R * 2.2),
    new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }));
  plane.rotation.y = Math.PI / 2; group.add(plane);
  label("0 = no match", "kvzero", new THREE.Vector3(0, -R * 0.78, 0));
  const tickObj = (text, up) => {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -1.2, 0), new THREE.Vector3(0, 1.2, 0)]);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 }));
    group.add(line);
    const lab = label(text, up ? "kvtick up" : "kvtick", new THREE.Vector3(0, up ? 1.2 : -1.2, 0));
    return { line, lab, x: 0, tx: 0, up };
  };
  const ticks = [];
  if (tr.kind === "attn") {
    [0.9, 0.5, 0.1, 0.01, 0.001].forEach((w, i) => ticks.push(Object.assign(tickObj(`${w >= 0.01 ? w * 100 : (w * 100).toFixed(1)}%`, i % 2 === 1), { w })));
  } else {
    const span = amax("x") * Math.max(...F.map((f) => f.q_norm)), step = Math.pow(10, Math.floor(Math.log10(span / 2 || 1)));
    const qn = F[F.length - 1].q_norm;  // memory ticks: match score at the last query's scale
    let i = 0;
    for (let v = -Math.ceil(span / step) * step; v <= span * 1.05; v += step) {
      const o = tickObj(v.toFixed(Math.max(0, -Math.floor(Math.log10(step)))), i++ % 2 === 1);
      o.x = o.tx = (v / qn) * sx; ticks.push(o);
    }
  }
  const keys = [];
  for (let i = 0; i < T; i++) {
    const col = posColor(T > 1 ? i / (T - 1) : 0);
    const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.5, transparent: true, opacity: 1 });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), mat);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1.25, 0.1, 6, 24), new THREE.MeshBasicMaterial({ color: 0xff5555 }));
    const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.25 }));
    const lab = label("", "kvtok", new THREE.Vector3());
    mesh.scale.setScalar(0); ring.visible = false; line.visible = false; lab.visible = false;
    group.add(mesh, ring, line);
    keys.push({ mesh, ring, line, lab, pos: new THREE.Vector3(), tpos: new THREE.Vector3(), r: 0, tr: 0, neg: false, on: false });
  }
  track = { l: tr.l, h: tr.h, run: runId, tr, sx, sy, sz, wmax, keys, ticks, qlab };
  setTrackFrame(F.length - 1, true);
  frame();
}

function setTrackFrame(t, snap = false) {
  const { tr, sx, sy, sz, wmax, keys, ticks, qlab } = track, f = tr.frames[t], data = UI.data;
  const tok = (i) => UI.esc(UI.show(data.tokens[i]).trim() || "·");
  const strong = new Set([...f.share.keys()].sort((a, b) => f.share[b] - f.share[a]).slice(0, 4).filter((i) => f.share[i] > 0.005));
  keys.forEach((k, i) => {
    const on = i <= t;
    if (!on) { k.on = false; k.tr = 0; k.lab.visible = false; k.line.visible = false; k.ring.visible = false; return; }
    const w = f.weight[i];
    k.tpos.set(f.x[i] * sx, f.y[i] * sy, f.z[i] * sz);
    k.tr = 0.35 + 2.2 * Math.sqrt(Math.abs(w) / wmax);
    if (!k.on || snap) { k.pos.copy(k.tpos); if (snap) k.r = k.tr; }  // a new key appears in place and grows
    k.on = true; k.neg = w < 0;
    k.mesh.material.opacity = k.neg ? 0.55 : 1;
    k.ring.visible = k.neg; k.line.visible = true;
    k.lab.visible = true;
    k.lab.element.className = "vlab " + (strong.has(i) ? "kvtok strong" : "kvtok");
    k.lab.element.innerHTML = tok(i) + (i === t ? " <i>(itself)</i>" : "") + (strong.has(i) ? ` <span class="kvw">${(f.share[i] * 100).toFixed(1)}%</span>` : "");
  });
  if (tr.kind === "attn") {  // weight ticks move with this query's normaliser
    const m = Math.max(...f.logit), lse = m + Math.log(f.logit.reduce((a, b) => a + Math.exp(b - m), 0));
    for (const o of ticks) { o.tx = ((Math.log(o.w) + lse) / (f.q_norm * tr.scale)) * sx; if (snap) o.x = o.tx; }
  }
  qlab.element.innerHTML = `query of <b>${UI.esc(UI.show(data.tokens[t]))}</b> → more match`;
  const kind = tr.kind === "gdn" ? `<span class="kindtag gdn">DeltaNet memory</span>` : `<span class="kindtag attn">softmax attention</span>`;
  $("kvtitle").innerHTML = `<b>L${tr.l}</b> ${kind} head ${tr.h} · query of <b>${UI.esc(UI.show(data.tokens[t]))}</b> (pos ${t}) · this head carries ${(f.head_share * 100).toFixed(1)}% of what the layer brings here` +
    `<div class="muted">playback: axes fixed over the whole prompt · x = match with the current query · size = weight used · labels = strongest sources and their share of the layer</div>`;
  stepTrack(snap ? 1 : 0);
}

function stepTrack(k) {  // ease every object toward its target; k = 1 snaps
  if (!track) return;
  for (const o of track.keys) {
    o.pos.lerp(o.tpos, k); o.r += ((o.on ? o.tr : 0) - o.r) * k;
    o.mesh.position.copy(o.pos); o.mesh.scale.setScalar(Math.max(o.r, 1e-3));
    o.ring.position.copy(o.pos); o.ring.scale.setScalar(Math.max(o.r, 1e-3));
    const a = o.line.geometry.attributes.position.array;
    a[0] = o.pos.x; a[1] = o.pos.y; a[2] = o.pos.z; a[3] = o.pos.x; a[4] = 0; a[5] = 0;
    o.line.geometry.attributes.position.needsUpdate = true;
    o.lab.position.set(o.pos.x, o.pos.y + o.r + 0.9, o.pos.z);
  }
  for (const o of track.ticks) {
    o.x += (o.tx - o.x) * k;
    o.line.position.x = o.x; o.lab.position.x = o.x;
    const vis = o.x > -R * 1.6 && o.x < R * 1.6;
    o.line.visible = vis; o.lab.visible = vis;
  }
}

function draw(d, data, t) {
  clear();
  const n = d.x.length, T = data.tokens.length;
  // Axes are scaled separately: keys vary far more orthogonally to the query than along it, and
  // only x (the match) is quantitative; y/z just spread tokens by similarity.
  const amax = (a) => Math.max(1e-6, ...a.map(Math.abs));
  const sx = R / amax(d.x), sy = (R * 0.7) / amax(d.y), sz = (R * 0.7) / amax(d.z);
  const P = (i) => new THREE.Vector3(d.x[i] * sx, d.y[i] * sy, d.z[i] * sz);
  const wmax = Math.max(1e-9, ...d.weight.map(Math.abs));
  const top = new Map(d.top.map((x) => [x.s, x.words]));
  // query arrow along +x from the origin
  const qlen = R * 1.15;
  group.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), qlen, 0xffffff, 2.2, 1.2));
  label(`query of <b>${UI.esc(UI.show(data.tokens[t]))}</b> → more match`, "kvq", new THREE.Vector3(qlen + 2, 0, 0));
  // faint plane of zero match
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(R * 2.2, R * 2.2),
    new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }));
  plane.rotation.y = Math.PI / 2; group.add(plane);
  label("0 = no match", "kvzero", new THREE.Vector3(0, -R * 0.78, 0));
  // Ticks along the query axis. Softmax heads: where a key would get 90/50/10/1/0.1% of the
  // attention (weight is monotonic in x). Memory heads: plain match-score ticks.
  const xs = d.x.map((v) => v * sx), xmin = Math.min(0, ...xs) - 2, xmax = Math.max(...xs, 0) + 4;
  let nTick = 0;
  const tick = (xw, text) => {
    if (xw < xmin || xw > xmax) return;
    const up = nTick++ % 2 === 1;  // alternate labels above/below the axis so neighbours don't collide
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xw, -1.2, 0), new THREE.Vector3(xw, 1.2, 0)]);
    group.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 })));
    label(text, up ? "kvtick up" : "kvtick", new THREE.Vector3(xw, up ? 1.2 : -1.2, 0));
  };
  if (d.kind === "attn") {
    const lse = Math.log(d.logit.reduce((a, b) => a + Math.exp(b - Math.max(...d.logit)), 0)) + Math.max(...d.logit);
    for (const w of [0.9, 0.5, 0.1, 0.01, 0.001])
      tick(((Math.log(w) + lse) / (d.q_norm * d.scale)) * sx, `${w >= 0.01 ? w * 100 : (w * 100).toFixed(1)}%`);
  } else {
    const span = Math.max(...d.x.map(Math.abs)), step = Math.pow(10, Math.floor(Math.log10(span / 2 || 1)));
    for (let v = -Math.ceil(span / step) * step; v <= span * 1.05; v += step) tick(v * sx, (v * d.q_norm).toFixed(Math.max(0, -Math.floor(Math.log10(step * d.q_norm || 1)))));
  }
  // keys
  for (let i = 0; i < n; i++) {
    const w = d.weight[i], a = Math.abs(w) / wmax;
    const col = posColor(T > 1 ? i / (T - 1) : 0);
    const r = 0.35 + 2.2 * Math.sqrt(a);
    const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.5, transparent: w < 0, opacity: w < 0 ? 0.55 : 1 });
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 12), mat);
    m.position.copy(P(i)); group.add(m);
    if (w < 0) {  // DeltaNet: negative effective weight (net subtraction after overwrites)
      const ring = new THREE.Mesh(new THREE.TorusGeometry(r * 1.25, 0.12, 6, 24), new THREE.MeshBasicMaterial({ color: 0xff5555 }));
      ring.position.copy(P(i)); group.add(ring);
    }
    // dotted line from the key down to the query axis: its x is the match
    const g = new THREE.BufferGeometry().setFromPoints([P(i), new THREE.Vector3(P(i).x, 0, 0)]);
    group.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.25 })));
    const words = top.get(i);
    const share = d.share[i];
    const txt = `${UI.esc(UI.show(data.tokens[i]).trim() || "·")}` + (i === t ? " <i>(itself)</i>" : "") +
      (words ? ` <span class="kvw">${(share * 100).toFixed(1)}% · ${UI.esc(words.slice(0, 2).map((x) => UI.show(x).trim()).join(" "))}</span>` : "");
    label(txt, words ? "kvtok strong" : "kvtok", P(i).clone().add(new THREE.Vector3(0, r + 0.9, 0)));
  }
  const kind = d.kind === "gdn"
    ? `<span class="kindtag gdn">DeltaNet memory</span> head ${d.h} (reads key head ${Math.floor(d.h / (d.heads / d.key_heads))})`
    : `<span class="kindtag attn">softmax attention</span> head ${d.h} (shares KV head ${Math.floor(d.h / (d.heads / d.kv_heads))})`;
  const corr = (() => { const xs = d.x, ws = d.weight, n_ = xs.length; if (n_ < 3) return null;
    const mx_ = xs.reduce((a, b) => a + b) / n_, mw = ws.reduce((a, b) => a + b) / n_;
    let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n_; i++) { sxy += (xs[i] - mx_) * (ws[i] - mw); sxx += (xs[i] - mx_) ** 2; syy += (ws[i] - mw) ** 2; }
    return sxy / Math.sqrt(sxx * syy + 1e-12); })();
  layoutMode();
  $("kvtitle").innerHTML = `<b>L${d.l}</b> ${kind} · into <b>${UI.esc(UI.show(data.tokens[t]))}</b> (pos ${t}) · this head carries ${(d.head_share * 100).toFixed(1)}% of what the layer brings here` +
    `<div class="muted">x = match between the query and each key${d.kind === "attn" ? " (× |q| × scale = the softmax logit)" : ""} · size = weight actually used` +
    (d.kind === "gdn" ? " — for memory heads that is the effective weight after decay and overwriting, so a key can match yet be forgotten (red ring = net negative)" : "") +
    (corr !== null ? ` · match↔weight correlation <b>${corr.toFixed(2)}</b>` : "") +
    ` · only x is to scale; y/z just spread tokens by similarity</div>`;
  frame();
}
function frame() {  // fit the camera to what was drawn
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  controls.target.copy(c);
  if ($("kvcam").value === "front") {
    // Number line: look straight down the depth axis with a narrow lens, so x reads left to right
    // almost to scale; orbiting is off (pan and zoom still work).
    camera.fov = 12; camera.updateProjectionMatrix();
    const fit = Math.max(size.x / camera.aspect, size.y) * 1.25;
    camera.position.set(c.x, c.y, c.z + fit / (2 * Math.tan((camera.fov * Math.PI) / 360)));
    controls.enableRotate = false;
  } else {
    camera.fov = 40; camera.updateProjectionMatrix();
    const len = size.length();
    camera.position.set(c.x + len * 0.25, c.y + len * 0.35, c.z + len * 1.05);
    controls.enableRotate = true;
  }
  controls.update();
}

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  if (!w || !h) return;
  renderer.setSize(w, h); labels.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
function resetCamera() { camera.position.set(R * 0.6, R * 0.9, R * 2.6); controls.target.set(R * 0.2, 0, 0); controls.update(); }
resetCamera();

$("kvhead").addEventListener("change", (e) => { stopPlay(); head = +e.target.value; $("kvkeep").checked = true; refresh(); });
let lastCam = $("kvcam").value;
$("kvcam").addEventListener("change", () => {
  stopPlay();
  const was = lastCam === "disks", now = disksMode(); lastCam = $("kvcam").value;
  layoutMode();
  if (was !== now) refresh(); else frame();
});
$("kvreset").addEventListener("click", () => (group.children.length ? frame() : resetCamera()));
window.addEventListener("lens:kvhead", (e) => { head = e.detail; cellKey = null; $("kvkeep").checked = true; $("kvcam").value = lastCam = "front"; layoutMode(); refresh(); });
layoutMode();
window.addEventListener("lens:view", (e) => {
  active = e.detail === "kv";
  let last = performance.now();
  renderer.setAnimationLoop(active ? (now) => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    stepTrack(1 - Math.exp(-dt * 6000 / Math.max(300, dur())));
    controls.update(); renderer.render(scene, camera); labels.render(scene, camera);
  } : null);
  if (!active) stopPlay();
  if (active) { resize(); refresh(); }
});
window.addEventListener("lens:select", () => { if (active) refresh(); });
window.addEventListener("lens:data", () => { cellKey = null; runId++; diskCache.clear(); shell = null; track = null; if (active) refresh(); });
$("kvplay").addEventListener("click", () => {
  if (!UI.data || !UI.sel) return;
  if (player.on) { setPlaying(false); return; }
  if (player.t >= UI.data.tokens.length - 1) player.t = 0;
  if (!disksMode() && head === null) return;
  setPlaying(true); playStep();
});
$("kvt").addEventListener("input", (e) => { if (!UI.data || !UI.sel) return; stopPlay(); showFrame(+e.target.value); });
$("kvspeed").addEventListener("change", () => document.documentElement.style.setProperty("--kvdur", `${Math.min(0.7, dur() / 1000 * 0.7)}s`));
