"use strict";

const NAMES = {
  akiec: "Queratosis actínica / carcinoma in situ",
  bcc: "Carcinoma basocelular",
  bkl: "Queratosis benigna",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Nevus (lunar común)",
  vasc: "Lesión vascular",
};
const LEVELS = {
  low: { label: "Riesgo bajo", text: "El modelo no encuentra rasgos sospechosos relevantes. Sigue vigilando el lunar (regla ABCDE) y consulta si cambia de tamaño, forma o color." },
  mid: { label: "Riesgo moderado", text: "El modelo detecta algunos rasgos compatibles con lesiones malignas. Se recomienda una revisión por un dermatólogo." },
  high: { label: "Riesgo alto", text: "El modelo detecta rasgos fuertemente asociados a lesiones malignas. Consulta a un dermatólogo lo antes posible." },
};
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const SIZE = 224, HIST_KEY = "dermascan.history";

const $ = (id) => document.getElementById(id);
let cfg, session, img, view = { scale: 1, x: 0, y: 0 };
// detector neuronal "lunar / no lunar" (opcional) y lunares encontrados en la foto actual
let detSession = null, cands = [], selected = -1;
// segmentador neuronal (contorno preciso de la lesión), opcional
let segSession = null, findRun = 0;

// ---------- navegación ----------
function show(id) {
  if (id !== "vCamera" && typeof stopCamera === "function") stopCamera();
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== id;
  window.scrollTo(0, 0);
}
function busy(text) {
  $("busy").hidden = !text;
  if (text) $("busyText").textContent = text;
}

// ---------- modelo ----------
async function loadModel() {
  if (session) return session;
  busy("Cargando modelo…");
  ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  cfg = await (await fetch("model/config.json")).json();
  // la ruta del modelo puede venir en config.json (en la versión web se publica con otra extensión)
  const bytes = new Uint8Array(await (await fetch(cfg.model_file || "model/model.onnx")).arrayBuffer());
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  if (cfg.det_file) {
    try {
      const db = new Uint8Array(await (await fetch(cfg.det_file)).arrayBuffer());
      detSession = await ort.InferenceSession.create(db, { executionProviders: ["wasm"] });
    } catch (e) { console.warn("detector no disponible", e); }
  }
  if (cfg.seg_file) {
    try {
      const sb = new Uint8Array(await (await fetch(cfg.seg_file)).arrayBuffer());
      segSession = await ort.InferenceSession.create(sb, { executionProviders: ["wasm"] });
    } catch (e) { console.warn("segmentador no disponible", e); }
  }
  busy(null);
  return session;
}

// probabilidad de "hay un lunar" en un recorte cuadrado (red neuronal); null si no hay detector
async function lesionProb(canvas) {
  if (!detSession) return null;
  const c = canvas.width === 128 ? canvas : (() => {
    const t = document.createElement("canvas"); t.width = t.height = 128;
    t.getContext("2d").drawImage(canvas, 0, 0, 128, 128); return t;
  })();
  return (await detSession.run({ input: toTensor(c, false, 128) })).probs.data[1];
}
const isLesion = (p, fallback) => (p == null ? fallback : p >= cfg.det_threshold);
// probabilidad de lesión por píxel en un recorte de SEG×SEG (segmentador)
async function segProb(canvas) {
  return (await segSession.run({ input: toTensor(canvas, false, SEG) })).mask.data;
}

// busca todos los lunares de la foto. Con segmentador: primero el del centro y luego el resto, pintando
// cada uno según aparece; sin él, el detector clásico propone manchas y la red confirma cada una
async function findLesions() {
  const run = ++findRun;
  if (segSession) {
    $("candBox").hidden = false;
    $("candText").textContent = "Buscando lunares…";
    $("candChips").innerHTML = "";
    const res = await locateLesionsSeg(img, img.width, img.height, segProb, detSession && lesionProb, cfg.det_threshold, (list) => {
      if (run !== findRun) return;
      mergeFound(list);
      draw();
    });
    if (run !== findRun) return;
    mergeFound(res);
  } else {
    cands = detSession ? await locateLesions(img, img.width, img.height, lesionProb, cfg.det_threshold) : [];
    selected = cands.length ? 0 : -1;
  }
  renderCandList();
  draw();
}
// añade lo encontrado sin perder lo que el usuario ya tocó (lunares marcados a mano y su selección)
function mergeFound(list) {
  const cur = selected >= 0 ? cands[selected] : null, manual = cands.filter((c) => c.manual);
  const same = (a, b) => Math.abs(a.cx - b.cx) < Math.max(a.w, b.w) * 0.5 && Math.abs(a.cy - b.cy) < Math.max(a.h, b.h) * 0.5;
  cands = manual.concat(list.filter((c) => !manual.some((m) => same(m, c))));
  selected = cur ? Math.max(0, cands.findIndex((c) => same(c, cur))) : cands.length ? 0 : -1;
}

// ---------- lunares detectados: contorno, número y selección ----------
function toCanvasXY(nx, ny) {
  const cv = $("cropCanvas"), s = baseScale() * view.scale;
  return [(cv.width - img.width * s) / 2 + view.x + nx * img.width * s, (cv.height - img.height * s) / 2 + view.y + ny * img.height * s];
}
// dibuja contornos (escala 600 = lienzo de encuadre). onlySel: solo el elegido (para la miniatura del resultado)
function drawCands(ctx, size, onlySel) {
  const k = size / 600, s = baseScale() * view.scale;
  cands.forEach((c, i) => {
    if (onlySel && i !== selected) return;
    const sel = i === selected;
    if (c.contour) {
      drawContour(ctx, c.contour.map(([x, y]) => toCanvasXY(x, y).map((v) => v * k)), sel ? "#4ade80" : "rgba(255,255,255,0.9)",
                  (sel ? 3 : 2) * Math.max(0.5, k));
    } else {
      const dot = Math.max(2, c.px * Math.max(img.width, img.height) * s * 0.9) * k;
      ctx.fillStyle = sel ? "#4ade80" : "rgba(255,255,255,0.85)";
      for (let j = 0; j < c.edge.length; j += 2) {
        const [x, y] = toCanvasXY(c.edge[j], c.edge[j + 1]);
        ctx.fillRect(x * k - dot / 2, y * k - dot / 2, dot, dot);
      }
    }
    if (onlySel) return;
    const [bx, by] = toCanvasXY(c.cx, c.cy - c.h / 2);
    ctx.beginPath(); ctx.arc(bx, Math.max(16, by - 16), 13, 0, 2 * Math.PI);
    ctx.fillStyle = sel ? "#4ade80" : "#ffffff"; ctx.fill();
    ctx.fillStyle = "#0b1716"; ctx.font = "700 15px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), bx, Math.max(16, by - 16));
  });
}
function renderCandList() {
  const n = cands.length;
  $("candBox").hidden = false;
  const hint = segSession ? " Si no es tu lunar, toca el lunar en la foto." : "";
  $("candText").textContent = !n
    ? (segSession ? "No se detectó ningún lunar con seguridad. Toca el lunar en la foto para marcarlo, o encuádralo a mano."
       : detSession ? "No se detectó ningún lunar con seguridad. Encuadra el lunar dentro del círculo a mano." : "")
    : n === 1 ? "Lunar detectado (contorno verde). Tócalo para centrarlo y ampliarlo." + hint
    : `Se detectaron ${n} lunares. Toca el que quieres analizar, en la foto o aquí abajo.` + hint;
  $("candChips").innerHTML = n > 1 ? cands.map((c, i) =>
    `<button class="chip${i === selected ? " sel" : ""}" data-i="${i}">Lunar ${i + 1}</button>`).join("") : "";
}
// elige un lunar y lo centra/amplía para que la red lo vea como en el entrenamiento
function selectCand(i) {
  const c = cands[i], cv = $("cropCanvas");
  if (!c) return;
  selected = i;
  const side = Math.max(c.w * img.width, c.h * img.height) * 2.5;
  view.scale = Math.min(6, Math.max(1, cv.width / side / baseScale()));
  const s = baseScale() * view.scale;
  view.x = -(c.cx * img.width - img.width / 2) * s;
  view.y = -(c.cy * img.height - img.height / 2) * s;
  $("zoom").value = view.scale;
  renderCandList();
  draw();
}
async function tapAt(clientX, clientY) {
  const cv = $("cropCanvas"), r = cv.getBoundingClientRect(), s = baseScale() * view.scale;
  const X = (clientX - r.left) * (cv.width / r.width), Y = (clientY - r.top) * (cv.height / r.height);
  const nx = (X - ((cv.width - img.width * s) / 2 + view.x)) / (img.width * s);
  const ny = (Y - ((cv.height - img.height * s) / 2 + view.y)) / (img.height * s);
  let bestI = -1, bestD = Infinity;
  cands.forEach((c, i) => {
    const d = Math.hypot((nx - c.cx) / Math.max(c.w, 0.03), (ny - c.cy) / Math.max(c.h, 0.03));
    if (d < 1.5 && d < bestD) { bestD = d; bestI = i; }
  });
  if (bestI >= 0) return selectCand(bestI);
  // ningún lunar marcado ahí: el segmentador busca la lesión en el punto tocado
  if (!segSession || nx < 0 || ny < 0 || nx > 1 || ny > 1) return;
  const run = findRun;
  $("candText").textContent = "Buscando el lunar donde tocaste…";
  // escalas relativas a lo que se ve en pantalla, para que el zoom del usuario ayude
  const vis = 1 / view.scale;
  const l = await lesionAt(img, img.width, img.height, nx, ny, segProb, detSession && lesionProb, cfg.det_threshold,
                           { keepAll: true, scales: [0.15, 0.3, 0.6].map((f) => f * vis) });
  if (run !== findRun) return;
  if (!l) {
    renderCandList();
    $("candText").textContent = "No encontré un borde claro donde tocaste. Amplía con el zoom y toca justo encima del lunar.";
    return;
  }
  l.manual = true;
  cands.push(l);
  selectCand(cands.length - 1);
  if (l.p != null && l.p < cfg.det_threshold)
    $("candText").textContent = "Marcado donde tocaste, pero el detector no está seguro de que sea un lunar. Revisa el contorno antes de analizar.";
}

// ---------- carga y encuadre de la imagen ----------
async function onFile(e) {
  const file = e.target.files[0];
  e.target.value = "";
  if (file) openImage(URL.createObjectURL(file)).catch(fail);
}

async function openImage(src) {
  $("errBox").hidden = true;
  const el = new Image();
  el.src = src;
  try { await el.decode(); } catch { throw new Error("No se pudo abrir la imagen. Prueba con otra foto (JPG o PNG)."); }
  openFrame(el, el.naturalWidth, el.naturalHeight);
  if (src.startsWith("blob:")) URL.revokeObjectURL(src);
}

// pasa una imagen o un fotograma de video a la pantalla de encuadre
function openFrame(source, w, h) {
  // reducir fotos de 12 MP para que el encuadre sea fluido
  const k = Math.min(1, 1600 / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * k); c.height = Math.round(h * k);
  c.getContext("2d").drawImage(source, 0, 0, c.width, c.height);
  img = c;
  $("zoom").value = 1;
  view = { scale: 1, x: 0, y: 0 };
  cands = []; selected = -1; findRun++;
  $("candBox").hidden = true;
  show("vCrop");
  draw();
  loadModel().then(findLesions).catch(fail);
}

function baseScale() {
  const cv = $("cropCanvas");
  return Math.max(cv.width / img.width, cv.height / img.height);
}
function clamp() {
  const cv = $("cropCanvas"), s = baseScale() * view.scale;
  const mx = Math.max(0, (img.width * s - cv.width) / 2), my = Math.max(0, (img.height * s - cv.height) / 2);
  view.x = Math.max(-mx, Math.min(mx, view.x));
  view.y = Math.max(-my, Math.min(my, view.y));
}
function draw() {
  const cv = $("cropCanvas"), ctx = cv.getContext("2d"), s = baseScale() * view.scale;
  clamp();
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.imageSmoothingQuality = "high";
  const w = img.width * s, h = img.height * s;
  ctx.drawImage(img, (cv.width - w) / 2 + view.x, (cv.height - h) / 2 + view.y, w, h);
  // los contornos van en un lienzo aparte para no contaminar la imagen que analiza el modelo
  const octx = $("cropOverlay").getContext("2d");
  octx.clearRect(0, 0, 600, 600);
  drawCands(octx, 600, false);
  checkQuality();
}

function setupPan() {
  const box = $("cropBox"), pts = new Map();
  let last = null, pinch = null;
  const ratio = () => $("cropCanvas").width / box.clientWidth;
  let down = null, multi = false;
  box.addEventListener("pointerdown", (e) => {
    box.setPointerCapture(e.pointerId); pts.set(e.pointerId, e); last = e;
    if (pts.size === 1) { down = { x: e.clientX, y: e.clientY }; multi = false; } else multi = true;
  });
  box.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, e);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (pinch) setZoom(view.scale * d / pinch);
      pinch = d;
    } else if (last) {
      view.x += (e.clientX - last.clientX) * ratio();
      view.y += (e.clientY - last.clientY) * ratio();
      last = e;
      draw();
    }
  });
  const up = (e) => {
    // un toque corto sin arrastrar selecciona el lunar tocado
    if (e.type === "pointerup" && down && !multi && pts.size === 1 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 8)
      tapAt(e.clientX, e.clientY).catch(fail);
    pts.delete(e.pointerId); pinch = null; last = [...pts.values()][0] || null;
  };
  box.addEventListener("pointerup", up);
  box.addEventListener("pointercancel", up);
  box.addEventListener("wheel", (e) => { e.preventDefault(); setZoom(view.scale * (e.deltaY < 0 ? 1.1 : 0.9)); }, { passive: false });
  $("zoom").addEventListener("input", (e) => setZoom(+e.target.value));
}
function setZoom(z) {
  view.scale = Math.max(1, Math.min(6, z));
  $("zoom").value = view.scale;
  draw();
}

// imagen recortada a 224x224 (lo que ve el modelo)
function cropped(size = SIZE) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage($("cropCanvas"), 0, 0, size, size);
  return c;
}

// ---------- control de calidad de la foto (brillo y nitidez) ----------
let qTimer;
function checkQuality() {
  clearTimeout(qTimer);
  qTimer = setTimeout(() => {
    const { data } = cropped(160).getContext("2d").getImageData(0, 0, 160, 160);
    const g = new Float32Array(160 * 160);
    let sum = 0;
    for (let i = 0; i < g.length; i++) { g[i] = 0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]; sum += g[i]; }
    const mean = sum / g.length;
    // varianza del laplaciano: valores bajos = foto desenfocada
    let s1 = 0, s2 = 0, n = 0;
    for (let y = 1; y < 159; y++) for (let x = 1; x < 159; x++) {
      const i = y * 160 + x, l = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - 160] - g[i + 160];
      s1 += l; s2 += l * l; n++;
    }
    const lapVar = s2 / n - (s1 / n) ** 2;
    const msgs = [];
    if (mean < 60) msgs.push("La foto está muy oscura; busca más luz.");
    if (mean > 225) msgs.push("La foto está sobreexpuesta; evita el flash o la luz directa.");
    if (lapVar < 25) msgs.push("La foto parece desenfocada; vuelve a tomarla esperando el enfoque.");
    $("quality").hidden = !msgs.length;
    $("quality").textContent = msgs.join(" ");
  }, 120);
}

// ---------- inferencia ----------
function toTensor(canvas, flip, size = SIZE) {
  const { data } = canvas.getContext("2d").getImageData(0, 0, size, size);
  const t = new Float32Array(3 * size * size), plane = size * size;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const src = 4 * (y * size + (flip ? size - 1 - x : x)), dst = y * size + x;
    for (let c = 0; c < 3; c++) t[c * plane + dst] = (data[src + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor("float32", t, [1, 3, size, size]);
}

async function analyze() {
  try {
    await loadModel();
    busy("Analizando…");
    const c = cropped();
    // TTA: original + volteo horizontal (igual que en la evaluación de la tesis)
    const a = (await session.run({ input: toTensor(c, false) })).probs.data;
    const b = (await session.run({ input: toTensor(c, true) })).probs.data;
    const probs = Array.from(a, (v, i) => (v + b[i]) / 2);
    const risk = cfg.malignant_classes.reduce((s, k) => s + probs[cfg.classes.indexOf(k)], 0);
    const level = risk >= cfg.threshold_high ? "high" : risk >= cfg.threshold_moderate ? "mid" : "low";
    // miniatura con el contorno del lunar analizado, para comprobar qué se analizó
    const tc = cropped(224);
    drawCands(tc.getContext("2d"), 224, true);
    const thumb = tc.toDataURL("image/jpeg", 0.85);
    const c128 = cropped(128);
    $("detNote").hidden = isLesion(await lesionProb(c128), detectLesion(c128.getContext("2d").getImageData(0, 0, 128, 128)).found);
    renderResult({ probs, risk, level, thumb, date: Date.now() });
    saveHistory({ risk, level, thumb, date: Date.now(), top: cfg.classes[probs.indexOf(Math.max(...probs))] });
  } catch (e) { fail(e); } finally { busy(null); }
}

function renderResult({ probs, risk, level, thumb }) {
  $("riskCard").className = "risk " + level;
  $("resImg").src = thumb;
  $("riskLabel").textContent = LEVELS[level].label;
  $("riskScore").textContent = `Índice de malignidad: ${(risk * 100).toFixed(1)}%`;
  $("riskText").textContent = LEVELS[level].text;
  const order = cfg.classes.map((k, i) => [k, probs[i]]).sort((x, y) => y[1] - x[1]);
  $("bars").innerHTML = order.map(([k, p]) => `
    <div class="bar-row${cfg.malignant_classes.includes(k) ? " mal" : ""}">
      <span class="name">${NAMES[k]}</span><span class="pct">${(p * 100).toFixed(1)}%</span>
      <div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
    </div>`).join("");
  const m = cfg.metrics;
  $("modelStats").textContent = `Rendimiento en el conjunto de prueba ${m.test_set} (${m.n_test} imágenes no vistas): AUC ${m.auc.toFixed(3)}, sensibilidad ${(m.sensitivity * 100).toFixed(1)}%, especificidad ${(m.specificity * 100).toFixed(1)}%.`;
  show("vResult");
}

// ---------- historial (solo en este navegador) ----------
function readHistory() {
  try { return JSON.parse(localStorage.getItem(HIST_KEY)) || []; } catch { return []; }
}
function saveHistory(entry) {
  try { localStorage.setItem(HIST_KEY, JSON.stringify([entry, ...readHistory()].slice(0, 30))); } catch {}
}
function renderHistory() {
  const h = readHistory();
  $("histList").innerHTML = h.length ? h.map((e) => `
    <div class="hist-item">
      <img src="${e.thumb}" alt="">
      <div><span class="lvl ${e.level}">${LEVELS[e.level].label}</span> · ${(e.risk * 100).toFixed(1)}%
        <small>${new Date(e.date).toLocaleString("es")} · ${NAMES[e.top]}</small></div>
    </div>`).join("") : `<p class="muted">Aún no hay análisis.</p>`;
  show("vHistory");
}

function fail(e) {
  console.error(e);
  busy(null);
  $("errBox").textContent = "No se pudo completar el análisis: " + (e && e.message ? e.message : e);
  $("errBox").hidden = false;
  window.scrollTo(0, 0);
}

// ---------- arranque ----------
$("inCamera").addEventListener("change", onFile);
$("inGallery").addEventListener("change", onFile);
document.querySelector(".inAlt").addEventListener("change", onFile);
$("btnBack").addEventListener("click", () => show("vHome"));
$("btnAnalyze").addEventListener("click", analyze);
$("btnNew").addEventListener("click", () => show("vHome"));
$("btnHistory").addEventListener("click", renderHistory);
$("btnHistBack").addEventListener("click", () => show("vHome"));
const askClear = (on) => { $("histConfirm").hidden = !on; $("btnHistClear").parentElement.hidden = on; };
$("btnHistClear").addEventListener("click", () => askClear(true));
$("btnHistCancel").addEventListener("click", () => askClear(false));
$("btnHistYes").addEventListener("click", () => {
  try { localStorage.removeItem(HIST_KEY); } catch {}
  askClear(false);
  renderHistory();
});
for (const b of document.querySelectorAll(".ex")) b.addEventListener("click", () => openImage(b.dataset.src).catch(fail));
setupPan();

// ---------- cámara en vivo (getUserMedia) ----------
// Solo en una web propia con HTTPS. Dentro de un marco (p. ej. claude.ai) el navegador la bloquea,
// así que "Tomar foto" abre directamente la cámara del sistema mediante el <input capture>.
// Mientras apuntas: detección del lunar (detect.js), medidor de nitidez, linterna, zoom y tocar para enfocar.
// Al capturar se usa el fotograma más nítido del último 1,5 s, no el del instante del toque.
let stream = null, track = null, caps = {}, loopTimer = null, best = null, torchOn = false;
let liveP = null, liveBusy = false, liveTick = 0, liveSeg = null;
const liveCamera = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) && window.top === window && isSecureContext;
const small = document.createElement("canvas");
small.width = small.height = 128;
const smallCtx = small.getContext("2d", { willReadFrequently: true });
const SHARP_OK = 25, BEST_MS = 1500;

const applyCam = (c) => track && track.applyConstraints({ advanced: [c] }).catch(() => {});

async function startCamera() {
  $("errBox").hidden = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
  } catch (e) {
    $("camDenied").hidden = false;
    show("vHome");
    return;
  }
  $("camDenied").hidden = true;
  track = stream.getVideoTracks()[0];
  caps = (track.getCapabilities && track.getCapabilities()) || {};
  if ((caps.focusMode || []).includes("continuous")) applyCam({ focusMode: "continuous" });
  torchOn = false;
  $("btnTorch").hidden = !caps.torch;
  $("btnTorch").setAttribute("aria-pressed", "false");
  $("camZoomRow").hidden = !caps.zoom;
  if (caps.zoom) {
    const z = $("camZoom"), start = Math.min(caps.zoom.max, Math.max(caps.zoom.min, 2));
    z.min = caps.zoom.min; z.max = caps.zoom.max; z.step = caps.zoom.step || 0.1; z.value = start;
    setCamZoom(start);
  }
  best = null; liveP = null; liveSeg = null;
  $("video").srcObject = stream;
  show("vCamera");
  $("video").play().catch(() => {});
  loop();
  loadModel().catch(fail);
}

function stopCamera() {
  clearTimeout(loopTimer);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = track = null;
  best = null;
  $("video").srcObject = null;
}

// análisis en vivo ~7 veces por segundo sobre el cuadrado central (lo que se ve en pantalla)
function loop() {
  const v = $("video");
  if (!stream) return;
  if (v.videoWidth) {
    const s = Math.min(v.videoWidth, v.videoHeight);
    smallCtx.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, 128, 128);
    const id = smallCtx.getImageData(0, 0, 128, 128);
    const det = detectLesion(id), sh = sharpness(id);
    if (segSession && det.reason !== "dark") {
      // segmentador: contorno preciso del lunar del centro (asíncrono, ~3 veces por segundo)
      if (!liveBusy && ++liveTick % 2 === 0) {
        liveBusy = true;
        const vw = v.videoWidth, vh = v.videoHeight, x0 = (vw - s) / 2, y0 = (vh - s) / 2;
        lesionAt(v, vw, vh, 0.5, 0.5, segProb, detSession && lesionProb, cfg.det_threshold, { keepAll: true, scales: [1, 0.55] })
          .then((l) => {
            if (!l) { liveSeg = { reason: "none" }; return; }
            const ext = Math.max(l.w * vw, l.h * vh) / s, dc = Math.hypot((l.cx * vw - vw / 2) / s, (l.cy * vh - vh / 2) / s);
            liveSeg = { contour: l.contour.map(([x, y]) => [(x * vw - x0) / s, (y * vh - y0) / s]),
                        reason: l.p != null && l.p < cfg.det_threshold ? "none" : dc > 0.25 ? "off_center"
                          : ext < 0.06 ? "small" : ext > 0.8 ? "big" : "ok" };
          })
          .catch(() => {}).finally(() => { liveBusy = false; });
      }
      if (liveSeg) { det.reason = liveSeg.reason; det.found = liveSeg.reason === "ok"; }
    } else if (detSession && !liveBusy && ++liveTick % 3 === 0) {
      liveBusy = true;
      lesionProb(small).then((p) => { liveP = p; }).finally(() => { liveBusy = false; });
    }
    if (!segSession) det.found = isLesion(liveP, det.found);
    const sharp = sh >= SHARP_OK;
    if (segSession) {
      const oc = $("camOverlay"), octx = oc.getContext("2d");
      octx.clearRect(0, 0, oc.width, oc.height);
      if (liveSeg && liveSeg.contour && det.reason !== "dark")
        drawContour(octx, liveSeg.contour.map(([x, y]) => [x * oc.width, y * oc.height]), det.found ? "#4ade80" : "#fbbf24", 3);
    } else drawOutline($("camOverlay"), det, det.found ? "#4ade80" : "#fbbf24", findCandidates(id, { minArea: 0.004 }));
    $("camRing").classList.toggle("ok", det.found && sharp);
    $("camStatus").textContent = !det.found ? DETECT_MSG[det.reason]
      : sharp ? "Lunar detectado y enfocado: pulsa Capturar" : "Lunar detectado: mantén el teléfono quieto para enfocar";
    $("camStatus").className = "cam-status" + (det.found && sharp ? " ok" : "");
    $("sharpFill").style.width = Math.min(100, (sh / (SHARP_OK * 3)) * 100) + "%";
    $("sharpFill").classList.toggle("ok", sharp);
    // guarda el mejor fotograma reciente (nitidez, con prioridad si hay lunar detectado)
    const score = sh * (det.found ? 2 : 1), now = performance.now();
    if (!best || now - best.t > BEST_MS || score >= best.score) {
      const k = Math.min(1, 1600 / Math.max(v.videoWidth, v.videoHeight));
      const c = (best && best.canvas) || document.createElement("canvas");
      c.width = Math.round(v.videoWidth * k); c.height = Math.round(v.videoHeight * k);
      c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
      best = { canvas: c, score, t: now };
    }
  }
  loopTimer = setTimeout(loop, 140);
}

function snap() {
  const v = $("video");
  if (!v.videoWidth) return;
  if (best && performance.now() - best.t <= BEST_MS) {
    const c = best.canvas;
    best = null;              // el canvas pasa a la pantalla de encuadre; no se reutiliza
    openFrame(c, c.width, c.height);
  } else openFrame(v, v.videoWidth, v.videoHeight);
  stopCamera();
}

function setCamZoom(z) {
  applyCam({ zoom: +z });
  $("camZoomVal").textContent = (+z).toFixed(1).replace(".0", "") + "×";
}

// tocar para enfocar: punto de interés donde el navegador lo permite; siempre muestra el indicador
$("camBox").addEventListener("click", (e) => {
  if (e.target.closest("button")) return;
  const r = $("camBox").getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const dot = $("focusDot");
  dot.style.left = x * 100 + "%"; dot.style.top = y * 100 + "%";
  dot.hidden = false; dot.classList.remove("pulse"); void dot.offsetWidth; dot.classList.add("pulse");
  setTimeout(() => (dot.hidden = true), 900);
  const modes = caps.focusMode || [];
  if (modes.includes("single-shot") || caps.pointsOfInterest) {
    applyCam({ focusMode: modes.includes("single-shot") ? "single-shot" : undefined, pointsOfInterest: [{ x, y }] });
    if (modes.includes("continuous")) setTimeout(() => applyCam({ focusMode: "continuous" }), 1500);
  }
  best = null;                // tras enfocar se descartan los fotogramas anteriores
});
$("btnTorch").addEventListener("click", () => {
  torchOn = !torchOn;
  applyCam({ torch: torchOn });
  $("btnTorch").setAttribute("aria-pressed", String(torchOn));
});
$("camZoom").addEventListener("input", (e) => setCamZoom(e.target.value));
$("candChips").addEventListener("click", (e) => {
  const b = e.target.closest(".chip");
  if (b) selectCand(+b.dataset.i);
});
$("btnCamera").addEventListener("click", (e) => {
  if (!liveCamera) return;      // sin cámara en vivo: la etiqueta abre el <input capture>
  e.preventDefault();
  startCamera();
});
$("btnSnap").addEventListener("click", snap);
$("btnCamCancel").addEventListener("click", () => { stopCamera(); show("vHome"); });

// ---------- QR con el enlace de la app (vista #qr) ----------
async function showQr() {
  let url = location.href.split("#")[0];
  // dentro de claude.ai la página corre en un marco aislado; el enlace público va en config.json
  try { url = (await (await fetch("model/config.json")).json()).share_url || url; } catch {}
  $("qrLink").value = url;
  $("qrBox").innerHTML = "";
  new QRCode($("qrBox"), { text: url, width: 240, height: 240, colorDark: "#0b1716", colorLight: "#ffffff",
                           correctLevel: QRCode.CorrectLevel.M });
  show("vQr");
}
$("btnQr").addEventListener("click", showQr);
$("btnQrBack").addEventListener("click", () => show("vHome"));
$("btnCopy").addEventListener("click", () => {
  const done = () => { $("btnCopy").textContent = "Enlace copiado"; setTimeout(() => ($("btnCopy").textContent = "Copiar enlace"), 2000); };
  navigator.clipboard.writeText($("qrLink").value).then(done, () => { $("qrLink").select(); $("btnCopy").textContent = "Selecciona y copia"; });
});
if (location.hash === "#qr") showQr();

if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
