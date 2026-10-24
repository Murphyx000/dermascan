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
  busy(null);
  return session;
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
  show("vCrop");
  draw();
  loadModel().catch(fail);
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
  checkQuality();
}

function setupPan() {
  const box = $("cropBox"), pts = new Map();
  let last = null, pinch = null;
  const ratio = () => $("cropCanvas").width / box.clientWidth;
  box.addEventListener("pointerdown", (e) => { box.setPointerCapture(e.pointerId); pts.set(e.pointerId, e); last = e; });
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
  const up = (e) => { pts.delete(e.pointerId); pinch = null; last = [...pts.values()][0] || null; };
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
function toTensor(canvas, flip) {
  const { data } = canvas.getContext("2d").getImageData(0, 0, SIZE, SIZE);
  const t = new Float32Array(3 * SIZE * SIZE), plane = SIZE * SIZE;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const src = 4 * (y * SIZE + (flip ? SIZE - 1 - x : x)), dst = y * SIZE + x;
    for (let c = 0; c < 3; c++) t[c * plane + dst] = (data[src + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor("float32", t, [1, 3, SIZE, SIZE]);
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
    const thumb = cropped(112).toDataURL("image/jpeg", 0.8);
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
let stream = null;
const liveCamera = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) && window.top === window && isSecureContext;

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
  $("video").srcObject = stream;
  show("vCamera");
  $("video").play().catch(() => {});
  loadModel().catch(fail);
}
function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  $("video").srcObject = null;
}
function snap() {
  const v = $("video");
  if (!v.videoWidth) return;
  openFrame(v, v.videoWidth, v.videoHeight);
  stopCamera();
}
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
