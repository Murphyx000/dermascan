"use strict";
// Detector de presencia de lunar (visión por computador clásica, sin red neuronal).
// Comprueba que haya piel y una mancha más oscura, compacta y cercana al centro.
// 1) luminancia suavizada  2) compensación de iluminación (plano ajustado a la piel del borde) y umbral
//    adaptativo sobre el residuo  3) apertura morfológica (elimina vellos finos)
// 4) componente conexa más grande cerca del centro  5) reglas de tamaño, contraste y piel (YCbCr).
// Entrada: ImageData cuadrado pequeño (p. ej. 128x128). Salida: {found, reason, mask, ...}.

// umbrales elegidos en validación (ml/out/detector_metrics.json); maskT y selR fijan cómo se forma la máscara
const DETECT = { minArea: 0.008, maxArea: 0.7, minContrast: 22, minSkin: 0.35, centerR: 0.30, maskT: 9.8, selR: 0.36 };

// segmentación común: devuelve la máscara de manchas oscuras, sus componentes conexas y la piel de referencia
// opts.hyst: umbral doble con histéresis (las manchas que superan el umbral estricto se expanden mientras
// superen hyst·T), para recuperar la lesión completa y no solo su núcleo más oscuro
function segment(imgData, opts = {}) {
  const { data, width: W, height: H } = imgData, N = W * H;
  const L = new Float32Array(N), skin = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const r = data[4 * i], g = data[4 * i + 1], b = data[4 * i + 2];
    L[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b, cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    skin[i] = cb >= 77 && cb <= 135 && cr >= 130 && cr <= 180 ? 1 : 0;
  }
  // suavizado 3x3
  const S = new Float32Array(N);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W) { s += L[yy * W + xx]; n++; }
    }
    S[y * W + x] = s / n;
  }
  const meanL = S.reduce((acc, v) => acc + v, 0) / N;
  if (meanL < 35) return { dark: true, meanL };

  // modelo de iluminación: plano L = a + b·x + c·y ajustado a la piel del borde (banda exterior del 14 %).
  // Se descarta el 35 % más oscuro del borde por si el lunar lo toca.
  const band = [], bw = Math.round(0.14 * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (x < bw || y < bw || x >= W - bw || y >= H - bw) band.push(y * W + x);
  const sorted = band.map((i) => S[i]).sort((p, q) => p - q), cut = sorted[Math.floor(sorted.length * 0.35)];
  const ref = band.filter((i) => S[i] >= cut);
  // mínimos cuadrados 3x3 (ecuaciones normales) con coordenadas centradas en [-1, 1]
  let sxx = 0, syy = 0, sxy = 0, sx = 0, sy = 0, n0 = 0, sl = 0, sxl = 0, syl = 0;
  for (const i of ref) {
    const x = (i % W) / W * 2 - 1, y = ((i / W) | 0) / H * 2 - 1, l = S[i];
    sxx += x * x; syy += y * y; sxy += x * y; sx += x; sy += y; n0++; sl += l; sxl += x * l; syl += y * l;
  }
  const A = [[n0, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]], B = [sl, sxl, syl];
  const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det3(A), coef = [0, 1, 2].map((k) => det3(A.map((row, r) => row.map((v, cIdx) => (cIdx === k ? B[r] : v)))) / D);
  const Rz = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = (i % W) / W * 2 - 1, y = ((i / W) | 0) / H * 2 - 1;
    Rz[i] = S[i] - (coef[0] + coef[1] * x + coef[2] * y);
  }
  let rs = 0, rs2 = 0;
  for (const i of ref) { rs += Rz[i]; rs2 += Rz[i] * Rz[i]; }
  const sigma = Math.sqrt(Math.max(0, rs2 / ref.length - (rs / ref.length) ** 2));
  const T = Math.max(DETECT.maskT, 2.5 * sigma);
  let mask = new Uint8Array(N);
  for (let i = 0; i < N; i++) mask[i] = Rz[i] < -T ? 1 : 0;

  // apertura morfológica 3x3 x2: quita vellos y ruido fino
  const morph = (m, erode) => {
    const o = new Uint8Array(N);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = erode ? 1 : 0;
      for (let dy = -1; dy <= 1 && v === (erode ? 1 : 0); dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = Math.min(H - 1, Math.max(0, y + dy)), xx = Math.min(W - 1, Math.max(0, x + dx));
        if (erode ? !m[yy * W + xx] : m[yy * W + xx]) { v = erode ? 0 : 1; break; }
      }
      o[y * W + x] = v;
    }
    return o;
  };
  mask = morph(morph(morph(morph(mask, true), true), false), false);
  if (opts.hyst) {
    const low = new Uint8Array(N), st = [];
    for (let i = 0; i < N; i++) low[i] = Rz[i] < -T * opts.hyst ? 1 : 0;
    for (let i = 0; i < N; i++) if (mask[i]) st.push(i);
    while (st.length) {
      const p = st.pop(), x = p % W, y = (p / W) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx, yy = y + dy, q = yy * W + xx;
        if (xx >= 0 && yy >= 0 && xx < W && yy < H && low[q] && !mask[q]) { mask[q] = 1; st.push(q); }
      }
    }
    mask = morph(morph(mask, true), false);
  }

  // componentes conexas: se elige la de mayor área ponderada por cercanía al centro
  const lab = new Int32Array(N).fill(-1), comps = [], stack = new Int32Array(N);
  const cx = W / 2, cy = H / 2, R = DETECT.selR * W, Rc = DETECT.centerR * W;
  for (let s0 = 0; s0 < N; s0++) {
    if (!mask[s0] || lab[s0] >= 0) continue;
    const id = comps.length;
    let sp = 0, area = 0, sx = 0, sy = 0, border = 0, perim = 0, touchesEdge = false;
    stack[sp++] = s0; lab[s0] = id;
    while (sp) {
      const p = stack[--sp], x = p % W, y = (p / W) | 0;
      area++; sx += x; sy += y;
      if (x === 0 || y === 0 || x === W - 1 || y === H - 1) touchesEdge = true;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) { perim++; continue; }
        const q = yy * W + xx;
        if (!mask[q]) { perim++; continue; }
        if (lab[q] < 0) { lab[q] = id; stack[sp++] = q; }
      }
    }
    const mx = sx / area, my = sy / area, dist = Math.hypot(mx - cx, my - cy);
    comps.push({ id, area, mx, my, dist, perim, touchesEdge, score: area * Math.max(0, 1 - dist / (R * 1.6)) });
  }
  let refSkin = 0;
  for (const i of ref) refSkin += skin[i];
  return { W, H, N, Rz, lab, comps, refMean: rs / ref.length, skinFrac: refSkin / ref.length, meanL, Rc };
}

// ¿hay un lunar en el centro? (versión evaluada en la tesis: ml/out/detector_metrics.json)
function detectLesion(imgData) {
  const g = segment(imgData);
  if (g.dark) return { found: false, reason: "dark", meanL: g.meanL };
  const { W, H, N, Rz, lab, comps, meanL, Rc } = g;
  if (!comps.length) return { found: false, reason: "none", meanL };
  const c = comps.reduce((a, b) => (b.score > a.score ? b : a));
  const sel = new Uint8Array(N);
  let inR = 0;
  for (let i = 0; i < N; i++) if (lab[i] === c.id) { sel[i] = 1; inR += Rz[i]; }
  const areaFrac = c.area / N, contrast = g.refMean - inR / c.area;
  const skinFrac = g.skinFrac, compact = (4 * Math.PI * c.area) / (c.perim * c.perim);
  let reason = "ok";
  if (skinFrac < DETECT.minSkin) reason = "no_skin";
  else if (c.score <= 0 || c.dist > Rc) reason = "off_center";
  else if (areaFrac < DETECT.minArea) reason = "small";
  else if (areaFrac > DETECT.maxArea) reason = "big";
  else if (contrast < DETECT.minContrast) reason = "low_contrast";
  return { found: reason === "ok", reason, mask: sel, areaFrac, contrast, skinFrac, compact, meanL,
           cx: c.mx / W, cy: c.my / H };
}

// todas las manchas candidatas de la imagen (para fotos con varios lunares). Cada candidato trae su
// centro, tamaño y contorno en coordenadas normalizadas [0, 1]; luego la red neuronal confirma cuáles son lunares.
function findCandidates(imgData, opts = {}) {
  // primero con histéresis (contorno completo); las manchas que así quedan irregulares se recuperan
  // con su núcleo estricto (sin histéresis), siempre que no estén ya cubiertas por otro candidato
  const full = candidatesFrom(imgData, opts, opts.hyst ?? 0.8);
  const core = candidatesFrom(imgData, opts, 0);
  const inside = (k, c) => Math.abs(k.cx - c.cx) < c.w / 2 && Math.abs(k.cy - c.cy) < c.h / 2;
  return full.concat(core.filter((k) => !full.some((c) => inside(k, c))))
    .sort((p, q) => q.areaFrac * q.contrast - p.areaFrac * p.contrast)
    .slice(0, opts.max ?? 8);
}

function candidatesFrom(imgData, opts, hyst) {
  const minArea = opts.minArea ?? 0.0015, maxArea = opts.maxArea ?? DETECT.maxArea;
  const minContrast = opts.minContrast ?? DETECT.minContrast * 0.8;
  const g = segment(imgData, { hyst });
  if (g.dark || g.skinFrac < DETECT.minSkin) return [];
  const { W, H, N, Rz, lab } = g;
  const acc = g.comps.map(() => ({ sum: 0, x0: W, y0: H, x1: 0, y1: 0, edge: [] }));
  for (let i = 0; i < N; i++) {
    const id = lab[i];
    if (id < 0) continue;
    const a = acc[id], x = i % W, y = (i / W) | 0;
    a.sum += Rz[i];
    if (x < a.x0) a.x0 = x; if (x > a.x1) a.x1 = x; if (y < a.y0) a.y0 = y; if (y > a.y1) a.y1 = y;
    if (x === 0 || y === 0 || x === W - 1 || y === H - 1 || lab[i - 1] !== id || lab[i + 1] !== id || lab[i - W] !== id || lab[i + W] !== id)
      a.edge.push(x / W, y / H);
  }
  return g.comps.map((c, k) => ({ c, a: acc[k] }))
    .map(({ c, a }) => ({ cx: c.mx / W, cy: c.my / H, areaFrac: c.area / N, contrast: g.refMean - a.sum / c.area,
                          w: (a.x1 - a.x0 + 1) / W, h: (a.y1 - a.y0 + 1) / H, edge: a.edge, px: 1 / W,
                          touchesEdge: c.touchesEdge, compact: (4 * Math.PI * c.area) / (c.perim * c.perim) }))
    // descarta lo que toca el borde (sombras, o un lunar cortado que no se puede analizar) y las formas tipo telaraña
    .filter((k) => k.areaFrac >= minArea && k.areaFrac <= maxArea && k.contrast >= minContrast &&
                   !k.touchesEdge && k.compact >= (opts.minCompact ?? 0.1));
}

// localiza los lunares de una foto: propuestas (findCandidates) confirmadas por la red (probFn sobre un
// recorte de 128 px) y, además, el recorte central, porque el usuario encuadra el lunar en el centro.
// Si el centro es un lunar y ninguna propuesta lo cubre, se añade con el contorno de detectLesion.
async function locateLesions(source, w, h, probFn, thr) {
  const k = Math.min(1, 320 / Math.max(w, h)), t = document.createElement("canvas");
  t.width = Math.round(w * k); t.height = Math.round(h * k);
  const tx = t.getContext("2d", { willReadFrequently: true });
  tx.drawImage(source, 0, 0, t.width, t.height);
  const c128 = document.createElement("canvas");
  c128.width = c128.height = 128;
  const x128 = c128.getContext("2d", { willReadFrequently: true });
  const out = [];
  for (const c of findCandidates(tx.getImageData(0, 0, t.width, t.height))) {
    const side = Math.max(48, Math.max(c.w * w, c.h * h) * 2);
    const x0 = Math.min(Math.max(c.cx * w - side / 2, 0), Math.max(0, w - side));
    const y0 = Math.min(Math.max(c.cy * h - side / 2, 0), Math.max(0, h - side));
    x128.drawImage(source, x0, y0, side, side, 0, 0, 128, 128);
    c.p = await probFn(c128);
    if (c.p >= thr) out.push(c);
  }
  // recorte central (el cuadrado que analiza el clasificador)
  const S = Math.min(w, h), ox = (w - S) / 2, oy = (h - S) / 2;
  x128.drawImage(source, ox, oy, S, S, 0, 0, 128, 128);
  const pc = await probFn(c128);
  if (pc >= thr && !out.some((c) => Math.hypot(c.cx - 0.5, c.cy - 0.5) < 0.2)) {
    const d = detectLesion(x128.getImageData(0, 0, 128, 128)), edge = [];
    let x0 = 128, y0 = 128, x1 = 0, y1 = 0;
    if (d.mask) {
      const m = d.mask;
      for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) {
        const i = y * 128 + x;
        if (!m[i]) continue;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        if (x === 0 || y === 0 || x === 127 || y === 127 || !m[i - 1] || !m[i + 1] || !m[i - 128] || !m[i + 128])
          edge.push((ox + (x / 128) * S) / w, (oy + (y / 128) * S) / h);
      }
    }
    if (!edge.length) {             // sin máscara fiable: círculo aproximado en el centro
      for (let a = 0; a < 2 * Math.PI; a += 0.05) edge.push(0.5 + (0.25 * S * Math.cos(a)) / w, 0.5 + (0.25 * S * Math.sin(a)) / h);
      x0 = 32; x1 = 96; y0 = 32; y1 = 96;
    }
    out.push({ cx: (ox + ((x0 + x1) / 2 / 128) * S) / w, cy: (oy + ((y0 + y1) / 2 / 128) * S) / h,
               w: ((x1 - x0 + 1) / 128) * S / w, h: ((y1 - y0 + 1) / 128) * S / h, edge, px: S / 128 / Math.max(w, h),
               p: pc, central: true });
  }
  // fusiona trozos del mismo lunar: si el centro de una detección cae dentro de otra mayor, sobra
  const area = (c) => c.w * c.h;
  const merged = out.sort((a, b) => area(b) - area(a)).filter((c, i, arr) =>
    !arr.slice(0, i).some((o) => Math.abs(c.cx - o.cx) < o.w * 0.6 && Math.abs(c.cy - o.cy) < o.h * 0.6));
  return merged.sort((a, b) => Math.hypot(a.cx - 0.5, a.cy - 0.5) - Math.hypot(b.cx - 0.5, b.cy - 0.5));
}

// nitidez: varianza del laplaciano sobre la luminancia (valores bajos = desenfocado)
function sharpness(imgData) {
  const { data, width: W, height: H } = imgData, g = new Float32Array(W * H);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2];
  let s1 = 0, s2 = 0, n = 0;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x, l = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - W] - g[i + W];
    s1 += l; s2 += l * l; n++;
  }
  return s2 / n - (s1 / n) ** 2;
}

// dibuja el contorno de la máscara en un canvas de overlay
function drawOutline(canvas, det, color, others) {
  const ctx = canvas.getContext("2d"), W = 128;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // otras manchas candidatas, en blanco y más finas
  if (others) {
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    const k = canvas.width;
    for (const c of others) {
      if (det && det.found && Math.hypot(c.cx - det.cx, c.cy - det.cy) < 0.04) continue;
      for (let j = 0; j < c.edge.length; j += 2) ctx.fillRect(c.edge[j] * k - 1.5, c.edge[j + 1] * k - 1.5, 3, 3);
    }
  }
  if (!det || !det.mask) return;
  const k = canvas.width / W, m = det.mask;
  ctx.fillStyle = color;
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (m[i] && (x === 0 || y === 0 || x === W - 1 || y === W - 1 || !m[i - 1] || !m[i + 1] || !m[i - W] || !m[i + W]))
      ctx.fillRect(x * k - k / 2, y * k - k / 2, k * 1.6, k * 1.6);
  }
}

const DETECT_MSG = {
  ok: "Lunar detectado",
  dark: "Poca luz: activa la linterna o busca más luz",
  none: "No veo ningún lunar en el círculo",
  no_skin: "No se reconoce piel: apunta a la zona del lunar",
  off_center: "Centra el lunar dentro del círculo",
  small: "El lunar se ve muy pequeño: usa el zoom o acércate un poco",
  big: "Demasiado cerca o mancha muy grande: aléjate un poco",
  low_contrast: "No distingo bien el lunar: mejora la luz o el enfoque",
};
