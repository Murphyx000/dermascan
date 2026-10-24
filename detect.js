"use strict";
// Detector de presencia de lunar (visión por computador clásica, sin red neuronal).
// Comprueba que haya piel y una mancha más oscura, compacta y cercana al centro.
// 1) luminancia suavizada  2) compensación de iluminación (plano ajustado a la piel del borde) y umbral
//    adaptativo sobre el residuo  3) apertura morfológica (elimina vellos finos)
// 4) componente conexa más grande cerca del centro  5) reglas de tamaño, contraste y piel (YCbCr).
// Entrada: ImageData cuadrado pequeño (p. ej. 128x128). Salida: {found, reason, mask, ...}.

// umbrales elegidos en validación (ml/out/detector_metrics.json); maskT y selR fijan cómo se forma la máscara
const DETECT = { minArea: 0.008, maxArea: 0.7, minContrast: 22, minSkin: 0.35, centerR: 0.30, maskT: 9.8, selR: 0.36 };

function detectLesion(imgData) {
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
  if (meanL < 35) return { found: false, reason: "dark", meanL };

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
  if (!comps.length) return { found: false, reason: "none", meanL };
  const c = comps.reduce((a, b) => (b.score > a.score ? b : a));
  const sel = new Uint8Array(N);
  let inR = 0, refSkin = 0;
  for (let i = 0; i < N; i++) if (lab[i] === c.id) { sel[i] = 1; inR += Rz[i]; }
  for (const i of ref) refSkin += skin[i];
  const areaFrac = c.area / N, contrast = rs / ref.length - inR / c.area;
  const skinFrac = refSkin / ref.length, compact = (4 * Math.PI * c.area) / (c.perim * c.perim);
  let reason = "ok";
  if (skinFrac < DETECT.minSkin) reason = "no_skin";
  else if (c.score <= 0 || c.dist > Rc) reason = "off_center";
  else if (areaFrac < DETECT.minArea) reason = "small";
  else if (areaFrac > DETECT.maxArea) reason = "big";
  else if (contrast < DETECT.minContrast) reason = "low_contrast";
  return { found: reason === "ok", reason, mask: sel, areaFrac, contrast, skinFrac, compact, meanL,
           cx: c.mx / W, cy: c.my / H };
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
function drawOutline(canvas, det, color) {
  const ctx = canvas.getContext("2d"), W = 128;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
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
