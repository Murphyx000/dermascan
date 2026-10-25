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

// ================= segmentador neuronal (ml/train_seg.py) =================
// segFn(canvas SEG×SEG) -> Float32Array con la probabilidad de lesión por píxel.
// La red da el contorno preciso; el detector "lunar / no lunar" (probFn) confirma cada lesión.
const SEG = 192;
// thr: núcleo de la lesión; low: el borde se extiende mientras la probabilidad supere low (histéresis, recupera
// el borde de lunares de color parecido a la piel); stretch: realce de contraste (si el modelo se entrenó con él)
const SEGP = { thr: 0.5, low: 0.35, minArea: 0.004, maxArea: 0.6, target: 0.5, stretch: false };

// realce de contraste por canal: percentiles 1 y 99 (sin el negro de relleno) a 0-255, ganancia máxima 3.
// Idéntico a stretch() de ml/train_seg.py
function stretchContrast(imgData) {
  const d = imgData.data, n = d.length / 4, valid = new Uint8Array(n);
  let nv = 0;
  for (let i = 0; i < n; i++) if (d[4 * i] + d[4 * i + 1] + d[4 * i + 2] > 15) { valid[i] = 1; nv++; }
  if (nv < 100) return imgData;
  const lut = [];
  for (let c = 0; c < 3; c++) {
    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) if (valid[i]) hist[d[4 * i + c]]++;
    const at = (rank) => { let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > rank) return v; } return 255; };
    const lo = at(Math.floor(0.01 * (nv - 1))), hi = at(Math.floor(0.99 * (nv - 1)));
    const g = Math.min(3, 255 / Math.max(hi - lo, 1)), mid = (lo + hi) / 2;
    lut.push(Array.from({ length: 256 }, (_, v) => Math.floor(Math.min(255, Math.max(0, (v - mid) * g + 127.5)))));
  }
  for (let i = 0; i < n; i++) {
    if (!valid[i]) { d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = 0; continue; }
    for (let c = 0; c < 3; c++) d[4 * i + c] = lut[c][d[4 * i + c]];
  }
  return imgData;
}

// contorno exterior de la componente `id` (seguimiento de borde de Moore), en píxeles de la máscara
function traceContour(lab, W, H, id) {
  let s = -1;
  for (let i = 0; i < W * H; i++) if (lab[i] === id) { s = i; break; }
  if (s < 0) return [];
  const DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, 1, 1, 1, 0, -1, -1, -1];   // E, SE, S, SO, O, NO, N, NE
  const inside = (x, y) => x >= 0 && y >= 0 && x < W && y < H && lab[y * W + x] === id;
  const sx = s % W, sy = (s / W) | 0, pts = [[sx, sy]];
  let x = sx, y = sy, back = 4, first = -1;
  for (let it = 0; it < 4 * W * H; it++) {
    let moved = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8;
      if (inside(x + DX[d], y + DY[d])) { moved = d; break; }
    }
    if (moved < 0) break;                              // píxel aislado
    if (x === sx && y === sy) {
      if (first < 0) first = moved; else if (moved === first) break;
    }
    x += DX[moved]; y += DY[moved];
    back = (moved + 5) % 8;                            // vecino vacío previo, visto desde el nuevo píxel
    pts.push([x, y]);
  }
  if (pts.length > 1) pts.pop();
  return pts;
}

// suaviza el contorno (media móvil circular) para que no se vea escalonado
function smoothContour(pts, r = 2) {
  const n = pts.length;
  if (n < 2 * r + 3) return pts;
  return pts.map((_, i) => {
    let x = 0, y = 0;
    for (let k = -r; k <= r; k++) { const p = pts[(i + k + n) % n]; x += p[0]; y += p[1]; }
    return [x / (2 * r + 1), y / (2 * r + 1)];
  });
}

// segmenta un recorte cuadrado de la imagen (x0, y0, lado en píxeles de la imagen; lo que sobresale queda negro)
// y devuelve sus componentes, con contorno en coordenadas normalizadas de la imagen
async function segmentRegion(source, w, h, x0, y0, side, segFn, thr = SEGP.thr) {
  const c = document.createElement("canvas");
  c.width = c.height = SEG;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, SEG, SEG);
  ctx.imageSmoothingQuality = "high";
  const k = SEG / side;
  ctx.drawImage(source, -x0 * k, -y0 * k, w * k, h * k);
  if (SEGP.stretch) ctx.putImageData(stretchContrast(ctx.getImageData(0, 0, SEG, SEG)), 0, 0);
  const prob = await segFn(c), N = SEG * SEG, low = Math.min(thr, SEGP.low);
  const lab = new Int32Array(N).fill(-1), stack = new Int32Array(N), comps = [];
  for (let s0 = 0; s0 < N; s0++) {
    if (prob[s0] < thr || lab[s0] >= 0) continue;
    const id = comps.length;
    let sp = 0, area = 0, psum = 0, sx = 0, sy = 0, bx0 = SEG, by0 = SEG, bx1 = 0, by1 = 0, edge = false;
    stack[sp++] = s0; lab[s0] = id;
    while (sp) {
      const p = stack[--sp], x = p % SEG, y = (p / SEG) | 0;
      area++; psum += prob[p]; sx += x; sy += y;
      if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y;
      if (x === 0 || y === 0 || x === SEG - 1 || y === SEG - 1) edge = true;
      for (const q of [p - 1, p + 1, p - SEG, p + SEG]) {
        if (q < 0 || q >= N || (q % SEG === 0 && p % SEG === SEG - 1) || (p % SEG === 0 && q % SEG === SEG - 1)) continue;
        if (lab[q] < 0 && prob[q] >= low) { lab[q] = id; stack[sp++] = q; }
      }
    }
    comps.push({ id, area, areaFrac: area / N, meanP: psum / area, mx: sx / area, my: sy / area,
                 bx0, by0, bx1, by1, touchesEdge: edge });
  }
  const toImg = (px, py) => [(x0 + (px + 0.5) / k) / w, (y0 + (py + 0.5) / k) / h];
  for (const cp of comps) {
    const poly = smoothContour(traceContour(lab, SEG, SEG, cp.id));
    let per = 0;
    poly.forEach(([x1, y1], i) => { const [x2, y2] = poly[(i + 1) % poly.length]; per += Math.hypot(x2 - x1, y2 - y1); });
    cp.compact = per ? Math.min(1, (4 * Math.PI * cp.area) / (per * per)) : 0;   // 1 = círculo; bajo = forma ramificada
    cp.contour = poly.map(([px, py]) => toImg(px, py));
    [cp.cx, cp.cy] = toImg((cp.bx0 + cp.bx1) / 2, (cp.by0 + cp.by1) / 2);
    cp.w = (cp.bx1 - cp.bx0 + 1) / k / w; cp.h = (cp.by1 - cp.by0 + 1) / k / h;
    cp.ext = Math.max(cp.bx1 - cp.bx0 + 1, cp.by1 - cp.by0 + 1) / SEG;    // tamaño relativo al recorte
    cp.distC = Math.hypot(cp.mx - SEG / 2, cp.my - SEG / 2) / SEG;
    cp.hitC = lab[(SEG / 2) * SEG + SEG / 2] === cp.id;
  }
  return comps;
}

// la lesión que hay en el punto (nx, ny) de la imagen: prueba varias escalas, se queda con la que encuadra
// la lesión completa a un tamaño parecido al del entrenamiento y la vuelve a segmentar centrada (borde fino)
async function lesionAt(source, w, h, nx, ny, segFn, probFn, thr, opts = {}) {
  const m = Math.min(w, h), px = nx * w, py = ny * h;
  const pick = (comps, maxD) => comps.filter((c) => !c.touchesEdge && c.areaFrac >= SEGP.minArea && c.areaFrac <= SEGP.maxArea &&
                                                  (c.hitC || c.distC <= maxD))
    .sort((a, b) => (b.hitC - a.hitC) || (a.distC - b.distC))[0];
  let best = null;
  const st = opts.segThr || SEGP.thr;
  // escalas pequeñas: en una foto a 15-20 cm un lunar ocupa ~2-5 % del lado y la red solo traza bien el borde si
  // la lesión ocupa al menos ~15 % del recorte (con menos del 10 % falla casi siempre)
  for (const f of opts.scales || [0.07, 0.12, 0.2, 0.35, 0.6, 1]) {
    const side = f * m;
    // tolerancia al centrado del usuario (~5 % del lado de la foto), mayor en proporción en los recortes pequeños
    const c = pick(await segmentRegion(source, w, h, px - side / 2, py - side / 2, side, segFn, st), Math.min(0.4, Math.max(0.12, 0.05 / f)));
    // escala preferida: lesión de tamaño parecido al del entrenamiento y con forma compacta (a escalas grandes la
    // red puede fundir dos lunares vecinos o un pelo en una sola mancha ramificada)
    // (y seguro: en recortes muy pequeños un poro o un pelo puede dar una mancha débil del tamaño "ideal")
    const cost = (x) => Math.abs(x.ext - SEGP.target) + (x.compact < 0.45 ? 0.6 : 0) + Math.max(0, 0.8 - x.meanP);
    if (c && (!best || cost(c) < cost(best.c))) best = { c, side };
  }
  // lunar tenue (color parecido a la piel): la red da probabilidades más bajas; segundo intento más sensible
  if (!best && opts.retryLow && st > 0.3) return lesionAt(source, w, h, nx, ny, segFn, probFn, thr, { ...opts, segThr: 0.3, retryLow: false });
  if (!best) return null;
  // refinado: recorte centrado en la lesión con la lesión ocupando ~la mitad
  const side = Math.max(32, (best.c.ext * best.side) / SEGP.target), cx = best.c.cx * w, cy = best.c.cy * h;
  const r = pick(await segmentRegion(source, w, h, cx - side / 2, cy - side / 2, side, segFn, st), 0.2) || best.c;
  const lesion = { cx: r.cx, cy: r.cy, w: r.w, h: r.h, contour: r.contour, meanP: r.meanP };
  if (probFn) {
    const ps = Math.max(48, Math.max(r.w * w, r.h * h) * 2), t = document.createElement("canvas");
    t.width = t.height = 128;
    const tx = t.getContext("2d");
    tx.fillStyle = "#000"; tx.fillRect(0, 0, 128, 128);
    tx.drawImage(source, r.cx * w - ps / 2, r.cy * h - ps / 2, ps, ps, 0, 0, 128, 128);
    lesion.p = await probFn(t);
    if (lesion.p != null && lesion.p < thr * (opts.thrScale ?? 1) && !opts.keepAll) return null;
  }
  return lesion;
}

// todos los lunares de la foto: primero el del centro (el usuario lo encuadra ahí) y luego un barrido por
// ventanas solapadas; cada mancha se refina y la confirma el detector. onUpdate(lista) permite ir pintando.
async function locateLesionsSeg(source, w, h, segFn, probFn, thr, onUpdate) {
  const out = [];
  const dup = (a) => out.some((o) => Math.abs(a.cx - o.cx) < Math.max(o.w, a.w) * 0.5 && Math.abs(a.cy - o.cy) < Math.max(o.h, a.h) * 0.5);
  // el del centro lo encuadró el usuario: búsqueda más sensible y confirmación más permisiva (lunares tenues)
  const center = await lesionAt(source, w, h, 0.5, 0.5, segFn, probFn, thr, { retryLow: true, thrScale: 0.5 });
  if (center) { center.central = true; out.push(center); onUpdate && onUpdate(out.slice()); }
  const m = Math.min(w, h), side = 0.5 * m, step = 0.25 * m;
  const props = [];
  for (let y0 = 0; y0 + side <= h + 1; y0 += step) for (let x0 = 0; x0 + side <= w + 1; x0 += step) {
    for (const c of await segmentRegion(source, w, h, x0, y0, side, segFn))
      if (!c.touchesEdge && c.areaFrac >= 0.0015 && c.meanP >= 0.6) props.push(c);
  }
  props.sort((a, b) => b.area * b.meanP - a.area * a.meanP);
  // más propuestas: las manchas oscuras del detector clásico (encuentra bien varios lunares pequeños juntos);
  // el segmentador les pone el contorno y el detector neuronal decide si son lunares
  const k = Math.min(1, 320 / Math.max(w, h)), t = document.createElement("canvas");
  t.width = Math.round(w * k); t.height = Math.round(h * k);
  const tx = t.getContext("2d", { willReadFrequently: true });
  tx.drawImage(source, 0, 0, t.width, t.height);
  const classic = findCandidates(tx.getImageData(0, 0, t.width, t.height), { max: 10 });
  for (const c of props.slice(0, 12).concat(classic)) {
    if (dup(c)) continue;
    const ext = Math.max(c.w * w, c.h * h) / m;
    const l = await lesionAt(source, w, h, c.cx, c.cy, segFn, probFn, thr,
                             { scales: [Math.min(1, Math.max(0.08, ext / 0.3)), Math.min(1, Math.max(0.12, ext / 0.6))],
                               thrScale: 1.4 });   // fuera del centro se exige más certeza (evita manchas claras de piel)
    if (l && !dup(l)) { out.push(l); onUpdate && onUpdate(out.slice()); }
    if (out.length >= 10) break;
  }
  return out;
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

// dibuja un contorno (lista de puntos en píxeles del lienzo) con un halo oscuro para que se vea en cualquier piel
function drawContour(ctx, pts, color, width = 3) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = width + 2.5; ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  ctx.restore();
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
