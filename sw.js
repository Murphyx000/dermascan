// Cache offline: tras la primera visita la app funciona sin conexión.
const CACHE = "dermascan-v5";
const FILES = [
  "./", "index.html", "style.css", "app.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png",
  "vendor/ort.min.js", "vendor/ort-wasm-simd-threaded.wasm", "vendor/ort-wasm-simd-threaded.mjs", "vendor/qrcode.min.js", "detect.js",
  "model/model.onnx", "model/det.onnx", "model/seg.onnx", "model/config.json", "examples/nev.jpg", "examples/bcc.jpg", "examples/mel.jpg",
];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
// primero la red (así las actualizaciones se ven al instante); sin conexión, la copia guardada
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok && new URL(e.request.url).origin === location.origin) {
      const copy = r.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
    }
    return r;
  }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
