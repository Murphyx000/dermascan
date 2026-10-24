// Cache offline: tras la primera visita la app funciona sin conexión.
const CACHE = "dermascan-v2";
const FILES = [
  "./", "index.html", "style.css", "app.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png",
  "vendor/ort.min.js", "vendor/ort-wasm-simd-threaded.wasm", "vendor/ort-wasm-simd-threaded.mjs", "vendor/qrcode.min.js",
  "model/model.onnx", "model/config.json", "examples/nev.jpg", "examples/bcc.jpg", "examples/mel.jpg",
];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((r) => r || fetch(e.request)));
});
