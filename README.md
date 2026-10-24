# DermaScan

Prototipo académico (tesis): app web que analiza la foto de un lunar con una red neuronal (MobileNetV3) que se ejecuta **en el propio navegador del teléfono**. La imagen nunca sale del dispositivo.

- Modelo entrenado con HAM10000 (dermatoscopia) y ajustado con PAD-UFES-20 (fotos de smartphone, CC BY 4.0).
- Inferencia con onnxruntime-web (WebAssembly). Funciona sin conexión después de la primera visita (PWA).

**No es un dispositivo médico ni sustituye el diagnóstico de un dermatólogo.**
