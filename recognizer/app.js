/**
 * Linzklar handwriting recognizer — ONNX Runtime Web + FP16 ConvNeXt-Tiny.
 *
 * Model/meta are served from ./recognizer/models/ (commit for GitHub Pages).
 * Refresh model after re-export:
 *   cp char-convnext/outputs/onnx/convnext_tiny_linzklar_fp16.onnx recognizer/models/
 *   cp char-convnext/outputs/onnx/model_meta_fp16.json recognizer/models/model_meta.json
 */

const MODEL_URL = "./recognizer/models/convnext_tiny_linzklar_fp16.onnx";
const META_URL = "./recognizer/models/model_meta.json";
const ORT_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const DEFAULT_MEAN = [0.485, 0.456, 0.406];
const DEFAULT_STD = [0.229, 0.224, 0.225];
const TOP_K = 5;

/** @type {import('onnxruntime-web').InferenceSession | null} */
let session = null;
/** @type {string[]} index -> class label */
let idxToClass = [];
let imageSize = 128;
let mean = DEFAULT_MEAN;
let std = DEFAULT_STD;
let inputName = "input";
let outputName = "logits";

let drawing = false;
let lastX = 0;
let lastY = 0;
let strokeWidth = 6;
let hasInk = false;

const el = {
  status: document.getElementById("status"),
  statusText: document.getElementById("status-text"),
  metaBits: document.getElementById("meta-bits"),
  canvas: /** @type {HTMLCanvasElement} */ (document.getElementById("draw")),
  btnClear: document.getElementById("btn-clear"),
  btnRecognize: document.getElementById("btn-recognize"),
  strokeRange: /** @type {HTMLInputElement} */ (document.getElementById("stroke-width")),
  results: document.getElementById("results-list"),
  resultsEmpty: document.getElementById("results-empty"),
  latency: document.getElementById("latency"),
};

const ctx = el.canvas.getContext("2d", { willReadFrequently: true });

function setStatus(state, text) {
  el.status.dataset.state = state;
  el.statusText.textContent = text;
}

function clearCanvas() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, el.canvas.width, el.canvas.height);
  ctx.restore();
  hasInk = false;
  el.results.innerHTML = "";
  el.resultsEmpty.hidden = false;
  el.latency.textContent = "";
}

function canvasPoint(evt) {
  const rect = el.canvas.getBoundingClientRect();
  const scaleX = el.canvas.width / rect.width;
  const scaleY = el.canvas.height / rect.height;
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top) * scaleY,
  };
}

function startDraw(evt) {
  if (!session) return;
  evt.preventDefault();
  drawing = true;
  const p = canvasPoint(evt);
  lastX = p.x;
  lastY = p.y;
  ctx.strokeStyle = "#000000";
  ctx.fillStyle = "#000000";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = strokeWidth;
  ctx.beginPath();
  ctx.arc(p.x, p.y, strokeWidth / 2, 0, Math.PI * 2);
  ctx.fill();
  hasInk = true;
}

function moveDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const p = canvasPoint(evt);
  ctx.strokeStyle = "#000000";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = strokeWidth;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x;
  lastY = p.y;
  hasInk = true;
}

function endDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  drawing = false;
}

/**
 * Resize draw canvas to model size and build NCHW float32 tensor (ImageNet norm).
 * @returns {Float32Array}
 */
function canvasToTensor() {
  const off = document.createElement("canvas");
  off.width = imageSize;
  off.height = imageSize;
  const octx = off.getContext("2d", { willReadFrequently: true });
  octx.fillStyle = "#ffffff";
  octx.fillRect(0, 0, imageSize, imageSize);
  octx.drawImage(el.canvas, 0, 0, imageSize, imageSize);

  const { data } = octx.getImageData(0, 0, imageSize, imageSize);
  const n = imageSize * imageSize;
  const out = new Float32Array(3 * n);

  for (let i = 0; i < n; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    out[i] = (r - mean[0]) / std[0];
    out[n + i] = (g - mean[1]) / std[1];
    out[2 * n + i] = (b - mean[2]) / std[2];
  }
  return out;
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    exps[i] = v;
    sum += v;
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

function topK(probs, k) {
  const idx = Array.from(probs.keys());
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, k).map((i) => ({
    index: i,
    label: idxToClass[i] ?? String(i),
    prob: probs[i],
  }));
}

function renderResults(items, ms) {
  el.resultsEmpty.hidden = true;
  el.results.innerHTML = "";
  items.forEach((item, rank) => {
    const li = document.createElement("li");
    // Escape is unnecessary for model labels (CJK single chars), but keep text via textContent below.
    const label = document.createElement("span");
    label.className = "label";

    const linz = document.createElement("span");
    linz.className = "label-linzklar";
    linz.textContent = item.label;

    const open = document.createElement("span");
    open.className = "label-brackets";
    open.textContent = "【";

    const han = document.createElement("span");
    han.className = "label-han";
    han.textContent = item.label;

    const close = document.createElement("span");
    close.className = "label-brackets";
    close.textContent = "】";

    label.append(linz, open, han, close);

    const rankEl = document.createElement("span");
    rankEl.className = "rank";
    rankEl.textContent = String(rank + 1);

    const barWrap = document.createElement("div");
    barWrap.className = "bar-wrap";
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = `${(item.prob * 100).toFixed(1)}%`;
    barWrap.appendChild(bar);

    const prob = document.createElement("span");
    prob.className = "prob";
    prob.textContent = `${(item.prob * 100).toFixed(1)}%`;

    li.append(rankEl, label, barWrap, prob);
    el.results.appendChild(li);
  });
  el.latency.textContent = ms != null ? `Inference: ${ms.toFixed(0)} ms` : "";
}

async function recognize() {
  if (!session) return;
  if (!hasInk) {
    setStatus("ready", "Ready — draw a character first");
    return;
  }

  el.btnRecognize.disabled = true;
  try {
    const tensorData = canvasToTensor();
    const tensor = new ort.Tensor("float32", tensorData, [1, 3, imageSize, imageSize]);
    const t0 = performance.now();
    const feeds = { [inputName]: tensor };
    const out = await session.run(feeds);
    const ms = performance.now() - t0;
    const logits = out[outputName].data;
    const probs = softmax(logits);
    const items = topK(probs, TOP_K);
    renderResults(items, ms);
    setStatus("ready", "Ready");
  } catch (err) {
    console.error(err);
    setStatus("error", `Inference failed: ${err.message || err}`);
  } finally {
    el.btnRecognize.disabled = false;
  }
}

async function loadMeta() {
  const res = await fetch(META_URL);
  if (!res.ok) throw new Error(`Failed to load meta (${res.status}): ${META_URL}`);
  const meta = await res.json();
  imageSize = meta.image_size || 128;
  mean = meta.mean || DEFAULT_MEAN;
  std = meta.std || DEFAULT_STD;
  inputName = meta.input_name || "input";
  outputName = meta.output_name || "logits";

  const classToIdx = meta.class_to_idx || {};
  const n = meta.num_classes || Object.keys(classToIdx).length;
  idxToClass = new Array(n);
  for (const [name, idx] of Object.entries(classToIdx)) {
    idxToClass[idx] = name;
  }

  el.metaBits.textContent = `${n} classes · ${imageSize}×${imageSize} · FP16 ONNX`;
  return meta;
}

async function loadSession() {
  setStatus("loading", "Loading ONNX Runtime…");
  ort.env.wasm.wasmPaths = ORT_WASM_PATH;

  setStatus("loading", "Downloading model (~54 MB)…");
  const providers = [];
  // Prefer WebGPU when available; WASM always as fallback.
  if (typeof navigator !== "undefined" && navigator.gpu) {
    providers.push("webgpu");
  }
  providers.push("wasm");

  const t0 = performance.now();
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: providers,
    });
  } catch (err) {
    // WebGPU can fail on some devices; fall back to WASM only.
    if (providers.includes("webgpu")) {
      console.warn("WebGPU session failed, retrying with wasm only", err);
      session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
      });
    } else {
      throw err;
    }
  }
  const ms = performance.now() - t0;
  setStatus("ready", `Ready (loaded in ${(ms / 1000).toFixed(1)}s)`);
  el.metaBits.textContent += ` · tried EP: ${providers.join(", ")}`;
  el.btnRecognize.disabled = false;
  el.btnClear.disabled = false;
  console.log("ORT session ready", { triedProviders: providers, loadMs: ms });
}

function bindUi() {
  clearCanvas();

  el.canvas.addEventListener("pointerdown", startDraw);
  el.canvas.addEventListener("pointermove", moveDraw);
  el.canvas.addEventListener("pointerup", endDraw);
  el.canvas.addEventListener("pointerleave", endDraw);
  el.canvas.addEventListener("pointercancel", endDraw);

  el.btnClear.addEventListener("click", () => {
    clearCanvas();
    if (session) setStatus("ready", "Ready");
  });
  el.btnRecognize.addEventListener("click", () => recognize());

  el.strokeRange.addEventListener("input", () => {
    strokeWidth = Number(el.strokeRange.value);
  });
  strokeWidth = Number(el.strokeRange.value);
}

async function main() {
  bindUi();
  el.btnRecognize.disabled = true;
  el.btnClear.disabled = true;

  try {
    if (typeof ort === "undefined") {
      throw new Error("onnxruntime-web failed to load from CDN");
    }
    await loadMeta();
    await loadSession();
  } catch (err) {
    console.error(err);
    setStatus("error", err.message || String(err));
  }
}

main();
