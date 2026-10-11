"use strict";
async function sha256(value) {
  const bytes =
    typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}
const C = BarnacleCore,
  $ = (id) => document.getElementById(id),
  canvas = $("canvas"),
  ctx = canvas.getContext("2d");
let doc = null,
  state = null,
  image = null,
  mode = "remove",
  view = { x: 0, y: 0, scale: 1 },
  history = [],
  tile = -1,
  tiles = [],
  pendingCorners = [],
  drag = null,
  loadedToken = 0,
  sampleKey = "unseen";
const help = {
  remove:
    "Click a proposal to remove it. Enable removed outlines to restore one.",
  add: "Click the center of a missed barnacle inside the frame. Click an added point to remove it.",
  pan: "Drag to move around the photograph. Scroll to zoom at the pointer.",
  frame:
    "Click the four inner frame corners in perimeter order. The old frame stays active until all four are valid.",
};
function status(text) {
  $("status").textContent = text;
}
function remember() {
  if (!state) return;
  try {
    localStorage.setItem(
      "barnacle-review-v1:" + sampleKey,
      JSON.stringify(state),
    );
  } catch {
    status(
      "Browser storage unavailable; export your corrections before leaving.",
    );
  }
}
function mutate(action, details, fn) {
  history.push(JSON.parse(JSON.stringify(state)));
  fn();
  state.events.push({ action, details, time: new Date().toISOString() });
  remember();
  render();
}
function resize() {
  const r = canvas.getBoundingClientRect(),
    dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function bounds(polygon) {
  return {
    x: Math.min(...polygon.map((p) => p[0])),
    y: Math.min(...polygon.map((p) => p[1])),
    r: Math.max(...polygon.map((p) => p[0])),
    b: Math.max(...polygon.map((p) => p[1])),
  };
}
function fit(box) {
  const r = canvas.getBoundingClientRect();
  view.scale = Math.min(
    (r.width - 50) / (box.r - box.x),
    (r.height - 70) / (box.b - box.y),
  );
  view.x = (r.width - (box.r - box.x) * view.scale) / 2 - box.x * view.scale;
  view.y = (r.height - (box.b - box.y) * view.scale) / 2 - box.y * view.scale;
  draw();
}
function fitFrame() {
  if (!state) return;
  tile = -1;
  fit(bounds(state.polygon));
  $("tile-position").textContent = "Overview";
}
function path(polygon) {
  ctx.beginPath();
  polygon.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.closePath();
}
function draw() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  if (!image || !state) return;
  ctx.save();
  ctx.translate(view.x, view.y);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(image, 0, 0, doc.width, doc.height);
  // Dim the photograph outside the active scientific region.
  ctx.beginPath();
  ctx.rect(0, 0, doc.width, doc.height);
  const p = state.polygon;
  ctx.moveTo(...p[0]);
  for (let i = 1; i < p.length; i++) ctx.lineTo(...p[i]);
  ctx.closePath();
  ctx.fillStyle = "rgba(3,18,17,.6)";
  ctx.fill("evenodd");
  const removed = new Set(state.removed),
    line = 1.15 / view.scale;
  if ($("show-masks").checked)
    for (const inst of doc.instances) {
      if (inst.score < state.threshold || !C.inside(state.polygon, inst.center))
        continue;
      const deleted = removed.has(inst.id);
      if (deleted && !$("show-removed").checked) continue;
      ctx.strokeStyle = deleted ? "#ff8767" : "#3dd2ac";
      ctx.lineWidth = line;
      ctx.fillStyle = deleted
        ? "rgba(255,135,103,.10)"
        : "rgba(61,210,172,.11)";
      ctx.setLineDash(deleted ? [4 / view.scale, 3 / view.scale] : []);
      for (const polygon of inst.polygons) {
        path(polygon);
        ctx.fill();
        ctx.stroke();
      }
    }
  ctx.setLineDash([]);
  ctx.strokeStyle = "#f3ca79";
  ctx.lineWidth = 2 / view.scale;
  path(state.polygon);
  ctx.stroke();
  for (const p of C.manualInside(state)) {
    ctx.strokeStyle = "#ff8767";
    ctx.fillStyle = "rgba(255,135,103,.2)";
    ctx.lineWidth = 1.5 / view.scale;
    ctx.beginPath();
    ctx.arc(...p.center, 5 / view.scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p.center[0] - 8 / view.scale, p.center[1]);
    ctx.lineTo(p.center[0] + 8 / view.scale, p.center[1]);
    ctx.moveTo(p.center[0], p.center[1] - 8 / view.scale);
    ctx.lineTo(p.center[0], p.center[1] + 8 / view.scale);
    ctx.stroke();
  }
  for (let i = 0; i < pendingCorners.length; i++) {
    ctx.fillStyle = "#f3ca79";
    ctx.beginPath();
    ctx.arc(...pendingCorners[i], 5 / view.scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = `${12 / view.scale}px sans-serif`;
    ctx.fillText(
      String(i + 1),
      pendingCorners[i][0] + 8 / view.scale,
      pendingCorners[i][1],
    );
  }
  ctx.restore();
  $("zoom-label").textContent = Math.round(view.scale * 100) + "%";
}
function render() {
  if (!doc || !state) return;
  const n = C.counts(doc, state);
  $("final-count").textContent = n.total.toLocaleString();
  $("proposal-count").textContent = n.proposals.toLocaleString();
  $("removed-count").textContent = n.removed;
  $("manual-count").textContent = n.manual;
  $("threshold").value = state.threshold;
  $("threshold-value").textContent = state.threshold.toFixed(2);
  $("undo").disabled = !history.length;
  draw();
}
function setMode(next) {
  mode = next;
  pendingCorners = [];
  document
    .querySelectorAll("[data-mode]")
    .forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("tool-help").textContent = help[mode];
  canvas.style.cursor =
    mode === "pan" ? "grab" : mode === "remove" ? "pointer" : "crosshair";
  draw();
}
function screenPoint(e) {
  const r = canvas.getBoundingClientRect();
  return [
    (e.clientX - r.left - view.x) / view.scale,
    (e.clientY - r.top - view.y) / view.scale,
  ];
}
function click(point) {
  if (!state) return;
  if (mode === "frame") {
    if (
      point[0] < 0 ||
      point[1] < 0 ||
      point[0] >= doc.width ||
      point[1] >= doc.height
    ) {
      status("Choose corners inside the photograph.");
      return;
    }
    pendingCorners.push(point.map(Math.round));
    if (pendingCorners.length === 4) {
      if (!C.convex(pendingCorners)) {
        pendingCorners = [];
        status(
          "Corners must form a convex quadrilateral in perimeter order. Try again.",
        );
        draw();
        return;
      }
      const polygon = pendingCorners.map((p) => p.slice());
      mutate("set_frame", { polygon }, () => {
        state.polygon = polygon;
      });
      pendingCorners = [];
      buildTiles();
      status(
        "Frame changed. Export its region JSON before running inference on a new photograph.",
      );
      setMode("remove");
    } else status(`Frame corner ${pendingCorners.length}/4 selected.`);
    draw();
    return;
  }
  if (!C.inside(state.polygon, point)) {
    status("Counts are restricted to the active frame interior.");
    return;
  }
  if (mode === "add") {
    const existing = state.manual.find(
      (p) =>
        Math.hypot(p.center[0] - point[0], p.center[1] - point[1]) <
        10 / view.scale,
    );
    if (existing) {
      mutate("remove_manual_point", { id: existing.id }, () => {
        state.manual = state.manual.filter((p) => p.id !== existing.id);
      });
      status("Manual point removed.");
    } else {
      const item = {
        id: "manual-" + Date.now() + "-" + state.manual.length,
        center: point.map((v) => Math.round(v * 10) / 10),
      };
      mutate("add_manual_point", item, () => state.manual.push(item));
      status("Missed barnacle marked.");
    }
    return;
  }
  if (mode === "remove") {
    const removed = new Set(state.removed);
    const found = doc.instances
      .filter(
        (p) =>
          p.score >= state.threshold &&
          C.inside(state.polygon, p.center) &&
          (removed.has(p.id) ? $("show-removed").checked : true) &&
          p.polygons.some((poly) => C.inside(poly, point)),
      )
      .sort(
        (a, b) =>
          Math.hypot(a.center[0] - point[0], a.center[1] - point[1]) -
          Math.hypot(b.center[0] - point[0], b.center[1] - point[1]),
      )[0];
    if (!found) {
      status(
        "No visible mask at that point. Use Mark missed to add a barnacle.",
      );
      return;
    }
    const restore = removed.has(found.id);
    mutate(
      restore ? "restore_proposal" : "remove_proposal",
      { id: found.id },
      () => {
        state.removed = restore
          ? state.removed.filter((id) => id !== found.id)
          : [...state.removed, found.id];
      },
    );
    status(
      restore ? "Proposal restored." : "Proposal removed; Undo is available.",
    );
  }
}
function buildTiles() {
  const b = bounds(state.polygon);
  tiles = [];
  for (let y = b.y; y < b.b; y += 512)
    for (let x = b.x; x < b.r; x += 512)
      tiles.push({
        x,
        y,
        r: Math.min(x + 512, b.r),
        b: Math.min(y + 512, b.b),
      });
  tile = -1;
}
function moveTile(direction) {
  if (!tiles.length) return;
  tile = (tile + direction + tiles.length) % tiles.length;
  fit(tiles[tile]);
  $("tile-position").textContent = `${tile + 1} / ${tiles.length}`;
  status(
    `Inspecting tile ${tile + 1} of ${tiles.length}. Tile navigation does not certify a completed review.`,
  );
}
function download(name, text, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
function exportReview() {
  const n = C.counts(doc, state);
  const value = {
    schema_version: 1,
    exported_at: new Date().toISOString(),
    review_complete: false,
    notice: "Provisional review; no verified completion or time-saving claim.",
    prediction_source: sampleKey,
    checkpoint_sha256: doc.checkpoint_sha256 || null,
    model_train_images: doc.model_train_images || [],
    image: {
      width: doc.width,
      height: doc.height,
      name: doc.image_original_name || doc.image,
      sha256: doc.image_sha256 || null,
    },
    count_rule: doc.count_rule,
    threshold: state.threshold,
    polygon: state.polygon,
    counts: n,
    accepted_model_ids: C.accepted(doc, state).map((p) => p.id),
    removed_model_ids: state.removed,
    manual_points: state.manual,
    events: state.events,
  };
  download(
    "barnacle-review.json",
    JSON.stringify(value, null, 2),
    "application/json",
  );
  status("Review and audit history exported.");
}
function safeImageUrl(value) {
  if (typeof value !== "string")
    throw Error("Prediction file is missing an image reference.");
  const url = new URL(value, location.href);
  if (
    url.origin !== location.origin ||
    !["http:", "https:"].includes(url.protocol)
  )
    throw Error(
      "Load the matching photo separately for external image references.",
    );
  return url.href;
}
async function showDocument(next, key, imageUrl) {
  const token = ++loadedToken;
  $("loading").classList.remove("hidden");
  try {
    C.validate(next);
    const fingerprint = await sha256(
      JSON.stringify({
        image_sha256: next.image_sha256,
        width: next.width,
        height: next.height,
        polygon: next.polygon,
        instances: next.instances,
        checkpoint: next.checkpoint_sha256,
      }),
    );
    const picture = new Image();
    await new Promise((resolve, reject) => {
      picture.onload = resolve;
      picture.onerror = () =>
        reject(
          Error("Could not load the photograph. Select its image file first."),
        );
      picture.src = imageUrl;
    });
    if (token !== loadedToken) return;
    if (
      picture.naturalWidth !== next.width ||
      picture.naturalHeight !== next.height
    )
      throw Error("Photo dimensions do not match the prediction coordinates.");
    doc = next;
    image = picture;
    sampleKey = key + ":" + fingerprint;
    state = C.initial(doc);
    history = [];
    try {
      const saved = JSON.parse(
        localStorage.getItem("barnacle-review-v1:" + sampleKey),
      );
      if (
        saved &&
        Number.isFinite(saved.threshold) &&
        saved.threshold >= 0.25 &&
        saved.threshold <= 0.95 &&
        Array.isArray(saved.removed) &&
        Array.isArray(saved.manual) &&
        saved.polygon?.length === 4 &&
        C.convex(saved.polygon) &&
        Array.isArray(saved.events)
      ) {
        state = saved;
      }
    } catch {}
    const train = (doc.model_train_images || []).join(", ");
    $("source-note").textContent =
      key === "unseen"
        ? `Unlabeled: no accuracy is known. Model trained on ${train}.`
        : `${key.startsWith("heldout") ? "Image-level held-out demonstration. " : ""}Model training images: ${train || "none"}.`;
    $("image-label").textContent = key.replaceAll("_", " ").toUpperCase();
    buildTiles();
    setMode("remove");
    resize();
    fitFrame();
    render();
    $("loading").classList.add("hidden");
    status("Click proposals to correct the count. Changes are saved locally.");
  } catch (error) {
    if (token === loadedToken) {
      $("loading").textContent = error.message;
      status(error.message);
    }
  }
}
async function loadSample(name) {
  try {
    const response = await fetch(`assets/${name}.json`);
    if (!response.ok)
      throw Error(
        "Sample data is unavailable. Serve the demo through Python HTTP server.",
      );
    const next = await response.json();
    await showDocument(next, name, safeImageUrl(next.image));
  } catch (error) {
    status(error.message);
    $("loading").textContent = error.message;
  }
}
$("sample").addEventListener("change", () => loadSample($("sample").value));
document
  .querySelectorAll("[data-mode]")
  .forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("threshold").addEventListener("change", () => {
  const threshold = Number($("threshold").value);
  mutate("set_threshold", { threshold }, () => {
    state.threshold = threshold;
  });
  status(
    "Threshold changed; this is exploratory review, not a new measured accuracy result.",
  );
});
$("threshold").addEventListener(
  "input",
  () =>
    ($("threshold-value").textContent = Number($("threshold").value).toFixed(
      2,
    )),
);
$("undo").addEventListener("click", () => {
  if (!history.length) return;
  const events = state.events.slice(),
    previous = history.pop();
  state = previous;
  state.events = events;
  state.events.push({ action: "undo", time: new Date().toISOString() });
  remember();
  buildTiles();
  render();
  status("Last correction undone.");
});
$("fit").addEventListener("click", fitFrame);
$("prev-tile").addEventListener("click", () => moveTile(-1));
$("next-tile").addEventListener("click", () => moveTile(1));
for (const id of ["show-masks", "show-removed"])
  $(id).addEventListener("change", draw);
$("export-json").addEventListener("click", exportReview);
$("export-csv").addEventListener("click", () => {
  const rows = [
    ["id", "origin", "x_original_pixel", "y_original_pixel", "model_score"],
  ];
  for (const p of C.accepted(doc, state))
    rows.push([p.id, "model", ...p.center, p.score]);
  for (const p of C.manualInside(state))
    rows.push([p.id, "manual", ...p.center, ""]);
  const csv = rows
    .map((row) =>
      row.map((v) => '"' + String(v).replaceAll('"', '""') + '"').join(","),
    )
    .join("\n");
  download("barnacle-accepted-points.csv", csv, "text/csv");
  status("Accepted points exported in original image coordinates.");
});
$("export-region").addEventListener("click", () =>
  download(
    "frame-region.json",
    JSON.stringify(
      {
        polygon: state.polygon,
        width: doc.width,
        height: doc.height,
        source: "human-selected frame interior",
      },
      null,
      2,
    ),
    "application/json",
  ),
);
$("reset").addEventListener("click", () => {
  mutate("reset_corrections", {}, () => {
    const events = state.events;
    state = C.initial(doc);
    state.events = events;
  });
  buildTiles();
  fitFrame();
  status("Corrections reset. Undo can restore them.");
});
canvas.addEventListener("pointerdown", (e) => {
  if (!state) return;
  if (mode === "pan") {
    drag = { sx: e.clientX, sy: e.clientY, x: view.x, y: view.y };
    canvas.setPointerCapture(e.pointerId);
  } else click(screenPoint(e));
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  view.x = drag.x + e.clientX - drag.sx;
  view.y = drag.y + e.clientY - drag.sy;
  draw();
});
canvas.addEventListener("pointerup", () => {
  drag = null;
});
canvas.addEventListener("pointercancel", () => {
  drag = null;
});
canvas.addEventListener(
  "wheel",
  (e) => {
    if (!state) return;
    e.preventDefault();
    const point = screenPoint(e),
      r = canvas.getBoundingClientRect();
    view.scale = Math.max(
      0.08,
      Math.min(7, view.scale * Math.exp(-e.deltaY * 0.001)),
    );
    view.x = e.clientX - r.left - point[0] * view.scale;
    view.y = e.clientY - r.top - point[1] * view.scale;
    draw();
  },
  { passive: false },
);
let customPhotoUrl = null,
  customPhotoHash = null;
$("photo-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  customPhotoHash = await sha256(await file.arrayBuffer());
  if (customPhotoUrl) URL.revokeObjectURL(customPhotoUrl);
  customPhotoUrl = URL.createObjectURL(file);
  const im = new Image();
  im.onload = () => {
    showDocument(
      {
        schema_version: 1,
        width: im.naturalWidth,
        height: im.naturalHeight,
        image: file.name,
        image_sha256: customPhotoHash,
        polygon: [
          [0, 0],
          [im.naturalWidth - 1, 0],
          [im.naturalWidth - 1, im.naturalHeight - 1],
          [0, im.naturalHeight - 1],
        ],
        instances: [],
        model_train_images: [],
        count_rule: "manual point inside/on frame",
      },
      "custom:" + file.name,
      customPhotoUrl,
    ).then(() => setMode("frame"));
  };
  im.src = customPhotoUrl;
});
$("prediction-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const next = C.validate(JSON.parse(await file.text()));
    if (
      customPhotoUrl &&
      next.image_sha256 &&
      customPhotoHash !== next.image_sha256
    )
      throw Error(
        "The loaded photo does not match the original image hash in these proposals. Load the matching original photo.",
      );
    const url = customPhotoUrl || safeImageUrl(next.image);
    await showDocument(
      next,
      "prediction:" + file.name + ":" + (next.checkpoint_sha256 || ""),
      url,
    );
  } catch (error) {
    status(error.message);
  }
});
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName))
    return;
  const modes = { a: "add", d: "remove", p: "pan", f: "frame" };
  if (modes[e.key.toLowerCase()]) setMode(modes[e.key.toLowerCase()]);
  if ((e.ctrlKey || e.metaKey) && e.key === "z") {
    e.preventDefault();
    $("undo").click();
  }
});
new ResizeObserver(resize).observe(canvas.parentElement);
fetch("../results/report.json")
  .then((r) => r.json())
  .then((report) => {
    const body = $("metrics-body");
    body.replaceChildren();
    for (const row of report.metrics.filter(
      (m) => m.method === "mask_rcnn" && m.score_threshold === 0.5,
    )) {
      const tr = document.createElement("tr");
      for (const value of [
        `Image ${row.test_image}`,
        `${row.reference_count.toLocaleString()} / ${row.predicted_count.toLocaleString()}`,
        `${(row.precision * 100).toFixed(1)}%`,
        `${(row.recall * 100).toFixed(1)}%`,
      ]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      body.append(tr);
    }
  })
  .catch(
    () =>
      ($("metrics-body").textContent =
        "See results/metrics.csv in the repository for measured results."),
  );
loadSample("unseen");
