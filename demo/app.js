"use strict";
async function sha256(value) {
  const bytes =
    typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}
const reviewLogic = BarnacleCore;
const byId = (id) => document.getElementById(id);
const canvas = byId("canvas");
const ctx = canvas.getContext("2d");
// Proposals stay unchanged. Corrections live separately so exports can explain the count.
let proposals = null;
let reviewState = null;
let image = null;
let mode = "remove";
let view = { x: 0, y: 0, scale: 1 };
let history = [];
let tile = -1;
let tiles = [];
let pendingCorners = [];
let drag = null;
let loadedToken = 0;
let sampleKey = "unseen";
const help = {
  remove:
    "Click a proposal to remove it. Enable removed outlines to restore one.",
  add: "Click the center of a missed barnacle inside the frame. Click an added point to remove it.",
  pan: "Drag to move around the photograph. Scroll to zoom at the pointer.",
  frame:
    "Click the four inner frame corners in perimeter order. The old frame stays active until all four are valid.",
};
function setStatus(text) {
  byId("status").textContent = text;
}
function saveReview() {
  if (!reviewState) return;
  try {
    localStorage.setItem(
      "barnacle-review-v1:" + sampleKey,
      JSON.stringify(reviewState),
    );
  } catch {
    setStatus(
      "Browser storage unavailable; export your corrections before leaving.",
    );
  }
}
// Snapshot before each edit; undo restores the count but keeps the audit history.
function editReview(action, details, applyChange) {
  history.push(JSON.parse(JSON.stringify(reviewState)));
  applyChange();
  reviewState.events.push({ action, details, time: new Date().toISOString() });
  saveReview();
  updateReviewDisplay();
}
let canvasSize = null;
function resize() {
  const r = canvas.getBoundingClientRect(),
    dpr = window.devicePixelRatio || 1;
  if (
    canvasSize &&
    canvasSize.width > 50 &&
    canvasSize.height > 70 &&
    reviewState &&
    r.width > 50 &&
    r.height > 70
  ) {
    // Keep the same image center and relative zoom when the workspace resizes.
    const box = bounds(reviewState.polygon);
    const fitScale = (size) =>
      Math.min(
        (size.width - 50) / (box.r - box.x),
        (size.height - 70) / (box.b - box.y),
      );
    const centerX = (canvasSize.width / 2 - view.x) / view.scale;
    const centerY = (canvasSize.height / 2 - view.y) / view.scale;
    view.scale *= fitScale(r) / fitScale(canvasSize);
    view.x = r.width / 2 - centerX * view.scale;
    view.y = r.height / 2 - centerY * view.scale;
  }
  canvasSize = { width: r.width, height: r.height };
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawCanvas();
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
  drawCanvas();
}
function fitFrame() {
  if (!reviewState) return;
  tile = -1;
  fit(bounds(reviewState.polygon));
  byId("tile-position").textContent = "Overview";
}
function path(polygon) {
  ctx.beginPath();
  polygon.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.closePath();
}
function drawCanvas() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  if (!image || !reviewState) return;
  ctx.save();
  ctx.translate(view.x, view.y);
  ctx.scale(view.scale, view.scale);
  ctx.drawImage(image, 0, 0, proposals.width, proposals.height);
  // Dim the photograph outside the active scientific region.
  ctx.beginPath();
  ctx.rect(0, 0, proposals.width, proposals.height);
  const p = reviewState.polygon;
  ctx.moveTo(...p[0]);
  for (let i = 1; i < p.length; i++) ctx.lineTo(...p[i]);
  ctx.closePath();
  ctx.fillStyle = "rgba(3,18,17,.6)";
  ctx.fill("evenodd");
  const removed = new Set(reviewState.removed),
    line = 1.15 / view.scale;
  if (byId("show-masks").checked)
    for (const inst of proposals.instances) {
      if (
        inst.score < reviewState.threshold ||
        !reviewLogic.inside(reviewState.polygon, inst.center)
      )
        continue;
      const deleted = removed.has(inst.id);
      if (deleted && !byId("show-removed").checked) continue;
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
  path(reviewState.polygon);
  ctx.stroke();
  for (const p of reviewLogic.manualInside(reviewState)) {
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
  byId("zoom-label").textContent = Math.round(view.scale * 100) + "%";
}
function updateReviewDisplay() {
  if (!proposals || !reviewState) return;
  const counts = reviewLogic.counts(proposals, reviewState);
  byId("final-count").textContent = counts.total.toLocaleString();
  byId("proposal-count").textContent = counts.proposals.toLocaleString();
  byId("removed-count").textContent = counts.removed;
  byId("manual-count").textContent = counts.manual;
  byId("threshold").value = reviewState.threshold;
  byId("threshold-value").textContent = reviewState.threshold.toFixed(2);
  byId("undo").disabled = !history.length;
  drawCanvas();
}
function setMode(next) {
  mode = next;
  pendingCorners = [];
  document.querySelectorAll("[data-mode]").forEach((b) => {
    const selected = b.dataset.mode === mode;
    b.classList.toggle("active", selected);
    b.setAttribute("aria-pressed", String(selected));
  });
  byId("tool-help").textContent = help[mode];
  canvas.style.cursor =
    mode === "pan" ? "grab" : mode === "remove" ? "pointer" : "crosshair";
  drawCanvas();
}
// Canvas coordinates are CSS pixels. Saved points always use original photo pixels.
function imagePointFromEvent(e) {
  const r = canvas.getBoundingClientRect();
  return [
    (e.clientX - r.left - view.x) / view.scale,
    (e.clientY - r.top - view.y) / view.scale,
  ];
}
function handleImageClick(point) {
  if (!reviewState) return;
  if (mode === "frame") {
    if (
      point[0] < 0 ||
      point[1] < 0 ||
      point[0] >= proposals.width ||
      point[1] >= proposals.height
    ) {
      setStatus("Choose corners inside the photograph.");
      return;
    }
    pendingCorners.push(point.map(Math.round));
    if (pendingCorners.length === 4) {
      if (!reviewLogic.convex(pendingCorners)) {
        pendingCorners = [];
        setStatus(
          "Corners must form a convex quadrilateral in perimeter order. Try again.",
        );
        drawCanvas();
        return;
      }
      const polygon = pendingCorners.map((p) => p.slice());
      editReview("set_frame", { polygon }, () => {
        reviewState.polygon = polygon;
      });
      pendingCorners = [];
      buildTiles();
      setStatus(
        "Frame changed. Export its region JSON before running inference on a new photograph.",
      );
      setMode("remove");
    } else setStatus(`Frame corner ${pendingCorners.length}/4 selected.`);
    drawCanvas();
    return;
  }
  if (!reviewLogic.inside(reviewState.polygon, point)) {
    setStatus("Counts are restricted to the active frame interior.");
    return;
  }
  if (mode === "add") {
    const existing = reviewState.manual.find(
      (p) =>
        Math.hypot(p.center[0] - point[0], p.center[1] - point[1]) <
        10 / view.scale,
    );
    if (existing) {
      editReview("remove_manual_point", { id: existing.id }, () => {
        reviewState.manual = reviewState.manual.filter(
          (p) => p.id !== existing.id,
        );
      });
      setStatus("Manual point removed.");
    } else {
      const item = {
        id: "manual-" + Date.now() + "-" + reviewState.manual.length,
        center: point.map((v) => Math.round(v * 10) / 10),
      };
      editReview("add_manual_point", item, () => reviewState.manual.push(item));
      setStatus("Missed barnacle marked.");
    }
    return;
  }
  if (mode === "remove") {
    const removed = new Set(reviewState.removed);
    const found = proposals.instances
      .filter(
        (p) =>
          p.score >= reviewState.threshold &&
          reviewLogic.inside(reviewState.polygon, p.center) &&
          (removed.has(p.id) ? byId("show-removed").checked : true) &&
          p.polygons.some((poly) => reviewLogic.inside(poly, point)),
      )
      .sort(
        (a, b) =>
          Math.hypot(a.center[0] - point[0], a.center[1] - point[1]) -
          Math.hypot(b.center[0] - point[0], b.center[1] - point[1]),
      )[0];
    if (!found) {
      setStatus(
        "No visible mask at that point. Use Mark missed to add a barnacle.",
      );
      return;
    }
    const restore = removed.has(found.id);
    editReview(
      restore ? "restore_proposal" : "remove_proposal",
      { id: found.id },
      () => {
        reviewState.removed = restore
          ? reviewState.removed.filter((id) => id !== found.id)
          : [...reviewState.removed, found.id];
      },
    );
    setStatus(
      restore ? "Proposal restored." : "Proposal removed; Undo is available.",
    );
  }
}
function buildTiles() {
  const b = bounds(reviewState.polygon);
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
  byId("tile-position").textContent = `${tile + 1} / ${tiles.length}`;
  setStatus(
    `Inspecting tile ${tile + 1} of ${tiles.length}. Check all barnacles in this tile.`,
  );
}
function download(name, text, type) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
// Tile navigation is not proof of review completion, so every export stays provisional.
function exportReview() {
  const counts = reviewLogic.counts(proposals, reviewState);
  const value = {
    schema_version: 1,
    exported_at: new Date().toISOString(),
    review_complete: false,
    notice: "Provisional count; review completion has not been verified.",
    prediction_source: sampleKey,
    checkpoint_sha256: proposals.checkpoint_sha256 || null,
    model_train_images: proposals.model_train_images || [],
    image: {
      width: proposals.width,
      height: proposals.height,
      name: proposals.image_original_name || proposals.image,
      sha256: proposals.image_sha256 || null,
    },
    count_rule: proposals.count_rule,
    threshold: reviewState.threshold,
    polygon: reviewState.polygon,
    counts,
    accepted_model_ids: reviewLogic
      .accepted(proposals, reviewState)
      .map((p) => p.id),
    removed_model_ids: reviewState.removed,
    manual_points: reviewState.manual,
    events: reviewState.events,
  };
  download(
    "barnacle-review.json",
    JSON.stringify(value, null, 2),
    "application/json",
  );
  setStatus("Review and audit history exported.");
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
// A fingerprint prevents corrections from being applied to a different model output.
async function showDocument(next, key, imageUrl) {
  // Ignore a slow image load if the user has already selected another sample.
  const token = ++loadedToken;
  byId("loading").classList.remove("hidden");
  try {
    reviewLogic.validate(next);
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
    proposals = next;
    image = picture;
    sampleKey = key + ":" + fingerprint;
    reviewState = reviewLogic.initial(proposals);
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
        reviewLogic.convex(saved.polygon) &&
        Array.isArray(saved.events)
      ) {
        reviewState = saved;
      }
    } catch {
      // An unreadable saved review should not stop a fresh review from loading.
    }
    const train = (proposals.model_train_images || []).join(", ");
    byId("source-note").textContent =
      key === "unseen"
        ? `No reference labels. Trained on ${train}.`
        : `Training photographs: ${train || "none"}.`;
    byId("image-label").textContent =
      {
        unseen: "unseen_img1.png",
        heldout_1: "img1.png · held out",
        heldout_2: "img2.png · held out",
      }[key] || "Your photograph";
    buildTiles();
    setMode("remove");
    resize();
    fitFrame();
    updateReviewDisplay();
    byId("loading").classList.add("hidden");
    setStatus(
      "Click proposals to correct the count. Changes are saved locally.",
    );
  } catch (error) {
    if (token === loadedToken) {
      byId("loading").textContent = error.message;
      setStatus(error.message);
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
    setStatus(error.message);
    byId("loading").textContent = error.message;
  }
}
// Wire the controls after defining the review and drawing helpers.
byId("sample").addEventListener("change", () =>
  loadSample(byId("sample").value),
);
document
  .querySelectorAll("[data-mode]")
  .forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
byId("threshold").addEventListener("change", () => {
  const threshold = Number(byId("threshold").value);
  editReview("set_threshold", { threshold }, () => {
    reviewState.threshold = threshold;
  });
  setStatus("Score cutoff changed. Check the newly visible proposals.");
});
byId("threshold").addEventListener(
  "input",
  () =>
    (byId("threshold-value").textContent = Number(
      byId("threshold").value,
    ).toFixed(2)),
);
byId("undo").addEventListener("click", () => {
  if (!history.length) return;
  const events = reviewState.events.slice(),
    previous = history.pop();
  reviewState = previous;
  reviewState.events = events;
  reviewState.events.push({ action: "undo", time: new Date().toISOString() });
  saveReview();
  buildTiles();
  updateReviewDisplay();
  setStatus("Last correction undone.");
});
byId("fit").addEventListener("click", () => {
  fitFrame();
  setStatus("Showing the whole frame.");
});
byId("prev-tile").addEventListener("click", () => moveTile(-1));
byId("next-tile").addEventListener("click", () => moveTile(1));
for (const id of ["show-masks", "show-removed"])
  byId(id).addEventListener("change", drawCanvas);
byId("export-json").addEventListener("click", exportReview);
byId("export-csv").addEventListener("click", () => {
  const rows = [
    ["id", "origin", "x_original_pixel", "y_original_pixel", "model_score"],
  ];
  for (const p of reviewLogic.accepted(proposals, reviewState))
    rows.push([p.id, "model", ...p.center, p.score]);
  for (const p of reviewLogic.manualInside(reviewState))
    rows.push([p.id, "manual", ...p.center, ""]);
  const csv = rows
    .map((row) =>
      row.map((v) => '"' + String(v).replaceAll('"', '""') + '"').join(","),
    )
    .join("\n");
  download("barnacle-accepted-points.csv", csv, "text/csv");
  setStatus("Accepted points exported in original image coordinates.");
});
byId("export-region").addEventListener("click", () =>
  download(
    "frame-region.json",
    JSON.stringify(
      {
        polygon: reviewState.polygon,
        width: proposals.width,
        height: proposals.height,
        source: "human-selected frame interior",
      },
      null,
      2,
    ),
    "application/json",
  ),
);
byId("reset").addEventListener("click", () => {
  editReview("reset_corrections", {}, () => {
    const events = reviewState.events;
    reviewState = reviewLogic.initial(proposals);
    reviewState.events = events;
  });
  buildTiles();
  fitFrame();
  setStatus("Corrections reset. Undo can restore them.");
});
canvas.addEventListener("pointerdown", (e) => {
  if (!reviewState) return;
  if (mode === "pan") {
    drag = { sx: e.clientX, sy: e.clientY, x: view.x, y: view.y };
    canvas.setPointerCapture(e.pointerId);
  } else handleImageClick(imagePointFromEvent(e));
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  view.x = drag.x + e.clientX - drag.sx;
  view.y = drag.y + e.clientY - drag.sy;
  drawCanvas();
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
    if (!reviewState) return;
    e.preventDefault();
    const point = imagePointFromEvent(e),
      r = canvas.getBoundingClientRect();
    view.scale = Math.max(
      0.08,
      Math.min(7, view.scale * Math.exp(-e.deltaY * 0.001)),
    );
    view.x = e.clientX - r.left - point[0] * view.scale;
    view.y = e.clientY - r.top - point[1] * view.scale;
    drawCanvas();
  },
  { passive: false },
);
let customPhotoUrl = null,
  customPhotoHash = null;
byId("photo-file").addEventListener("change", async (e) => {
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
byId("prediction-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const next = reviewLogic.validate(JSON.parse(await file.text()));
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
    setStatus(error.message);
  }
});
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName))
    return;
  const modes = { a: "add", d: "remove", p: "pan", f: "frame" };
  if (modes[e.key.toLowerCase()]) setMode(modes[e.key.toLowerCase()]);
  if ((e.ctrlKey || e.metaKey) && e.key === "z") {
    e.preventDefault();
    byId("undo").click();
  }
});
new ResizeObserver(resize).observe(canvas.parentElement);
fetch("../results/report.json")
  .then((r) => r.json())
  .then((report) => {
    const body = byId("metrics-body");
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
      (byId("metrics-body").textContent =
        "See results/metrics.csv in the repository for measured results."),
  );
loadSample("unseen");
