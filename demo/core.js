/* Counting logic is separate from canvas/UI so it can be regression tested. */
(function (root) {
  function inside(polygon, point) {
    let hit = false;
    const [x, y] = point;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [ax, ay] = polygon[j],
        [bx, by] = polygon[i];
      const cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax);
      if (
        Math.abs(cross) < 1e-6 &&
        x >= Math.min(ax, bx) &&
        x <= Math.max(ax, bx) &&
        y >= Math.min(ay, by) &&
        y <= Math.max(ay, by)
      )
        return true;
      if (ay > y != by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax)
        hit = !hit;
    }
    return hit;
  }
  function convex(polygon) {
    if (polygon.length !== 4) return false;
    let sign = 0;
    for (let i = 0; i < 4; i++) {
      const a = polygon[i],
        b = polygon[(i + 1) % 4],
        c = polygon[(i + 2) % 4];
      const cross =
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      if (Math.abs(cross) < 1e-6) return false;
      if (sign && Math.sign(cross) !== sign) return false;
      sign = Math.sign(cross);
    }
    return true;
  }
  function accepted(doc, state) {
    const removed = new Set(state.removed);
    return doc.instances.filter(
      (p) =>
        p.score >= state.threshold &&
        !removed.has(p.id) &&
        inside(state.polygon, p.center),
    );
  }
  function manualInside(state) {
    return state.manual.filter((p) => inside(state.polygon, p.center));
  }
  function counts(doc, state) {
    const eligible = doc.instances.filter(
      (p) => p.score >= state.threshold && inside(state.polygon, p.center),
    );
    const kept = accepted(doc, state),
      manual = manualInside(state);
    return {
      proposals: eligible.length,
      removed: eligible.length - kept.length,
      manual: manual.length,
      total: kept.length + manual.length,
    };
  }
  function validate(doc) {
    if (
      doc.schema_version !== 1 ||
      !Number.isInteger(doc.width) ||
      !Number.isInteger(doc.height) ||
      doc.width < 1 ||
      doc.height < 1 ||
      doc.width > 30000 ||
      doc.height > 30000
    )
      throw Error("Unsupported prediction file or image dimensions.");
    const point = (p) =>
      Array.isArray(p) &&
      p.length === 2 &&
      p.every(Number.isFinite) &&
      p[0] >= 0 &&
      p[1] >= 0 &&
      p[0] < doc.width &&
      p[1] < doc.height;
    if (
      !Array.isArray(doc.polygon) ||
      !doc.polygon.every(point) ||
      !convex(doc.polygon)
    )
      throw Error("The frame must be a convex four-corner polygon.");
    if (!Array.isArray(doc.instances) || doc.instances.length > 50000)
      throw Error("Invalid proposal list.");
    const ids = new Set();
    for (const p of doc.instances) {
      if (
        typeof p.id !== "string" ||
        ids.has(p.id) ||
        !point(p.center) ||
        !Number.isFinite(p.score) ||
        p.score < 0 ||
        p.score > 1 ||
        !Array.isArray(p.box) ||
        p.box.length !== 4 ||
        !p.box.every(Number.isFinite) ||
        !Array.isArray(p.polygons)
      )
        throw Error("Invalid instance record.");
      ids.add(p.id);
      for (const polygon of p.polygons)
        if (!Array.isArray(polygon) || !polygon.every(point))
          throw Error("Invalid mask polygon.");
    }
    return doc;
  }
  function initial(doc) {
    return {
      threshold: 0.5,
      removed: [],
      manual: [],
      polygon: doc.polygon.map((p) => p.slice()),
      events: [],
    };
  }
  const api = {
    inside,
    convex,
    accepted,
    manualInside,
    counts,
    validate,
    initial,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BarnacleCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
