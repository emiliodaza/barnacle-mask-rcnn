/* Shared counting rules. Nothing here reads the DOM or changes a review.
 * The Python pipeline and browser both include centroids on the frame boundary.
 */
(function (root) {
  // Ray casting with an explicit edge check: boundary points count as inside.
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
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax)
        hit = !hit;
    }
    return hit;
  }
  // All turns must have the same sign; reject crossed or collinear frame corners.
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
  // Hiding proposals by score or region never deletes their correction history.
  function accepted(proposals, review) {
    const removed = new Set(review.removed);
    return proposals.instances.filter(
      (p) =>
        p.score >= review.threshold &&
        !removed.has(p.id) &&
        inside(review.polygon, p.center),
    );
  }
  function manualInside(review) {
    return review.manual.filter((p) => inside(review.polygon, p.center));
  }
  function counts(proposals, review) {
    const eligible = proposals.instances.filter(
      (p) => p.score >= review.threshold && inside(review.polygon, p.center),
    );
    const kept = accepted(proposals, review),
      manual = manualInside(review);
    return {
      proposals: eligible.length,
      removed: eligible.length - kept.length,
      manual: manual.length,
      total: kept.length + manual.length,
    };
  }
  // Check imported JSON before drawing or accepting any saved edits.
  function validate(proposals) {
    if (
      proposals.schema_version !== 1 ||
      !Number.isInteger(proposals.width) ||
      !Number.isInteger(proposals.height) ||
      proposals.width < 1 ||
      proposals.height < 1 ||
      proposals.width > 30000 ||
      proposals.height > 30000
    )
      throw Error("Unsupported prediction file or image dimensions.");
    const point = (p) =>
      Array.isArray(p) &&
      p.length === 2 &&
      p.every(Number.isFinite) &&
      p[0] >= 0 &&
      p[1] >= 0 &&
      p[0] < proposals.width &&
      p[1] < proposals.height;
    if (
      !Array.isArray(proposals.polygon) ||
      !proposals.polygon.every(point) ||
      !convex(proposals.polygon)
    )
      throw Error("The frame must be a convex four-corner polygon.");
    if (
      !Array.isArray(proposals.instances) ||
      proposals.instances.length > 50000
    )
      throw Error("Invalid proposal list.");
    const ids = new Set();
    for (const p of proposals.instances) {
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
    return proposals;
  }
  function initial(proposals) {
    return {
      threshold: 0.5,
      removed: [],
      manual: [],
      polygon: proposals.polygon.map((p) => p.slice()),
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
