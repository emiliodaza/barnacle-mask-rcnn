const assert = require("node:assert/strict");
const C = require("../demo/core.js");
const doc = {
  schema_version: 1,
  width: 100,
  height: 100,
  polygon: [
    [0, 0],
    [90, 0],
    [90, 90],
    [0, 90],
  ],
  instances: [
    {
      id: "1",
      center: [10, 10],
      score: 0.9,
      box: [5, 5, 15, 15],
      polygons: [
        [
          [5, 5],
          [15, 5],
          [15, 15],
          [5, 15],
        ],
      ],
    },
    {
      id: "2",
      center: [20, 20],
      score: 0.4,
      box: [15, 15, 25, 25],
      polygons: [
        [
          [15, 15],
          [25, 15],
          [25, 25],
          [15, 25],
        ],
      ],
    },
  ],
};
C.validate(doc);
const state = C.initial(doc);
assert.equal(C.counts(doc, state).total, 1);
state.removed = ["1"];
assert.equal(C.counts(doc, state).total, 0);
state.manual = [
  { id: "m", center: [30, 30] },
  { id: "outside", center: [99, 99] },
];
assert.equal(C.counts(doc, state).total, 1);
state.threshold = 0.25;
assert.equal(C.counts(doc, state).total, 2);
assert.equal(C.inside(doc.polygon, [0, 10]), true);
assert.equal(C.inside(doc.polygon, [99, 10]), false);
assert.equal(
  C.convex([
    [0, 0],
    [10, 10],
    [0, 10],
    [10, 0],
  ]),
  false,
);
assert.throws(() =>
  C.validate({ ...doc, instances: [doc.instances[0], doc.instances[0]] }),
);
assert.throws(() => C.validate({ ...doc, width: Infinity }));
console.log(
  "Review counting, region boundaries, threshold changes, and input validation passed.",
);
