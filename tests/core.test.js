// Unit tests for the math, embedding and parsing modules. Run with: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { pca, TSNE, kmeans, fitUnit, cosineSim, reduceForLayout, mulberry32, gaussFn } from "../js/math.js";
import { LexicalEmbedder } from "../js/embedders.js";
import { parseVectors, parseMeta } from "../js/parsers.js";
import { buildDemo } from "../js/samples.js";

// Three well separated Gaussian blobs in 20 dimensions
function blobs(perBlob = 25, dim = 20, seed = 1) {
  const g = gaussFn(mulberry32(seed));
  const centers = [0, 1, 2].map(() => Array.from({ length: dim }, () => g() * 5));
  const X = [], y = [];
  centers.forEach((c, k) => {
    for (let i = 0; i < perBlob; i++) { X.push(Float32Array.from(c, v => v + g() * 0.3)); y.push(k); }
  });
  return { X, y };
}
// Share of each point's 5 nearest neighbors (in 2D) that carry the same label
function neighborPurity(coords, y, k = 5) {
  let same = 0, total = 0;
  coords.forEach((a, i) => {
    coords.map((b, j) => [j, Math.hypot(a[0] - b[0], a[1] - b[1])])
      .filter(([j]) => j !== i).sort((p, q) => p[1] - q[1]).slice(0, k)
      .forEach(([j]) => { total++; if (y[j] === y[i]) same++; });
  });
  return same / total;
}

test("PCA returns components in decreasing order of variance", () => {
  const { X } = blobs();
  const P = pca(X, 3);
  assert.equal(P.coords.length, X.length);
  assert.ok(P.varRatio[0] >= P.varRatio[1] && P.varRatio[1] >= P.varRatio[2]);
  assert.ok(P.varRatio.reduce((a, b) => a + b, 0) <= 1.0001);
});

test("PCA components are orthonormal", () => {
  const { X } = blobs();
  const { comps } = pca(X, 3);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    const d = comps[a].reduce((s, v, i) => s + v * comps[b][i], 0);
    assert.ok(Math.abs(d - (a === b ? 1 : 0)) < 1e-6);
  }
});

test("t-SNE keeps clusters together", () => {
  const { X, y } = blobs();
  const t = new TSNE(reduceForLayout(X), { dim: 2, perplexity: 10 });
  for (let i = 0; i < 300; i++) t.step();
  assert.ok(neighborPurity(t.coords(), y) > 0.95);
});

test("k-means recovers separated blobs", () => {
  const { X, y } = blobs();
  const a = kmeans(X, 3);
  // each true blob should map to exactly one cluster
  for (const k of [0, 1, 2]) {
    const ids = new Set(a.filter((_, i) => y[i] === k));
    assert.equal(ids.size, 1);
  }
});

test("fitUnit scales coordinates into the unit sphere", () => {
  const { xyz } = fitUnit([[10, 0], [-10, 0], [0, 5]], 2);
  for (let i = 0; i < 3; i++) assert.ok(Math.hypot(xyz[i * 3], xyz[i * 3 + 1]) <= 1 + 1e-6);
});

test("lexical embedder ranks shared wording as more similar", () => {
  const texts = ["the cat sat on the mat", "a cat sat on a mat", "stock markets fell sharply today"];
  const E = new LexicalEmbedder(texts, { dim: 512 });
  const [a, b, c] = E.embedAll();
  assert.ok(cosineSim(a, b) > cosineSim(a, c));
  assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-5);
});

test("parseVectors reads TSV with labels and JSON", () => {
  const t = parseVectors("cat\t0.1\t0.2\ndog\t0.3\t0.4");
  assert.deepEqual(t.map(r => r.label), ["cat", "dog"]);
  assert.deepEqual(t[1].vector, [0.3, 0.4]);
  const j = parseVectors('[{"text":"hi","vector":[1,2],"group":"g"}]');
  assert.equal(j[0].group, "g");
  assert.throws(() => parseVectors("1\tx\t3"), /not a number/);
});

test("parseMeta treats the first row as a header only for multi-column data", () => {
  assert.deepEqual(parseMeta("a\nb").rows, [["a"], ["b"]]);
  const m = parseMeta("text\tgroup\nhello\tg1");
  assert.deepEqual(m.headers, ["text", "group"]);
  assert.deepEqual(m.rows, [["hello", "g1"]]);
});

test("demo vectors are unit length and grouped", () => {
  const d = buildDemo();
  assert.ok(d.length > 100);
  for (const r of d) assert.ok(Math.abs(Math.hypot(...r.vector) - 1) < 1e-4);
  assert.equal(new Set(d.map(r => r.group)).size, 8);
});
