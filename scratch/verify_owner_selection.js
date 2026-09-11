/**
 * Verification harness for the popup's capture-owner selection logic (audit P1-04).
 *
 * The popup runs as a plain (non-module) script, so its helpers are defined inside an
 * IIFE and cannot be imported. This harness extracts the *real* `pickCaptureOwner`
 * source text from `popup.js` and evaluates it, so the shipped logic is what gets
 * exercised rather than a hand-copied duplicate.
 *
 * Usage:  node scratch/verify_owner_selection.js
 */

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const POPUP_PATH = path.join(__dirname, "..", "extension_firefox", "popup", "popup.js");

/** Extract a top-level `function name(...) { ... }` block by brace matching. */
function extractFunction(source, name) {
  const marker = `function ${name}(`;
  const start = source.indexOf(marker);
  if (start === -1) throw new Error(`function ${name} not found in popup.js`);

  const bodyStart = source.indexOf("{", start);
  if (bodyStart === -1) throw new Error(`no body for ${name}`);

  let depth = 0;
  for (let i = bodyStart; i < source.length; i++) {
    const ch = source[i];
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces for ${name}`);
}

const popupSource = fs.readFileSync(POPUP_PATH, "utf8");
const pickCaptureOwner = eval(`(${extractFunction(popupSource, "pickCaptureOwner")})`);

const results = [];
function test(label, fn) {
  try {
    fn();
    results.push(`  PASS  ${label}`);
  } catch (err) {
    results.push(`  FAIL  ${label}\n        ${err.message}`);
    process.exitCode = 1;
  }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------
test("returns null when there are no candidates", () => {
  assert.strictEqual(pickCaptureOwner([]), null);
  assert.strictEqual(pickCaptureOwner(null), null);
  assert.strictEqual(pickCaptureOwner(undefined), null);
});

test("ignores frames without a video", () => {
  const owner = pickCaptureOwner([
    { frameId: 0, info: { hasVideo: false, isTop: true, isCapturing: false, score: -1 } },
    { frameId: 7, info: null },
  ]);
  assert.strictEqual(owner, null);
});

test("ignores frames that are already capturing", () => {
  const owner = pickCaptureOwner([
    { frameId: 3, info: { hasVideo: true, isTop: false, isCapturing: true, score: 999 } },
  ]);
  assert.strictEqual(owner, null);
});

test("picks the highest-scoring frame", () => {
  const owner = pickCaptureOwner([
    { frameId: 0, info: { hasVideo: true, isTop: true, isCapturing: false, score: 100 } },
    { frameId: 9, info: { hasVideo: true, isTop: false, isCapturing: false, score: 5000 } },
    { frameId: 4, info: { hasVideo: true, isTop: false, isCapturing: false, score: 250 } },
  ]);
  assert.strictEqual(owner.frameId, 9);
});

test("prefers the top frame when scores tie", () => {
  const owner = pickCaptureOwner([
    { frameId: 12, info: { hasVideo: true, isTop: false, isCapturing: false, score: 700 } },
    { frameId: 0, info: { hasVideo: true, isTop: true, isCapturing: false, score: 700 } },
  ]);
  assert.strictEqual(owner.frameId, 0);
});

test("breaks remaining ties by lowest frameId", () => {
  const owner = pickCaptureOwner([
    { frameId: 8, info: { hasVideo: true, isTop: false, isCapturing: false, score: 50 } },
    { frameId: 2, info: { hasVideo: true, isTop: false, isCapturing: false, score: 50 } },
  ]);
  assert.strictEqual(owner.frameId, 2);
});

test("stays deterministic when frameId is missing (no NaN comparison)", () => {
  const owner = pickCaptureOwner([
    { frameId: undefined, info: { hasVideo: true, isTop: false, isCapturing: false, score: 10 } },
    { frameId: undefined, info: { hasVideo: true, isTop: false, isCapturing: false, score: 20 } },
  ]);
  assert.ok(owner, "expected an owner even when frameId is undefined");
  assert.strictEqual(owner.info.score, 20);
});

test("a cross-origin player iframe beats an incidental muted top-frame video", () => {
  // Mirrors the saved page: the top document has no real player, the embedded player
  // iframe does. Scores come from scoreVideo() (playing video scores far higher).
  const owner = pickCaptureOwner([
    { frameId: 0, info: { hasVideo: true, isTop: true, isCapturing: false, score: 20000 } },
    { frameId: 5, info: { hasVideo: true, isTop: false, isCapturing: false, score: 1050000 } },
  ]);
  assert.strictEqual(owner.frameId, 5);
});

console.log("pickCaptureOwner verification:");
console.log(results.join("\n"));
console.log(process.exitCode ? "\nRESULT: FAILURES PRESENT" : "\nRESULT: all checks passed");
