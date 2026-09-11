// Mock DOM environment for testing SubtitleRenderer in Node.js
class MockClassList {
  constructor() {
    this.classes = new Set();
  }
  add(cls) { this.classes.add(cls); }
  remove(cls) { this.classes.delete(cls); }
  contains(cls) { return this.classes.has(cls); }
  toggle(cls, force) {
    if (force === undefined) {
      if (this.classes.has(cls)) this.classes.delete(cls);
      else this.classes.add(cls);
    } else if (force) {
      this.classes.add(cls);
    } else {
      this.classes.delete(cls);
    }
  }
}

class MockElement {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.classList = new MockClassList();
    this.children = [];
    this.parentElement = null;
    this.textContent = "";
    this.attributes = {};
  }
  setAttribute(name, val) { this.attributes[name] = String(val); }
  getAttribute(name) { return this.attributes[name]; }
  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx !== -1) {
      this.children.splice(idx, 1);
      child.parentElement = null;
    }
    return child;
  }
  replaceChildren(...newChildren) {
    this.children = [];
    for (const c of newChildren) {
      if (c && c.children) {
        for (const gc of c.children) {
          gc.parentElement = this;
          this.children.push(gc);
        }
      } else if (c) {
        c.parentElement = this;
        this.children.push(c);
      }
    }
  }
  remove() {
    if (this.parentElement) {
      this.parentElement.removeChild(this);
    }
  }
  get parentNode() {
    return this.parentElement;
  }
  querySelectorAll(sel) {
    const results = [];
    const check = (node) => {
      if (sel.startsWith(".")) {
        const cls = sel.slice(1);
        if (node.classList.contains(cls)) results.push(node);
      } else if (sel.startsWith("[data-sentence-id=")) {
        const match = sel.match(/\[data-sentence-id="([^"]+)"\]/);
        if (match && node.getAttribute("data-sentence-id") === match[1]) results.push(node);
      }
      for (const c of node.children) check(c);
    };
    for (const c of this.children) check(c);
    return results;
  }
  querySelector(sel) {
    // Basic selector match
    if (sel.includes(":not(.bs-evicting)")) {
      const baseSel = sel.replace(":not(.bs-evicting)", "");
      const res = this.querySelectorAll(baseSel);
      return res.find(el => !el.classList.contains("bs-evicting")) || null;
    }
    if (sel.startsWith(".")) {
      const cls = sel.slice(1);
      if (this.classList.contains(cls)) return this;
      for (const c of this.children) {
        const found = c.querySelector(sel);
        if (found) return found;
      }
    } else if (sel.startsWith("[data-sentence-id=")) {
      const match = sel.match(/\[data-sentence-id="([^"]+)"\]/);
      if (match && this.getAttribute("data-sentence-id") === match[1]) return this;
      for (const c of this.children) {
        const found = c.querySelector(sel);
        if (found) return found;
      }
    }
    return null;
  }
  get firstElementChild() {
    return this.children.length > 0 ? this.children[0] : null;
  }
}

global.document = {
  createElement(tag) { return new MockElement(tag); },
  createDocumentFragment() {
    const frag = new MockElement("fragment");
    return frag;
  }
};
global.window = {};

// Load SubtitleRenderer
const fs = require("fs");
const code = fs.readFileSync(__dirname + "/../extension_firefox/lib/subtitle-renderer.js", "utf8");
eval(code);
const SubtitleRenderer = window.SubtitleRenderer;

function assert(condition, message) {
  if (!condition) {
    console.error("❌ FAILED:", message);
    process.exit(1);
  }
  console.log("✅ PASSED:", message);
}

// Test Suite
async function runTests() {
  console.log("--- BẮT ĐẦU KIỂM THỬ SUBTITLE RENDERER ---\n");

  const container = new MockElement("div");
  const renderer = new SubtitleRenderer(container);

  // 1. Kiểm thử công thức tính thời gian hiển thị theo độ dài câu dịch
  console.log("1. Kiểm thử _calculateDuration:");
  const shortDur = renderer._calculateDuration("Xin chào"); // 8 chars -> 2500 + 480 = 2980 -> clamp min 3000
  assert(shortDur === 3000, `Câu ngắn 'Xin chào' = ${shortDur}ms (kỳ vọng 3000ms)`);

  const midDur = renderer._calculateDuration("Hôm nay trời nhiều mây và có thể có mưa."); // 40 chars -> 2500 + 2400 = 4900
  assert(midDur === 4900, `Câu trung bình 40 ký tự = ${midDur}ms (kỳ vọng 4900ms)`);

  const longText = "a".repeat(150); // 150 chars -> 2500 + 9000 = 11500 -> clamp max 10000
  const longDur = renderer._calculateDuration(longText);
  assert(longDur === 10000, `Câu siêu dài 150 ký tự = ${longDur}ms (kỳ vọng trần 10000ms)`);

  // 2. Kiểm thử reset bộ đếm khi câu Tầng 2 bị đẩy lên Tầng 1
  console.log("\n2. Kiểm thử reset bộ đếm khi câu lên Tầng 1:");
  renderer.clear();
  const t0 = Date.now();
  
  // Thêm câu A vào Tầng 2
  renderer.onUtteranceUpdate({ utterance_id: "utt-1", ui_text: "Hello world", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-1", translated: "Xin chào thế giới" }); // 17 chars -> 2500 + 1020 = 3520ms
  
  const sentA = renderer.completedSentences.find(s => s.id === "utt-1");
  assert(sentA !== undefined, "Câu A được thêm vào completedSentences");
  assert(sentA.duration === 3520, `Thời gian câu A = ${sentA.duration}ms`);
  const initialExpireA = sentA.expireAt;

  // Giả lập thời gian trôi qua 1000ms
  await new Promise(r => setTimeout(r, 50));
  
  // Thêm câu B vào Tầng 2 -> Câu A bị đẩy lên Tầng 1
  renderer.onUtteranceUpdate({ utterance_id: "utt-2", ui_text: "Good morning", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-2", translated: "Chào buổi sáng tốt lành nhé mọi người" }); // 38 chars -> 2500 + 2280 = 4780ms

  assert(renderer.completedSentences.length === 2, "Có 2 câu trong completedSentences (Tầng 1 + Tầng 2)");
  assert(sentA.expireAt > initialExpireA, "Câu A đã được RESET bộ đếm (expireAt mới lớn hơn ban đầu)");

  // 3. Kiểm thử ràng buộc: Tầng 1 dài hơn Tầng 2 -> Tầng 2 mất cùng lúc Tầng 1
  console.log("\n3. Kiểm thử ràng buộc Tầng 1 dài hơn Tầng 2:");
  renderer.clear();
  
  // Câu 1 (dài): 60 chars -> 2500 + 3600 = 6100ms
  renderer.onUtteranceUpdate({ utterance_id: "utt-long", ui_text: "Long sentence", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-long", translated: "Đây là một câu rất dài nhằm kiểm tra thời gian hiển thị của Tầng 1." });
  
  const longSent = renderer.completedSentences[0];
  const longDurExpected = 2500 + "Đây là một câu rất dài nhằm kiểm tra thời gian hiển thị của Tầng 1.".length * 60;
  assert(longSent.duration === longDurExpected, `Câu Tầng 1 có duration = ${longSent.duration}ms`);

  // Câu 2 (ngắn): 8 chars -> 3000ms min
  renderer.onUtteranceUpdate({ utterance_id: "utt-short", ui_text: "Hi", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-short", translated: "Xin chào" });

  const shortSent = renderer.completedSentences[1];
  assert(shortSent.duration === 3000, `Câu Tầng 2 duration gốc = ${shortSent.duration}ms`);
  assert(shortSent.expireAt === longSent.expireAt, `Ràng buộc thành công: Câu Tầng 2 có expireAt = ${shortSent.expireAt} bằng với Tầng 1 = ${longSent.expireAt}`);

  // 4. Kiểm thử Tầng 2 dài hơn Tầng 1 -> Tầng 1 mất trước, Tầng 2 tiếp tục
  console.log("\n4. Kiểm thử Tầng 2 dài hơn Tầng 1:");
  renderer.clear();

  // Câu 1 (ngắn): 3000ms
  renderer.onUtteranceUpdate({ utterance_id: "utt-s", ui_text: "Yes", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-s", translated: "Vâng ạ" });

  // Câu 2 (dài): 6000ms
  const longTranslation2 = "Câu này dài hơn rất nhiều so với câu ở Tầng 1 phía trên";
  renderer.onUtteranceUpdate({ utterance_id: "utt-l", ui_text: "Long text", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-l", translated: longTranslation2 });

  const s1 = renderer.completedSentences[0];
  const s2 = renderer.completedSentences[1];
  assert(s2.expireAt > s1.expireAt, `Tầng 2 expireAt (${s2.expireAt}) lớn hơn Tầng 1 (${s1.expireAt})`);

  // 5. Kiểm thử chu trình fade-out và xóa khỏi DOM
  console.log("\n5. Kiểm thử fade-out 0.5s:");
  renderer.clear();
  renderer.fadeDurationMs = 100; // rút ngắn để test nhanh

  renderer.onUtteranceUpdate({ utterance_id: "utt-fade", ui_text: "Test fade", is_final: true });
  renderer.onTranslation({ sentence_id: "utt-fade", translated: "Kiểm tra mờ dần" });

  const fadeItem = renderer.completedSentences[0];
  // Ép expireAt về quá khứ để kích hoạt tick
  fadeItem.expireAt = Date.now() - 10;
  renderer._onLifecycleTick();

  assert(fadeItem.isFadingOut === true, "Item được đánh dấu isFadingOut = true khi hết expireAt");
  
  // Sau fadeDurationMs (100ms), tick tiếp theo sẽ xoá item
  await new Promise(r => setTimeout(r, 120));
  renderer._onLifecycleTick();
  assert(renderer.completedSentences.length === 0, "Item đã biến mất hoàn toàn khỏi completedSentences sau khi kết thúc fade-out");

  // 6. Kiểm thử hiệu ứng chuyển tiếp mượt mà khi câu A bị đẩy khỏi Tầng 1
  console.log("\n6. Kiểm thử hiệu ứng chuyển tiếp mượt mà khi câu bị đẩy khỏi Tầng 1:");
  renderer.clear();
  renderer.maxLines = 2; // 1 Tầng 1 + 1 Tầng 2

  // Câu 1 xuất hiện
  renderer.onUtteranceUpdate({ utterance_id: "câu-1", ui_text: "Sent 1", is_final: true });
  renderer.onTranslation({ sentence_id: "câu-1", translated: "Câu một" });

  // Câu 2 xuất hiện -> Câu 1 bị đẩy lên Tầng 1, Câu 2 ở Tầng 2
  renderer.onUtteranceUpdate({ utterance_id: "câu-2", ui_text: "Sent 2", is_final: true });
  renderer.onTranslation({ sentence_id: "câu-2", translated: "Câu hai" });

  const elCau1Before = renderer.historyLayer.querySelector('[data-sentence-id="câu-1"]');
  assert(elCau1Before !== null, "Câu 1 đang hiển thị ở Tầng 1");
  assert(!elCau1Before.classList.contains("bs-evicting"), "Câu 1 chưa bị evicting");

  // Câu 3 xuất hiện -> Câu 2 bị đẩy lên Tầng 1, Câu 1 bị đẩy KHỎI Tầng 1
  renderer.onUtteranceUpdate({ utterance_id: "câu-3", ui_text: "Sent 3", is_final: true });
  renderer.onTranslation({ sentence_id: "câu-3", translated: "Câu ba" });

  // Kiểm tra: Câu 1 KHÔNG bị xoá ngay lập tức (không biến mất đột ngột)!
  const elCau1During = renderer.historyLayer.querySelector('[data-sentence-id="câu-1"]');
  assert(elCau1During !== null, "Câu 1 KHÔNG biến mất đột ngột, vẫn còn trong DOM");
  assert(elCau1During.classList.contains("bs-evicting"), "Câu 1 được gán class .bs-evicting để mờ dần và trôi lên");

  const elCau2 = renderer.historyLayer.querySelector('[data-sentence-id="câu-2"]');
  assert(elCau2 !== null, "Câu 2 đã bước vào Tầng 1");

  // Đợi sau timeout 400ms (> 380ms)
  await new Promise(r => setTimeout(r, 420));
  const elCau1After = renderer.historyLayer.querySelector('[data-sentence-id="câu-1"]');
  assert(elCau1After === null, "Câu 1 đã được gỡ bỏ khỏi DOM sau khi hoàn thành animation mượt mà");

  console.log("\n🎉 TẤT CẢ CÁC BÀI KIỂM THỬ ĐỀU THÀNH CÔNG!");
  process.exit(0);
}

runTests().catch(err => {
  console.error("Lỗi:", err);
  process.exit(1);
});
