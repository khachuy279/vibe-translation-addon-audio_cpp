// Kiểm thử CHỨC NĂNG cho quy tắc hiển thị bản dịch (tắt "chạy chữ").
//
// Vì sao: người dùng yêu cầu bản dịch chỉ hiện MỘT LẦN khi đã có bản dịch hoàn chỉnh.
// Quyết định đó nằm ở `extension_firefox/lib/subtitle-policy.js` (logic thuần) nên test
// được bằng Node, không cần DOM/browser.
//
// Dùng: node backend/tests/js/subtitle_policy_test.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");

function findPolicy() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const candidate = path.join(dir, "extension_firefox", "lib", "subtitle-policy.js");
    if (fs.existsSync(candidate)) return candidate;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy extension_firefox/lib/subtitle-policy.js");
}

const POLICY_PATH = process.argv[2] || findPolicy();
const failures = [];
function check(cond, msg, extra) {
  if (cond) {
    console.log("  PASS " + msg);
  } else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

// Nạp policy trong scope giả (self) như trong content script.
const src = fs.readFileSync(POLICY_PATH, "utf8");
const scope = {};
const vm = require("vm");
vm.runInContext(src, vm.createContext(scope), { filename: POLICY_PATH });
const P = scope.BSSubtitlePolicy;

console.log(`Policy: ${path.relative(process.cwd(), POLICY_PATH)}`);
check(!!P && typeof P.shouldApplyTranslation === "function", "module phơi BSSubtitlePolicy");
check(P.DEFAULT_SHOW_TRANSLATION_ONCE === true, "mặc định là TẮT chạy chữ (hiện 1 lần)");

const isPartial = P.isPartialTranslation;
check(isPartial({ partial: true }) === true, "nhận diện `partial: true`");
check(isPartial({ is_partial: true }) === true, "nhận diện `is_partial: true`");
check(isPartial({ isPartial: true }) === true, "nhận diện `isPartial: true`");
check(isPartial({ status: "partial" }) === true, "nhận diện `status: partial`");
check(isPartial({ status: "streaming" }) === true, "nhận diện `status: streaming`");
check(isPartial({ status: "ok", text: "xin chào" }) === false, "bản hoàn chỉnh KHÔNG bị coi là partial");
check(isPartial(null) === false, "payload null an toàn");
check(isPartial("chuỗi") === false, "payload không phải object an toàn");

// Mặc định (showOnce = true): bỏ partial, nhận bản hoàn chỉnh
check(P.shouldApplyTranslation({ partial: true }) === false,
  "MẶC ĐỊNH: bỏ qua mảnh dịch dở (=> không chạy chữ)");
check(P.shouldApplyTranslation({ status: "ok", text: "Xin chào" }) === true,
  "MẶC ĐỊNH: hiện bản dịch hoàn chỉnh");
check(P.shouldApplyTranslation({ partial: true }, {}) === false,
  "opts rỗng vẫn dùng mặc định");
check(P.shouldApplyTranslation({ status: "ok" }, { showOnce: undefined }) === true,
  "showOnce=undefined => mặc định");

// Người dùng bật lại chạy chữ (showOnce = false): chấp nhận cả partial
check(P.shouldApplyTranslation({ partial: true }, { showOnce: false }) === true,
  "showOnce=false: cho phép chạy chữ (partial được hiện)");
check(P.shouldApplyTranslation({ status: "ok" }, { showOnce: false }) === true,
  "showOnce=false: bản hoàn chỉnh vẫn hiện");

// Mô phỏng đúng chuỗi thông điệp backend gửi cho MỘT câu:
//   partial (nhiều lần) -> final
const stream = [
  { sentence_id: "u1", text: "Hello", partial: true, status: "ok" },
  { sentence_id: "u1", text: "Hello everyone", partial: true, status: "ok" },
  { sentence_id: "u1", text: "Xin chào", partial: true, status: "ok" },
  { sentence_id: "u1", text: "Xin chào tất cả mọi người", status: "ok" },
];
const appliedShowOnce = stream.filter((m) => P.shouldApplyTranslation(m, { showOnce: true }));
const appliedTyping = stream.filter((m) => P.shouldApplyTranslation(m, { showOnce: false }));
check(appliedShowOnce.length === 1 && appliedShowOnce[0].text === "Xin chào tất cả mọi người",
  "chuỗi streaming: CHỈ 1 lần hiển thị và là bản hoàn chỉnh",
  `nhận ${appliedShowOnce.length}`);
check(appliedTyping.length === 4, "khi bật chạy chữ: hiện cả 4 lần (hành vi cũ)", `nhận ${appliedTyping.length}`);

console.log("");
if (failures.length) {
  console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
  process.exit(1);
}
console.log("KẾT QUẢ: PASS (quy tắc hiển thị bản dịch đúng)");
process.exit(0);
