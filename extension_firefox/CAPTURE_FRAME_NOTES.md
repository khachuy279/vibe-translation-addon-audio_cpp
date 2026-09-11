# Capture-frame invariants — DO NOT BREAK THESE

> **Đọc file này trước khi sửa `manifest.json`, `popup/popup.js`, hoặc `content/content-script.js`.**
>
> Các invariant dưới đây từng bị phá vỡ trong quá trình phát triển, và **mỗi lần phá vỡ đều
> cho ra CÙNG MỘT triệu chứng gây nhầm lẫn**:
>
> ```
> ❌ Không tìm thấy video nào (Hãy bấm Play video trước)
> ```
>
> …dù trang **đang phát video bình thường**. Triệu chứng này KHÔNG có nghĩa là "không có
> video" — nó gần như luôn có nghĩa là **popup không nói chuyện được với frame chứa video**.
>
> Guard tự động: `backend_cpp/tests/test_extension_invariants.py` (10 test).
> Chạy: `python -m pytest backend_cpp/tests/test_extension_invariants.py -q`

---

## 1. Bối cảnh: vì sao phức tạp như vậy

Layout phổ biến của các trang streaming (vlxx, và nhiều site khác):

```html
<!-- top frame: https://vlxx.phd/video/... -->
<div class="video-player">
  <iframe src="https://play.vlstream.net/embed/xxxx/s1"></iframe>   <!-- cross-origin -->
</div>
```

**Top frame KHÔNG chứa thẻ `<video>` nào.** Video nằm hoàn toàn trong **iframe cross-origin**.
Vì vậy `findVideo()` ở top frame luôn trả `null` — điều này là **đúng**, không phải bug.

Hệ quả: extension **buộc phải** nói chuyện được với content script trong iframe đó.

---

## 2. Chuỗi nguyên nhân (mỗi mắt xích từng bị đứt)

```
Popup bấm START
   │
   ├─(A) Có host permission cho origin của iframe không?
   │      KHÔNG → iframe VÔ HÌNH với mọi API (executeScript / webNavigation / sendMessage)
   │      ⚠️ activeTab KHÔNG phủ iframe con — chỉ phủ document top-level
   │
   ├─(B) Content script có thực sự chạy trong iframe không?
   │      KHÔNG → mọi kênh messaging im lặng; executeScript trả biến toàn cục = null
   │      ⚠️ declarative content script KHÔNG được chèn hồi tố vào document đã load trước khi cấp quyền
   │
   ├─(C) Popup có tìm ra frame chứa video không?
   │      Dùng webNavigation.getAllFrames + tabs.sendMessage({frameId})
   │      ⚠️ scripting.executeScript({allFrames:true}) KHÔNG đáng tin với iframe cross-origin
   │
   └─(D) START có tới đúng 1 frame không?
          ⚠️ broadcast chỉ trả về response ĐẦU TIÊN → frame không có video phải IM LẶNG
```

---

## 3. Invariants (khớp 1-1 với các test)

### INV-1 — `<all_urls>` PHẢI nằm trong `optional_host_permissions`

```json
"optional_host_permissions": [ "<all_urls>" ]
```

- Để ở `optional_host_permissions` thì popup mới gọi được `permissions.request()`.
- **KHÔNG** để `<all_urls>` ở `host_permissions` nữa; **KHÔNG** để ở cả hai nơi (lỗi manifest).
- `host_permissions` chỉ giữ origin backend cụ thể (`wss://localhost:8765/*`, …).
- **Vì sao:** `activeTab` chỉ phủ document top-level của tab đang hoạt động. Không có host
  permission cho `play.vlstream.net` thì iframe **không tồn tại** với extension: cả
  `scripting.executeScript({allFrames:true})` lẫn `webNavigation.getAllFrames()` đều không thấy,
  và `tabs.sendMessage` không tới được. Không có đường vòng.

### INV-2 — Phải có permission `scripting` và `webNavigation`

`scripting` cho đường tự chèn content script; `webNavigation.getAllFrames` là cách **duy nhất
đáng tin** để liệt kê frame (kể cả iframe cross-origin).

### INV-3 — `content_scripts[0].all_frames` PHẢI là `true`

Và `matches` phải chứa `<all_urls>`. Không có `all_frames` thì iframe không bao giờ được chèn.

### INV-4 — `ensureHostAccess()` phải là `await` ĐẦU TIÊN trong handler click START

```js
btnStart.addEventListener("click", async () => {
  btnStart.disabled = true;
  if (hasHostAccess !== true) {
    const granted = await ensureHostAccess();   // ← await ĐẦU TIÊN, trước mọi thứ khác
    ...
  }
  const tab = await fetchActiveTab();           // ← await sau mới tới
```

**Vì sao:** Firefox chỉ cho hiện prompt cấp quyền **trong user-gesture handler**. Bất kỳ `await`
nào chạy trước đó sẽ "tiêu thụ" gesture và prompt bị từ chối.

### INV-5 — Popup PHẢI tự chèn được content script

`injectContentScripts(tab)` gọi `scripting.executeScript({target:{tabId, allFrames:true}, files: CONTENT_SCRIPT_FILES})`.

**Vì sao:** declarative content script chỉ được chèn vào document được load **trong khi**
extension đã có quyền. Trang mở trước khi cấp quyền → **không frame nào có content script** →
mọi kênh im lặng. Tự chèn giúp extension **tự phục hồi** thay vì bắt người dùng F5.

**An toàn khi chèn lặp:** `content-script.js` thoát ngay nếu `window.__bsContentScriptLoaded`
đã được set; các file còn lại chỉ định nghĩa global.

### INV-6 — `CONTENT_SCRIPT_FILES` PHẢI khớp `manifest.json` (đúng thứ tự)

Hai danh sách này lệch nhau sẽ làm đường tự chèn nạp thiếu/sai file. Test sẽ fail nếu lệch.

### INV-7 — Frame không có video PHẢI im lặng khi nhận broadcast START

```js
if (!captureOwnerToken && !findVideo()) {
  return false;      // KHÔNG gọi sendResponse
}
```

**Vì sao:** `tabs.sendMessage` **không kèm frameId** chỉ giao về **response ĐẦU TIÊN**. Nếu
frame top (không video) trả lời `"No video found"` trước, nó sẽ **che mất** frame thật sự start.

### INV-8 — Kênh messaging theo từng frame PHẢI tồn tại ở cả hai phía

- Popup: `webNavigation.getAllFrames` → `sendToFrame(tabId, frameId, msg)`
- Content script: handler cho `DISCOVER_CAPTURE`, `RELEASE_CAPTURE_OWNER`; `START_TRANSLATION`
  nhận `__ownerToken` và **tự claim** token đó.

Đây là kênh duy nhất hoạt động chắc chắn với iframe cross-origin.

### INV-9 — Backend admission guard PHẢI là "newest-wins"

`config.ws.max_sessions = 1` + `_supersede_excess_sessions()` trong `ws/ws_handler.py`.

**Vì sao:** policy "từ chối kết nối mới" biến **một session cũ còn sót** (tab cũ chưa đóng,
hoặc bridge trong service worker giữ WebSocket) thành **khoá vĩnh viễn** — mọi lần START sau
đó đều bị đóng với close code 1013. Phải **đóng session CŨ NHẤT** để nhường chỗ.

---

## 4. Gỡ lỗi: đọc log theo thứ tự này

Mở console của popup (chuột phải vào popup → **Inspect**). Log xuất hiện theo đúng thứ tự chẩn đoán:

| Log | Ý nghĩa | Nếu sai |
|---|---|---|
| `[Popup] host permission <all_urls>:` | phải là `true` | INV-1/INV-4 |
| `[Popup] frames in tab: [...]` | iframe player **phải có mặt** trong danh sách | INV-1 |
| `[Popup] frame candidates (messaging): [...]` | iframe phải có `"hasVideo":true` | INV-3/INV-5 |
| `[Popup] no frame answered -- content scripts are missing, injecting` | đường tự phục hồi | INV-5 |
| `[Popup] start outcome: frame-targeted <id> [{"success":true}]` | kết quả đúng | INV-7/INV-8 |

Log tốt trên trang dạng iframe:

```
[Popup] frame candidates (messaging): [
  {"frameId":0,"hasVideo":false,"isTop":true,"score":-1,"url":"https://vlxx.phd/video/..."},
  {"frameId":21474836487,"hasVideo":true,"isTop":false,"score":3143600,"url":"https://play.vlstream.net/embed/..."}
]
[Popup] start outcome: frame-targeted 21474836487 [{"success":true}]
```

**Bảng chẩn đoán nhanh:**

| Triệu chứng | Nguyên nhân | Invariant |
|---|---|---|
| `frame candidates` **rỗng**, `frames in tab` chỉ có frame 0 | thiếu host permission cho origin iframe | INV-1 |
| `frames in tab` có iframe nhưng `frame candidates` rỗng | content script chưa được chèn | INV-5 |
| `capture candidates` có iframe nhưng `hasVideo:false` | content script chạy nhưng `findVideo()` không thấy video (shadow DOM sâu / canvas / DRM) | — |
| `error":"Not the designated capture owner"` | frame nhận START nhưng không có token/fallback | INV-7/INV-8 |
| `"WebSocket connection closed (code: 1013)"` | backend từ chối session | INV-9 |

---

## 5. Lưu ý về `manifest.json`

JSON **không hỗ trợ comment**, nên không thể ghi chú trực tiếp trong file. Vì vậy:

- Mọi invariant liên quan manifest nằm trong file này và trong `test_extension_invariants.py`.
- Nếu cần sửa `manifest.json`, **chạy lại test** trước khi commit:

```powershell
python -m pytest backend_cpp/tests/test_extension_invariants.py -q
python backend_cpp/scripts/verify_all.py
node --check extension_firefox/popup/popup.js
node --check extension_firefox/content/content-script.js
```

---

## 6. Sau khi sửa extension

1. **Reload extension**: `about:debugging#/runtime/this-firefox` → **Reload**
   (bắt buộc nếu đổi `manifest.json`).
2. Mở trang có video trong iframe → bấm icon → **START**.
3. Nếu popup báo thiếu quyền → đồng ý prompt (hoặc bật thủ công trong
   `about:addons` → **Bilingual Subtitle** → **Quyền** → *Truy cập dữ liệu của bạn trên mọi website*).
