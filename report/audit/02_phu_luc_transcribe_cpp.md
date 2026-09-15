# PHỤ LỤC 02 — AUDIT CHI TIẾT `external/transcribe.cpp/**` (native C++ / ggml)

> Phụ lục của `report/audit/00_BAO_CAO_AUDIT_HIEU_NANG.md`
> Phạm vi: `external/transcribe.cpp` (engine C++ trên ggml) + binding ctypes đã cài (`site-packages/transcribe_cpp`)
> Bối cảnh: pipeline **latency-critical** — `session.run()` được gọi lặp lại trên một buffer audio **đang lớn dần** mỗi ~350 ms (`backend/asr/engine.py:296, 358-363, 381`)
> **Read-only. Không sửa file nào.**
> `[VERIFIED]` = đã đọc code; `[SUSPECTED]` = suy luận.

---

## 0. Tóm tắt điều hành

| # | Phát hiện | Mức độ | Tác động |
| --- | --- | --- | --- |
| **A** | Không tái sử dụng prefix/state giữa các poll: mel + encoder + prefill **đầy đủ** được tính lại trên buffer đang lớn dần mỗi 350 ms ⇒ **O(N²)** | 🔴 **CRITICAL** | Latency E2E tăng không giới hạn theo độ dài câu |
| **B** | Threadpool CPU của ggml là **dùng-một-lần cho mỗi `ggml_graph_compute`** (không có pool bền vững cho bất kỳ family nào trừ parakeet) ⇒ ~3 OS thread create/join mỗi graph × ~32 graph mỗi run | 🔴 **CRITICAL** (CPU) / 🟠 High (GPU) | Đốt CPU + µs–ms khởi động mỗi graph, mỗi poll |
| **C** | Scheduler **và** compute context của ggml bị destroy + dựng lại mỗi lần `run()` offline; qwen3_asr còn `ggml_init` **2 lần** mỗi run | 🟠 **HIGH** | Chính thư viện tự đo: **~1 ms (Metal) đến ~10 ms (CPU) mỗi run** (`transcribe-session.h:305-307`) |
| **D** | Công việc host O(T²) + H2D O(T²) mỗi run: causal mask `T_prompt²`, `build_cu_seqlens_mask`, `build_sinusoid_pe`, mel re-pack | 🟠 **HIGH** | ~1.3 MB build host + 7 upload blocking mỗi run |
| **E** | **Output encoder bị D2H về host rồi upload H2D lại** trong cùng một run (`qwen3_asr/model.cpp:709` → `:802`) | 🟠 **HIGH** (GPU) | Round trip vô nghĩa + 2 pipeline sync mỗi run |
| **F** | Python copy **toàn bộ buffer PCM đang lớn dần** mỗi `run()` (`from_buffer_copy`, `__init__.py:466`), **trước khi** vào C++ | 🟠 **HIGH** | 1.92 MB memcpy @30 s, mỗi 350 ms |
| **G** | Front-end mel spawn `stft_threads-1` `std::thread` **mới mỗi lần `compute()`** (`transcribe-mel.cpp:493-507`) | 🟠 **HIGH** | Fan-out thread độc lập thứ hai mỗi run, cộng thêm vào B |
| **H** | Không có pinned host memory ở đâu; mọi `set/get` CUDA là `cudaMemcpyAsync` + **`cudaStreamSynchronize` ngay lập tức** (`ggml-cuda.cu:786-787, 794-795`) | 🟡 Medium | Không overlap transfer/compute; H2D/D2H blocking |
| **I** | CUDA graphs **TẮT** trong cả hai build checked-in; option được ghi chú "llama.cpp only" | 🟡 Medium | Sàn kernel-launch mỗi token + full sched sync |
| **J** | Front-end Kaldi-fbank (nemotron / funasr_nano, sensevoice) **hoàn toàn đơn luồng** | 🟡 Medium | Front-end tuần tự cho câu 30 s |
| **K** | `TRANSCRIBE_PERF_DEBUG` tự ghi nhãn `enc_build` = "(graph + sched + uploads)" — thư viện **biết** đây là chi phí cố định mỗi run | ℹ️ Info | Xác nhận C/D là overhead đã biết, chưa sửa |
| **L** | API batch (`transcribe_run_batch` + `run_batch` trong Python) **tồn tại nhưng không được dùng** | 🟡 Medium (cơ hội) | ~2× throughput trên GPU nhàn rỗi cho multi-session |

### Trả lời trực tiếp các câu hỏi đặt ra

* Patch threadpool oversubscription: **ĐÃ ĐƯỢC APPLY** (không chỉ nằm trong `patches/`).
* Thread: **được tạo lại, không tái sử dụng**, cho mọi graph compute không phải parakeet.
* `sched_yield` spin-wait: **không có** spin-wait `sched_yield` vô điều kiện do patch thêm vào — patch *giảm* spin bằng cách yield, và chỉ sau 4096 lần pause.
* Busy-wait thật: **CÓ** — ggml `poll=50` ⇒ 6.5 triệu vòng `_mm_pause`.
* Global lock serialize session đồng thời: **KHÔNG có** ở tầng thư viện (ràng buộc multi-session là về *resource*, không phải *lock* — xem §6, §7 và mục C2 của báo cáo chính).

---

## 1. Thread pool / oversubscription

### 1a. Patch ĐÃ ĐƯỢC APPLY — `[VERIFIED]`

`patches/ggml/0001-fix-threadpool-oversubscription.patch` (63 dòng) **không** chỉ nằm trong `patches/`. Cả ba hunk đều sống trong cây ggml checked-out:

* Nhánh MSVC `pause` — `ggml/src/ggml-cpu/ggml-cpu.c:527-534`:
```c
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
// MSVC defines _M_X64 / _M_IX86 (NOT __x86_64__), so without this branch the
// spin-wait relax compiled to an empty no-op on MSVC ...
static inline void ggml_thread_cpu_relax(void) {
    YieldProcessor();
}
```
* `ggml_thread_yield()` + ngân sách spin — `ggml-cpu.c:548-564` (`SwitchToThread()` trên `_WIN32`, ngược lại `sched_yield()`; `#define GGML_BARRIER_SPIN_BEFORE_YIELD 4096`).
* Yield fallback trong `ggml_barrier` — `ggml-cpu.c:624-636`:
```c
int spin = 0;
while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
    if (++spin < GGML_BARRIER_SPIN_BEFORE_YIELD) {
        ggml_thread_cpu_relax();
    } else {
        ggml_thread_yield();
        spin = 0;
    }
}
```
**Đánh giá:** fix đúng cho deadlock oversubscription. Ở điều kiện khoẻ mạnh, barrier resolve trong **ít hơn** 4096 lần pause, nên syscall `SwitchToThread()` không nằm trên đường bình thường. Nó chỉ thành syscall mỗi 4096 pause nếu một worker bị preempt (khả dĩ ở đây, vì Python thread copy cả buffer giữa các poll) — **Low**.

### 1b. 🔴 CRITICAL — Không có threadpool bền vững ⇒ threadpool dùng-một-lần mỗi graph compute — `[VERIFIED]`

Thư viện transcribe gắn `n_threads` vào CPU backend nhưng **không bao giờ gắn `ggml_threadpool`**, trừ đúng một chỗ.

* `configure_sched_n_threads` chỉ resolve proc đặt **số** thread — `src/transcribe-batch-util.cpp:69-88`:
```cpp
auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
    ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
if (fn != nullptr) {
    fn(be, n_threads);
}
```
* Nó rơi vào `ggml_backend_cpu_set_n_threads`, chỉ set `ctx->n_threads` — `ggml/src/ggml-cpu/ggml-cpu.cpp:253-258`. `ctx->threadpool` vẫn `NULL` (set tại `ggml-cpu.cpp:226-227`).
* `ggml_backend_cpu_graph_compute` chuyển tiếp `NULL` đó — `ggml-cpu.cpp:170-190`.
* `ggml_graph_compute` sau đó **dựng và phá huỷ cả một pool** — `ggml-cpu.c:3400-3416` và `:3458-3464`:
```c
bool disposable_threadpool = false;
if (threadpool == NULL) {
    disposable_threadpool = true;
    struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
    threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
}
...
if (disposable_threadpool) { ggml_threadpool_free(threadpool); }
```
* `ggml_threadpool_new_impl` tạo `n_threads - 1` OS thread cộng sync object — `ggml-cpu.c:3357-3370`:
```c
ggml_mutex_init(&threadpool->mutex);
ggml_cond_init(&threadpool->cond);
for (int j = 1; j < tpp->n_threads; j++) {
    ...
    int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
```
* Đường này **đang sống**: OpenMP **TẮT** (xác nhận lại: `GGML_OPENMP:BOOL=OFF` trong cả `build/CMakeCache.txt` và `build_x64/CMakeCache.txt`), nên `ggml-cpu.c:3418-3441` (`#pragma omp parallel`) bị compile out và `:3442-3452` (kickoff + worker thread) là đường chạy thật.
* Thư viện **biết** về fix này và áp dụng ở **đúng một chỗ** — `src/arch/parakeet/decoder.cpp:505-521`:
```cpp
// Persistent threadpool so the per-step graph_compute reuses workers instead
// of spawning a transient pool each call. ggml's default hybrid-polling
// (poll=50) keeps the workers hot between the sub-millisecond per-step
// dispatches; poll=0 (park-immediately) was measured to regress pred to
// 70-110 ms because parked workers are slow to reschedule.
...
    // If unresolved (non-CPU exotic backend), g.tp stays null and each
    // graph_compute spawns a transient pool — correct, just less tuned.
```
  Grep `set_threadpool|threadpool_new|threadpool_free` trên `src/` chỉ trả **các hit parakeet này**. qwen3_asr, whisper, cohere, canary, sensevoice, voxtral, moonshine, funasr_nano — **tất cả đều dùng-một-lần**.

**Khuếch đại mỗi poll (qwen3-asr-1.7b, model mặc định):** một `run()` thực hiện **ba họ graph** riêng biệt, mỗi cái là một `ggml_backend_sched_graph_compute` riêng:
* encoder — `src/arch/qwen3_asr/model.cpp:678`
* prefill — `src/arch/qwen3_asr/model.cpp:828`
* step graph, **một lần mỗi token sinh ra** — `src/arch/qwen3_asr/model.cpp:953`

Với `n_threads = 4` (`backend/asr/engine.py:87` `threads: int = 4`, áp dụng ở `:180`), ~32 graph compute mỗi run ⇒ **~96 OS thread create/join + ~64 SRWLock/condvar init/destroy mỗi poll 350 ms** trên build CPU-only. (Trên CUDA/Vulkan chỉ các split rơi về CPU backend phải chịu; số nhỏ hơn nhưng khác 0.)
Ước lượng: thread create trên Windows ≈ 50–150 µs ⇒ **~5–15 ms CPU-side mỗi run, thuần thread churn**.

### 1c. Busy-wait: `poll=50` ⇒ 6.5 triệu vòng pause — `[VERIFIED]`

`ggml/src/ggml.c:8014-8020` đặt default, và `ggml-cpu.c:3207-3219` đốt nó:
```c
const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;   // poll = 50  => 6,553,600
for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
    ggml_thread_cpu_relax();
}
```
Worker nhàn rỗi spin-pause tới **6.553.600 lần** trước khi park trên condvar (`ggml-cpu.c:3230-3236`). Với pool **dùng-một-lần**, worker không nán lại giữa các graph, nên phơi nhiễm bị chặn trong các `ggml_barrier` nội graph — nhưng nó vẫn có nghĩa worker đốt CPU trong `_mm_pause` khi main thread còn đang giữa các barrier. Nhân với 32 graph/run × 3 poll/s. **Medium**, và đây là hành vi upstream ggml, không phải regression của transcribe.

### 1d. Kết luận §1

| Câu hỏi | Trả lời |
| --- | --- |
| Chi phí create/destroy threadpool mỗi call? | **CÓ** — §1b |
| `sched_yield` spin-wait? | Patch **thêm** yield có ngưỡng vào barrier; **không** có `sched_yield` spin vô điều kiện. Trước patch, nhánh MSVC là `{;}` (relax rỗng) = spin loop thuần no-op, **tệ hơn**; patch đã sửa. **Không phải finding.** |
| Mutex thêm mỗi compute? | `ggml_graph_compute_kickoff` lấy `threadpool->mutex` một lần mỗi graph (`ggml-cpu.c:3283`), cộng init/destroy từ pool dùng-một-lần |

---

## 2. Cấp phát mỗi call & tái sử dụng buffer

### 2a. 🟠 HIGH — Scheduler + compute context bị giải phóng sau **mỗi** run offline — `[VERIFIED]`

`src/transcribe.cpp:2071-2084` định nghĩa guard; nó được arm tại điểm commit của cả hai entry point offline:
* `transcribe_run_impl` — `src/transcribe.cpp:2251-2253`
* `transcribe_run_batch_impl` — `src/transcribe.cpp:2366-2368`

`release_scratch()` → `release_compute_scratch(sched, compute_ctx)` **free cả hai** — `src/transcribe-model.cpp:20-23` và `src/transcribe-backend.cpp:152-159`:
```cpp
void release_compute_scratch(ggml_backend_sched_t & sched, struct ggml_context *& compute_ctx) noexcept {
    safe_sched_free(sched);
    sched = nullptr;
    if (compute_ctx != nullptr) { ggml_free(compute_ctx); compute_ctx = nullptr; }
}
```
**Thư viện tự công bố giá phải trả** — `src/transcribe-session.h:305-307`:
> *"The measured recreation cost is about 1 ms per run on Metal and up to about 10 ms on CPU for families that reserve a worst-case decoder workspace each run, such as Canary and Cohere."*

và `src/transcribe-session.h:314-315`:
> *"Streaming entry points do not invoke this hook. A streaming session keeps its scheduler until a later offline run or session destruction."*

Vì qwen3_asr **không có** streaming hook (§4), engine Python **luôn** đi đường offline, nên **trả chi phí này mỗi poll**. Lưu ý qwen3_asr **không** override `on_scratch_released()` (chỉ whisper/cohere/parakeet/gigaam/medasr/canary có).

Hệ quả trong `qwen3_asr::run`: `if (cc->sched == nullptr) cc->sched = ggml_backend_sched_new(...)` (`src/arch/qwen3_asr/model.cpp:634-641`) **gần như luôn đúng**, tức là một `ggml_backend_sched` mới với `graph_size=16384` mỗi run.

### 2b. 🟠 HIGH — `qwen3_asr run()` dựng lại mọi thứ, hai lần mỗi run — `[VERIFIED]`

Danh sách chi phí đầy đủ mỗi `run()` (`src/arch/qwen3_asr/model.cpp`):

| Bước | Dòng | Chi phí mỗi call |
| --- | --- | --- |
| Mel trên toàn buffer | 569-570 | O(N) |
| `ggml_free(compute_ctx)` + `ggml_init(16 MB, no_alloc)` | 610-625 | alloc + metadata |
| `build_encoder_graph` | 628 | O(n_nodes) |
| `ggml_backend_sched_new` | 634-641 | sched mới |
| `ggml_backend_sched_reset` + `alloc_graph` | 642-647 | gallocr pass |
| `pack_mel_chunks` vào `std::vector<float> mel_batched` **mới** | 650-651 | O(N) alloc+copy |
| `build_sinusoid_pe` (vector mới) | 656 | O(T) |
| `build_cu_seqlens_mask` (vector mới) | 662 | **O(T²)** build |
| Upload mask (convert f32→f16 hoặc f32) | 663-671 | O(T²) H2D |
| `configure_sched_n_threads` (tra registry string mỗi backend) | 674 | nhỏ |
| `ggml_backend_sched_graph_compute` (encoder) | 678 | compute |
| `ggml_backend_tensor_get(eb.out, cc->enc_host...)` | 707-710 | **D2H** |
| `build_prompt_tokens` | 720 | O(T_enc) |
| KV cache zeroed cho prefill mới | 770-775 | O(kv bytes) |
| `build_prefill_graph` | 782-784 | O(n_nodes) |
| sched reset + alloc + 3 upload | 790-809 | gallocr + H2D |
| **Causal mask `T_prompt × T_prompt` f16 dựng bằng vòng lặp lồng** | 812-824 | **O(T²)** build + H2D |
| `ggml_backend_sched_graph_compute` (prefill) | 828 | compute |
| `ggml_backend_tensor_get(pb.out, logits[vocab])` | 852-853 | **D2H** toàn vocab |
| `ggml_free(compute_ctx)` + `ggml_init(8 MB)` **lần nữa** | 889-905 | context alloc thứ hai |
| `build_step_graph` + sched reset/alloc | 906-917 | graph + gallocr |
| Mỗi token: 3 × `tensor_set` + `tensor_get` | 934-948, 961 | 4 transfer blocking/token |

Causal mask — `src/arch/qwen3_asr/model.cpp:812-824`:
```cpp
std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * T_prompt, mask_neg_inf);
for (int r = 0; r < T_prompt; ++r) {
    for (int c = 0; c <= r; ++c) {
        mask[static_cast<size_t>(r) * T_prompt + c] = mask_zero;
    }
}
ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
```
Với `T_prompt ≈ 800`: **1.28 MB cấp phát, 320k vòng lặp trong, 1.28 MB H2D blocking — mỗi poll**.

**Làm ĐÚNG:** decode step graph được dựng **một lần** và dùng lại cho mọi token — `src/arch/qwen3_asr/model.cpp:877-918`:
```cpp
// Build the step graph ONCE and reuse every step, sized for the actual workload ...
StepBuild sb = build_step_graph(cc->compute_ctx, ...);
```
và `src/causal_lm/causal_lm.cpp:421-426` (*"graph stays static (zero per-step rebuild)"*). ✅

### 2c. Buffer front-end mel bị free (nhả capacity) mỗi call — `[VERIFIED]`

`src/transcribe-mel.cpp:700-702`:
```cpp
std::vector<double>().swap(padded);
std::vector<float>().swap(padded_f32);
std::vector<float>().swap(window_f32);
```
`swap` với vector rỗng **nhả capacity**, nên `compute()` kế tiếp phải `malloc` và page-fault lại. Cộng với cấp phát mới ở `:422` (`std::vector<double> emph(n_samples)` — zero-initialised), `:441/:444` (`padded.resize(...)` — zero-initialised), `:478` (`log_mel`), `:561` (`power`), `:759/:788/:827` (`out_mel.resize`), một câu 30 s **churn vài MB heap mỗi poll**. **Medium** — page-fault + memset traffic chứ không phải leak.

### 2d. Copy toàn buffer phía Python mỗi call — `[VERIFIED]`

`site-packages/transcribe_cpp/__init__.py:466`:
```python
return (ctypes.c_float * n).from_buffer_copy(floats), n
```
`from_buffer_copy` cấp phát ctypes array mới và copy **toàn bộ buffer PCM** mỗi `session.run()`. Ở 30 s / 16 kHz = 480.000 float = **1.92 MB memcpy + heap alloc mỗi 350 ms**, trước khi bất kỳ code C++ nào chạy. Docstring module ghi chú call dài nhả GIL (`__init__.py:17`), nên đây là chi phí hot duy nhất phía Python.

---

## 3. Mel spectrogram / trích xuất đặc trưng

### 3a. 🟠 HIGH — Tính lại trên toàn câu đang lớn dần mỗi poll ⇒ O(N²)

`MelFrontend::compute` **luôn bắt đầu từ `pcm[0]`** — `src/transcribe-mel.cpp:390-398` (pre-emphasis trên toàn `n_samples`), reflect pad (`:399-413`), rồi STFT trên `n_frames = n_samples/hop + 1` (`:341`, `:361`). **Không có frame cache, không có cursor "đã phát", không có entry point mel từng phần** trên đường offline.

Kết hợp với vòng poll Python — `backend/asr/engine.py:89` (`poll_interval_ms: int = 350`), `:380-381`, `:296` — và việc qwen3_asr không có streaming hook (`src/arch/qwen3_asr/model.cpp:1716-1719`), **mỗi poll làm lại mel + encoder 1.7B đầy đủ + prefill đầy đủ + greedy decode đầy đủ trên toàn bộ câu tới thời điểm đó**. Tổng công việc qua T poll là **Θ(T²) tính theo frame**.

→ **Đây là xác nhận ở tầng C++ cho F-01 trong báo cáo chính**, và là bằng chứng mạnh hơn mô hình phân tích: không chỉ ASR model bị lặp, mà cả **mel front-end** cũng bị lặp bậc hai.

### 3b. FFT plan / window / filterbank — phần lớn đã cache `[VERIFIED, TỐT]`

Cache một lần trong constructor, vì `MelFrontend` được dựng lúc load model (`src/arch/qwen3_asr/model.cpp:562` dùng `cm->mel`, dựng trong `load`):
* Hann window: `src/transcribe-mel.cpp:286-295` (hoặc do checkpoint cung cấp, `:280-286`)
* Slaney mel filterbank: `:297-303` (hoặc checkpoint, `:298-299`)
* mixed-radix LUT sin/cos: `:308-318` (`cos_lut_.resize(cfg.n_fft)`…, sized để **không có `sin/cos` sống trong butterfly**)

Header ghi rõ thiết kế — `src/transcribe-mel.h:14-16`: *"Construction is one-shot: the constructor precomputes the hann window ... and the Slaney-normalized mel filterbank. compute() is then a function of (config, audio) only."* ✅

**Ngoại lệ (chỉ macOS):** `vDSP_create_fftsetupD` được tạo **bên trong** `compute()` và destroy trước khi return — `src/transcribe-mel.cpp:578` và `:605`. FFT-plan setup mỗi call. Windows/Linux không bị (dùng `fft_radix2` ở `:616` hoặc worker mixed-radix hợp nhất ở `:523`). **Low**.

### 3c. 🟠 HIGH — Fan-out `std::thread` mỗi call trong front-end mel — `[VERIFIED]`

`src/transcribe-mel.cpp:493-507`:
```cpp
auto run_threaded = [&](auto && worker) {
    if (stft_threads <= 1) { worker(0); return; }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(stft_threads - 1));
    for (int tid = 1; tid < stft_threads; ++tid) { pool.emplace_back(worker, tid); }
    worker(0);
    for (auto & th : pool) { th.join(); }
};
```
Thread **mới, join, mỗi `compute()`**. Với `stft_threads = min(n_threads, n_frames)` (`:482-488`) và `n_threads = 4`, một poll 350 ms (≈35 mel frame) tạo **3 thread để làm 35 × FFT 400 điểm** — tức thread creation (~50–150 µs mỗi cái trên Windows) là phần **không nhỏ** của công việc STFT. Đây là **fan-out độc lập thứ hai**, cộng lên trên pool ggml ở §1b. Không pool nào được tái sử dụng.

### 3d. Medium — Front-end kaldi-fbank (nemotron/funasr_nano, sensevoice) hoàn toàn đơn luồng

`KaldiFbankFrontend::compute` — `src/transcribe-kaldi-fbank.cpp:154-260` — **không có** threading (grep `thread` trong file: không có), và cấp phát `frame` / `frame_f` / `power` / `mel` / `out_features` mới mỗi call (`:179-184`, `:260`). Nó chạy chuỗi per-frame tuần tự (DC removal, pre-emphasis, Hamming, pad, FFT, power, mel matmul — `:186-250`) rồi stack LFR tuần tự O(T_lfr × lfr_m × d_input) (`:261-284`).
Được gọi không có tham số thread — `src/arch/funasr_nano/model.cpp:479`:
```cpp
const int T_lfr = cm->frontend->compute(pcm, static_cast<size_t>(n_samples), cc->frontend_buf);
```
Caller của nó biết nó thread-safe (dùng trong `parallel_for_all` cho batch — `src/arch/sensevoice/model.cpp:807`, `src/arch/funasr_nano/model.cpp:1103`), nhưng **đường single-utterance là tuần tự**. Với câu 30 s đây là **nút thắt tuần tự cứng** trên đường latency.

### 3e. Medium — Voxtral Realtime re-STFT toàn bộ cửa sổ 750 frame mỗi decode

`src/arch/voxtral_realtime/model.cpp:1394-1441` dựng sliding window rồi:
```cpp
// ---- 2. Streaming mel over the window. ----
const int64_t t_mel0 = ggml_time_us();
if (auto st = cm->mel->compute(buffer.data(), win_len, cc->mel_buf, mm, mel_n, cc->n_threads);
```
`win_len` phủ encoder sliding window (`enc_sliding_window` = 750 frame, comment ở `:1330-1334`). Một feed 350 ms thêm ~35 frame mới, nên **~95% STFT bị tính lại mỗi decode**. Bounded (không O(N²)) nhưng **~20× mel work dư thừa** trên đường streaming.

**Đối chiếu TỐT:** parakeet/nemotron streaming giữ mel mỗi feed bounded bằng cách trim — `src/arch/parakeet/model.cpp:3025-3037`:
```
// Recompute mel from the sliding PCM buffer (trimmed after each emit
// to keep per-feed mel cost bounded).
```

---

## 4. KV cache / context / incremental state

* **Trong một run: incremental (TỐT).** KV cache được pre-allocate lúc session init (`src/arch/qwen3_asr/model.cpp:311-321`, `n_ctx=2048`), grow theo nhu cầu (`:743-775`), ghi một lần bởi prefill graph (`:782-835`, `cc->kv_cache.n = T_prompt` ở `:834`), rồi đọc/ghi mỗi step bởi step graph **tĩnh** (`:906-917`, `src/causal_lm/causal_lm.cpp:421-426`). Buffer mask mỗi step được dùng lại host-side qua các step (`:920-948`). **Phần này được thiết kế tốt.**
* **Qua các run: không giữ gì.** Vì toàn bộ câu được transcribe lại với prompt khác, cache bị wipe và prefill làm lại mỗi poll — `src/arch/qwen3_asr/model.cpp:768-775`:
```cpp
} else {
    // Clear stale positions for a fresh prefill.
    if (cc->kv_cache.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
    }
    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;
}
```
Grep `n_past|prefix|retain|state` trong `src/causal_lm/` chỉ thấy `n_past` như **cursor decode trong-run** (`causal_lm.cpp:185-199`, `:914-966`) và `pick_kv_cache_context` (`:166-183`) để sizing. **Không có API công khai hay nội bộ cho "mở rộng prefix trước đó và giữ KV cache".** Đây là **khoảng trống cấu trúc lớn nhất** cho workload đang mô tả.
* **Các family streaming CÓ incremental state thật:**
  * Voxtral Realtime: encoder KV ring bounded + decoder sliding-window ring — `voxtral_realtime/model.cpp:1881-1947` (`k_enc_ring_ctx`, `dec_ring = dec_sliding_window`), ring write ở `:1772-1779`.
  * Parakeet/nemotron: cache-aware streaming với decoder state bền — `parakeet/model.cpp:383-386` (`stream_dec_state.lstm_state.reset(...)`, `prev_token_id`, `frame_offset`), dùng ở `:2218-2225` và `:2611-2619`; emit loop ở `:3055-3087`.
  * Caveat tăng trưởng đã được ghi: `src/transcribe-session.h:316-318` — *"Moonshine Streaming's decode graph cross-attends over the complete committed stream, so its workspace grows with total stream length."*
* Núm `n_ctx` tồn tại (`src/transcribe-session.h:68-75`, `:39-48`) nhưng đó là **trần**, không phải cơ chế tái sử dụng.

> **Hệ quả cho backend:** `TranscribeEngine._ensure_model_loaded` tạo `loaded_model.session(n_threads=...)` **không truyền `n_ctx`** (`backend/asr/engine.py:180`), nên session dùng default context của model. Engine cũng **không đọc** `session.limits.effective_max_audio_ms` (binding có: `__init__.py:1345-1355`) để biết trần audio hợp lệ ⇒ có thể feed audio vượt trần và bị `OUTPUT_TRUNCATED` mà không hề biết (engine không kiểm tra `transcribe_was_truncated`).

---

## 5. Transfer CPU ↔ GPU

### 5a. Không có pinned host memory — `[VERIFIED]`

Grep `pinned|host_buffer|ggml_backend_dev_buffer_from_host_ptr` trên `src/` **không có site cấp phát nào**. Mọi host buffer là `std::vector` thường (pageable).

### 5b. Mọi upload/download CUDA đều blocking với stream sync ngay lập tức — `[VERIFIED]`

`ggml/src/ggml-cuda/ggml-cuda.cu:782-795`:
```c
static void ggml_backend_cuda_buffer_set_tensor(...) {
    ...
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
static void ggml_backend_cuda_buffer_get_tensor(...) {
    ...
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```
Biến thể 2-D cũng vậy (`:803-805`, `:813-815`), và đường tensor-copy của scheduler cũng vậy (`:827-835`). ⇒ **Không overlap, không staging buffer, không batch transfer.**

### 5c. 🟠 HIGH — Output encoder round trip host vô nghĩa trong cùng một run — `[VERIFIED]`

`src/arch/qwen3_asr/model.cpp:701-710` (D2H):
```cpp
cc->enc_host.resize(static_cast<size_t>(d_enc) * static_cast<size_t>(T_enc));
const int64_t t_d2h_start = ggml_time_us();
ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));
```
và `:800-802` (H2D lại vào prefill graph):
```cpp
ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
ggml_backend_tensor_set(pb.enc_out_in, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));
```
Comment ở `:701-704` biện minh (*"Read encoder output to host for the LM prefill"*) — nhưng tensor encoder **nằm trên cùng backend** với prefill graph, nên đây là **detour device→host→device thuần**: 2 transfer blocking + 2 full pipeline sync + một host allocation `d_enc × T_enc × 4` byte (≈12 MB ở 30 s với `d_enc = 1024`). Nó còn **buộc rebuild** `mel_batched` / `pe` / `mask` giữa hai graph. Trên GPU đây là **thuế latency thật mỗi poll**; trên CPU là một cặp memcpy vô ích.

### 5d. Kiểm kê transfer mỗi run và mỗi token (qwen3_asr)

* Mỗi run, H2D blocking: `mel_batched` (`:652`), `pe` (`:657`), encoder mask (`:668`/`:670`), `input_ids` (`:801`), `enc_out_in` (`:802`), `positions` (`:809`), prefill causal mask (`:823`) = **7 upload blocking** (+ `kv_cache.buffer` clear ở `:771`).
* Mỗi run, D2H blocking: encoder output (`:709`), full-vocab logits (`:853`) = **2**.
* Mỗi token sinh ra: `input_id` (`:934`), `position` (`:936`), `kv_idx` (`:938`), step mask (`:948`) = **4 H2D blocking**, cộng D2H argmax (`:961`) cộng một `ggml_backend_sched_graph_compute` đầy đủ.
* **Mọi graph compute kết thúc bằng hard device sync** — `ggml/src/ggml-backend.cpp:1943-1945`:
```cpp
enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
```
⇒ encoder → prefill → step graph **không bao giờ overlap**, và prefill không thể bắt đầu khi D2H encoder còn đang bay.
* **Copy thêm bên trong scheduler:** mỗi split, input của user bị copy lại vào bản nội bộ và synchronize — `ggml/src/ggml-backend.cpp:1613-1620`:
```cpp
if (input->flags & GGML_TENSOR_FLAG_INPUT) {
    // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
    } else {
        ggml_backend_synchronize(split_backend);
    }
    ggml_backend_tensor_copy(input, input_cpy);
}
```
Trên CUDA đó là event sync + D2D memcpy đồng bộ (`ggml-cuda.cu:827-835`) mỗi input mỗi split. Trên CPU `ggml_backend_synchronize` là NULL op nên rẻ.
* **Có `.cpu()` trên hot path không?** **Không** — D2H luôn là `ggml_backend_tensor_get`. Cái **không cần thiết** là `:709` (§5c).

### 5e. Medium — CUDA graphs bị tắt ⇒ sàn launch/sync mỗi token

`GGML_CUDA_GRAPHS:BOOL=OFF` trong cả hai build. Option được upstream ghi rõ không áp dụng ở đây — `ggml/CMakeLists.txt:209`:
```
option(GGML_CUDA_GRAPHS "ggml: use CUDA graphs (llama.cpp only)" ${GGML_CUDA_GRAPHS_DEFAULT})
```
Với `USE_CUDA_GRAPH` undefined, machinery capture `ggml_cuda_lock` (`ggml-cuda.cu:4180-4286`) là **inert** và mỗi trong ~32 graph compute mỗi run **re-launch chuỗi kernel**. Cộng với hard sync mỗi graph (§5d) và 4 transfer blocking mỗi token, **sàn latency mỗi token là cấu trúc**.

### 5f. Backend thực tế đang chạy — `[VERIFIED tại runtime]`

Binding đã cài là package ctypes thuần Python dlopen provider library — `site-packages/transcribe_cpp/_library.py:8-10`:
> *"`transcribe-cpp-native` (the default — platform wheels bundling CPU+Metal on macOS arm64, **CPU+Vulkan on Linux/Windows**) and `transcribe-cpp-native-cu12` (opt-in CUDA 12 ...)"*

**Kiểm chứng trực tiếp trên máy này** (`python -c "import transcribe_cpp; ..."`):
```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 5060 Ti (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0
                | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
load_backend: loaded Vulkan backend from ...\transcribe_cpp_native\_native\ggml-vulkan.dll
load_backend: loaded CPU backend   from ...\transcribe_cpp_native\_native\ggml-cpu-haswell.dll
```
Ba kết luận:
1. Provider đang dùng là `transcribe_cpp_native` (**CPU + Vulkan**), **không phải** `transcribe_cpp_native_cu12`.
2. **Không có backend CUDA nào được load** ⇒ ASR chạy trên **Vulkan**, trong khi translation (llama.cpp CUDA) và TTS (PyTorch CUDA) chạy trên **CUDA**. Xác nhận **F-19** của báo cáo chính ở mức runtime.
3. CPU backend là tier **`haswell`** (AVX2), **không** phải icelake/alderlake/znver4 — nên mọi op rơi về CPU dùng SIMD thấp hơn khả năng của CPU hiện đại.

Cả hai cây build checked-in là **CPU-only** (`GGML_CUDA:BOOL=OFF`, `GGML_VULKAN:BOOL=OFF`) ⇒ nếu ai build từ source và dùng bản đó, ASR sẽ chạy **thuần CPU**. §5b–5e áp cho cấu hình wheel; §1b (thread churn) áp cho **mọi** cấu hình (build Vulkan/CUDA vẫn đẩy op fallback về CPU backend).

---

## 6. Locking

* **Hot path trong thư viện transcribe: không có lock.** `std::mutex` duy nhất trong `src/transcribe.cpp` là guard idempotency một lần, không phải hot path — `src/transcribe.cpp:895-900`:
```cpp
// Idempotency: loading the same directory twice would re-dlopen and
// re-register every module (duplicate devices). Keyed on the canonical path ...
static std::mutex            s_mutex;
static std::set<std::string> s_loaded_dirs;
std::lock_guard<std::mutex>  lock(s_mutex);
```
* ggml threadpool: một lần acquire mutex mỗi graph compute trong `ggml_graph_compute_kickoff` (`ggml-cpu.c:3283`), một lần trong `ggml_threadpool_free` (`:2754`), một shared-mode lock trong đường park (`:3230`). Với pool dùng-một-lần, đó là **thêm** một SRWLock init + acquire + destroy mỗi compute (§1b).
* ggml scheduler: **không có global lock**; `ggml_backend_sched_split_graph` (`ggml/src/ggml-backend.cpp:1055`) chạy đơn luồng mỗi compute.
* CUDA: `static std::mutex ggml_cuda_lock` (`ggml-cuda.cu:698`) chỉ được giữ **quanh lúc bắt đầu/kết thúc capture CUDA-graph** — `:4281-4283` và `:4191-4194`. **Không** được giữ xuyên suốt compute. Với `GGML_CUDA_GRAPHS=OFF` nó hoàn toàn inert.

**Kết luận: hiện tại không có global lock nào serialize các session đồng thời.** Ràng buộc multi-session là về **resource** (pool dùng-một-lần tranh core, không batching), **không phải** về lock.

⚠️ **Cảnh báo cho build CUDA-graphs trong tương lai:** nếu bật `GGML_CUDA_GRAPHS`, capture sẽ **serialize giữa các session** qua `ggml_cuda_lock`.

⚠️ **Quan trọng — đối chiếu với ràng buộc đã công bố:** `include/transcribe.h:11-20` ghi rõ *"at most one transcribe_run / transcribe_run_batch / active stream may be in flight across ALL sessions of a given model at a time. Sessions share the model's backend instances and some per-family model state, so overlapping runs race (observed: corrupted decodes on CPU, command-buffer failures on Metal)."*
⇒ Việc **không** có global lock trong thư viện **không** có nghĩa là chạy song song an toàn: nó có nghĩa là **thư viện không tự bảo vệ**, và `_infer_lock` (class-level RLock) ở `backend/asr/engine.py:51` là thứ backend **phải** có. Đây chính là lý do F-03 (scale) là giới hạn cứng: muốn N session song song thật phải **load 1 model instance cho mỗi worker** — tốn 2.1 GB VRAM mỗi instance cho Qwen3-ASR-1.7B Q8_0.

---

## 7. Batching

* **API C có hỗ trợ.** `transcribe_run_batch_impl` — `src/transcribe.cpp:2297+`; validate param chung một lần (`:2322-2347`), rồi family fast path (`:2370-2383`) hoặc fallback tuần tự từng utterance (`:2385+`). Helper ở `src/transcribe-batch-util.{h,cpp}`: `pack_pad_channel_major` (`:125-142`), `fill_keypad_mask` (`:144-160`), `fill_valid_frame_mask` (`:162-174`), `decode_batch_slices` (`:176-198`), `run_batched_encdec_step_loop` (`:200-365`).
* **qwen3_asr implement đường batch thật** — `src/arch/qwen3_asr/model.cpp:1714` (`.run_batch = run_batch`), một encoder graph chia sẻ và một batched prefill + step graph; debug print ở `:1696-1701` xác nhận:
```
"qwen3_asr run_batch: n=%d valid=%d max_n_kv=%d steps=%d (all phases batched x%d)\n"
"  enc_pass=%.1fms (mel=%.1f parallel + enc_compute=%.1f, 1 graph)\n"
"  prefill_pass=%.1fms (1 batched graph: build/sched/compute/readback)\n"
```
* **Binding Python phơi ra** — `site-packages/transcribe_cpp/__init__.py:1136` (`def run_batch`), docstring `:1147-1151`: *"Families with a batched compute path process every utterance in a single device dispatch (≈2x throughput on an underused GPU)"*. Primitive batching tầng C cũng có cho KV cache (`kv_init_batched`, `src/causal_lm/causal_lm.cpp:107-164`) và mask (`fill_prefill_chunk_mask`, `:185-199`).
* **Nhưng ASR engine không bao giờ dùng** — grep `run_batch` trên `backend/` chỉ thấy `session.run` (`backend/asr/engine.py:296`) và `session.stream` (`:284`). Với mục tiêu multi-session đã nêu, đây là **cơ hội chưa dùng cụ thể nhất**: batch utterance của N session đồng thời vào một `transcribe_run_batch`.
* **Giới hạn cần biết:** vòng decode batch feed prompt **cùng độ dài** cho mọi row (`src/transcribe-batch-util.cpp:200-213`, `:279-297`), và `decode_batch_slices` vẫn lặp từng utterance cho family không có batched path (`:185-196`). Thêm nữa **không có batched streaming API** (`stream_begin/feed/finalize` là single-session trong vtable `Arch` — ví dụ `src/arch/qwen3_asr/model.cpp:1716-1719`), nên pattern poll trên buffer đang lớn dần hiện tại **không có dạng batched**. Batching là **thắng throughput, không phải thắng latency**: step loop vẫn phát một `ggml_backend_sched_graph_compute` mỗi step bất kể `n`.

---

## 8. Memory

* **Bounded (đã verify TỐT):** streaming raw history được trim tường minh — `src/transcribe.cpp:1205-1207`:
```cpp
session->stream_raw_history.push_back(session->full_text);
while (session->stream_raw_history.size() > agreement_n) {
    session->stream_raw_history.pop_front();
}
```
* `batch_results` tăng theo batch size nhưng được clear mỗi run (`src/transcribe.cpp:2246`, `:2361`).
* **Giữ lại có chủ ý** (đã ghi rõ, không phải leak): `src/transcribe-session.h:308-312` — *"Host-side vectors that scale with input length, such as mel buffers, encoder host copies, and positional banks, deliberately retain their capacity... this hook releases only the ggml scheduler and compute context."*
* **Câu chuyện memory thật là dao động, không phải tăng trưởng:** §2a/§2c giải phóng ggml scratch và mel buffer mỗi run, nên RSS răng cưa và allocator bị hành hạ chứ **không leak**. Đây là trade-off có chủ ý (header dẫn "Handy #2000" — một utterance dài đơn lẻ ghim high-water mark của scheduler — ở `src/transcribe-session.h:303-304`).
* **Cấp phát O(T²) mỗi call:** prefill causal mask của qwen3_asr (`src/arch/qwen3_asr/model.cpp:817`) và `build_cu_seqlens_mask` (`:662`) được dựng lại mỗi run.
* **Shift O(N) mỗi feed:** Voxtral Realtime trim bằng `vector::erase` từ đầu — `src/arch/voxtral_realtime/model.cpp:1429-1432`. Bounded nhưng là memmove mỗi feed; ring buffer sẽ cho O(1). **Low**.
* Không leak session state: `transcribe_session::~transcribe_session` → `release_compute_scratch` (`src/transcribe-model.cpp:16-18`), và teardown đi qua wrapper `transcribe::safe_*` (`src/transcribe-backend.cpp:131-159`) theo lint rule của repo.

---

## 9. Benchmark của chính thư viện đo gì (và bỏ sót gì)

**`tools/transcribe-bench/main.cpp`** đo mỗi vòng: `mel_ms`, `encode_ms`, `decode_ms`, `total_ms` (tổng ba cái) và `wall_ms` (`steady_clock` quanh `transcribe_run`) — `:408-414`; rồi báo min/max/mean và hai RTF — `:443-454`, `:491-492`:
```cpp
// RTF based on wall time (the user-visible number — includes
// uninstrumented run overhead like scheduler alloc, tensor upload,
// thread fan-out). rtf_compute is the phase-timer sum for backward
// compat with v1 consumers, but wall is the number to use for
// perf decisions.
const double rtf_wall_mean = (sample_duration_s > 0.0) ? sample_duration_s / (s_wall.mean_v / 1000.0) : 0.0;
```
Default `iters = 2`, `warmup = 1` (`:45-46`), và chạy trên một WAV **cố định** (`:317-323`).

**`scripts/bench/run.py`** quét ma trận `(variant, quant, sample)`, một subprocess mỗi cell (`:549-570`), sample mặc định `jfk` và `dots` (`:108`) — **clip ngắn** — và aggregate vào `reports/perf/<machine-slug>/` (`:718-735`).

### Hệ quả — harness **về mặt cấu trúc không thể thấy** bất kỳ finding nào ở trên

1. Mỗi vòng lặp lại **cùng một sample độ dài cố định**, nên tăng trưởng O(N²) do re-transcribe buffer đang lớn dần (§3a) **vô hình theo thiết kế**.
2. **Streaming API không bao giờ được bench.** Không có timing `stream.feed()`, không time-to-first-token, không hành vi `min_decode_interval`, không replay buffer đang lớn.
3. Overhead cố định mỗi call — thread fan-out (§1b, §3c), sched/context create+destroy (§2a) — chỉ bị chôn trong `wall_ms − total_ms`, và **hiệu số đó không được tính ở đâu cả** trong `main.cpp` lẫn `run.py`. Chú ý `main.cpp:448-452` gọi tên đúng ba thành phần đó là nội dung của khoảng cách ⇒ maintainer **biết** có overhead nhưng **không có metric** cho nó.
4. Không có bench multi-session / concurrency scaling, dù mục tiêu là multi-stream.
5. Cờ `--threads` tồn tại (`main.cpp:68`, `:174-182`) nhưng ma trận **không bao giờ** thay đổi `n_threads`, nên hiệu ứng pool-size không được đo.

### Instrumentation giàu hơn đã tồn tại nhưng chưa nối vào bench JSON

qwen3_asr in breakdown từng phase dưới `TRANSCRIBE_PERF_DEBUG` — `src/arch/qwen3_asr/model.cpp:1034` (gate) và `:1040-1058`:
```
"  mel              %8.2f ms\n"
"  enc_build        %8.2f ms  (graph + sched + uploads)\n"
"  enc_compute      %8.2f ms\n"
"  enc_d2h          %8.2f ms  (%d floats)\n"
"  prefill_build    %8.2f ms  (kv_init + prompt + graph + sched + uploads)\n"
"  prefill_compute  %8.2f ms  (T_prompt=%d)\n"
"  prefill_logits   %8.2f ms  (readback + argmax, vocab=%d)\n"
"  step_loop        %8.2f ms  (%d steps, %.2f ms/step)\n"
"    tensor_set / compute / tensor_get ...
```
Các nhãn `enc_build = "graph + sched + uploads"` và `prefill_build = "kv_init + prompt + graph + sched + uploads"` là **lời thú nhận của chính thư viện** rằng §2 và §5 là chi phí cố định mỗi run.

> 🎯 **Khuyến nghị đo lường giá trị cao nhất cho dự án:** bật `TRANSCRIBE_PERF_DEBUG=1` và chạy một kịch bản replay `run()` trên buffer **đang lớn dần** ở nhịp cố định, báo cáo `wall_ms` theo độ dài audio. Đây là **metric duy nhất** phơi ra Finding A, và nó biến mô hình phân tích ở §A2 của báo cáo chính thành **số đo thật** mà không cần viết code mới.

---

## 10. Quét TODO / FIXME / PERF / HACK

Grep `TODO|FIXME|HACK|XXX|PERF|slow|hot path` trên `src/**/*.cpp`:

**Ghi chú hiệu năng đáng trích dẫn (`[VERIFIED]`):**
* `src/transcribe-session.h:305-307` — sched recreation *"about 1 ms per run on Metal and up to about 10 ms on CPU for families that reserve a worst-case decoder workspace each run, such as Canary and Cohere."* → hỗ trợ Finding C.
* `src/transcribe-session.h:314-318` — streaming giữ sched; parakeet/voxtral realtime dùng workspace per-chunk bounded; *"Moonshine Streaming's decode graph cross-attends over the complete committed stream, so its workspace grows with total stream length."*
* `src/arch/parakeet/decoder.cpp:505-509` — lý do threadpool bền, **kèm regression đã đo**: *"poll=0 (park-immediately) was measured to regress pred to 70-110 ms because parked workers are slow to reschedule."* → thư viện **đã đo** chi phí park threadpool; §1b là phiên bản **chưa được tổng quát hoá** của fix đó.
* `src/arch/parakeet/decoder.cpp:520-521` — *"If unresolved (non-CPU exotic backend), g.tp stays null and each graph_compute spawns a transient pool — correct, just less tuned."*
* `src/arch/parakeet/encoder.cpp:846-849` — *"On CPU at streaming geometry the flash kernel is dramatically slower (~33 ms/layer vs ~1 ms on a Cortex-A55)."* Trade-off policy có chủ ý, không phải bug.
* `src/arch/gigaam/model.cpp:718-721` — chia sẻ **một** ngân sách thread bounded giữa song song mức-utterance và mức-frame để tránh *"nesting n_threads full mel pools"*. Đường batch **có** ý thức về oversubscription; đường single-utterance (§3c) **không**.
* `src/transcribe-mel.cpp:149-152` — *"Why not vendor pocketfft / KISS FFT: n_fft=512 is small ... Drop in pocketfft only if profiling later shows the need."* → lựa chọn FFT là trade-off đã chấp nhận và ghi rõ.
* `src/arch/qwen3_asr/model.cpp:778-779` — `slice_last` (thủ thuật `inp_out_ids` của llama.cpp) đã áp cho FFN của block cuối, tiết kiệm *"~25 ms"*. **Đã làm.** ✅
* `src/transcribe-batch-util.h:36-46` — `default_n_threads(int cap = 8)` **affinity-aware** (đọc `sched_getaffinity` / `GetProcessAffinityMask`, `src/transcribe-batch-util.cpp:32-67`), đây là lý do patch oversubscription có liên quan.
* `ggml/src/ggml-cpu/ggml-cpu.cpp:136` — `cpu_plan->cgraph = *cgraph; // FIXME: deep copy`. Không trên hot path.
* `ggml/src/ggml-cuda/ggml-cuda.cu:364` — workaround device-scheduling cho iGPU cc121 *"to avoid delays in cuda synchronize calls"* — upstream cũng thừa nhận chi phí synchronize mỗi call.

**Không tồn tại marker `TODO`/`HACK`/`XXX` trong hot path transcribe** (`src/transcribe.cpp`, `src/transcribe-mel.cpp`, `src/arch/qwen3_asr/*`). Mọi annotation perf đều là diagnostic có gate `TRANSCRIBE_PERF_DEBUG`.

---

## 11. Những gì ĐÚNG (nêu ngắn, kèm bằng chứng)

* **Patch oversubscription đã apply và đúng** — `ggml-cpu.c:527-534`, `:548-564`, `:624-636`.
* **Mel window / filterbank / FFT-twiddle LUT dựng một lần mỗi model, không phải mỗi call** — `transcribe-mel.cpp:277-320`; thiết kế ở `transcribe-mel.h:14-16`.
* **Decode step graph dựng một lần và dùng lại cho mọi token** — `qwen3_asr/model.cpp:877-918`; `causal_lm.cpp:421-426`.
* **KV cache cấp phát lúc session init và dùng lại trong run** — `qwen3_asr/model.cpp:311-321`, `:743-835`.
* **`stream_raw_history` có bound** — `transcribe.cpp:1205-1207`.
* **Không có global lock serialize session** — `transcribe.cpp:898-900` chỉ init; `ggml-cuda.cu:4191/4281` chỉ cho CUDA-graph capture và inert với `GGML_CUDA_GRAPHS=OFF`.
* **Family streaming (voxtral realtime, parakeet/nemotron) giữ incremental state bounded và mel per-feed bounded** — `voxtral_realtime/model.cpp:1394-1441,1881-1947`; `parakeet/model.cpp:3025-3087`.
* **Teardown được kiểm soát và không leak theo thiết kế** — `transcribe-backend.cpp:114-159`, được enforce bởi `tests/lint_teardown.cmake`.
* **Nhận diện số CPU là affinity-aware**, đúng prerequisite cho pool sizing — `transcribe-batch-util.cpp:32-67`.

---

## 12. SUSPECTED (chưa verify trực tiếp)

1. **SUSPECTED:** chi phí ms tuyệt đối của threadpool churn (§1b). Đã verify **rằng** pool được create/destroy mỗi graph và **rằng** nó spawn `n_threads-1` thread, nhưng chưa đo wall time. Chuỗi suy luận: ≥32 graph compute mỗi qwen3_asr run × 3 thread, ở ~50–150 µs mỗi Windows thread create ⇒ ~5–15 ms/run.
   **Cách verify:** `transcribe-bench --threads 1` vs `--threads 4` trên cùng sample, so `wall_ms − total_ms`; pool bền sẽ thu hẹp khoảng cách này. Kèm `TRANSCRIBE_PERF_DEBUG=1` in `enc_build`/`prefill_build`, và `t_step_build_once_us` (`qwen3_asr/model.cpp:918`) cô lập build step-graph một lần.
2. **SUSPECTED:** trên CUDA/Vulkan, bao nhiêu split thực sự rơi về CPU backend (⇒ bao nhiêu pool dùng-một-lần mỗi run). Chưa liệt kê supported-op table. **Verify:** `TRANSCRIBE_PERF_DEBUG` + build với `GGML_SCHED_DEBUG=2` in split assignment.
3. **SUSPECTED:** kích thước chính xác của `T_enc`/`T_prompt` cho qwen3 1.7B ở 30 s (chi phối chi phí mask O(T²), §2b/§8). Đã dùng `T_prompt ≈ 800` như ước lượng bậc độ lớn.
4. **SUSPECTED:** liệu backend Vulkan của wheel đang deploy có giữ encoder trên GPU cho qwen3_asr hay không — điều này quyết định §5c là round trip off-device hay same-device. Code path giống nhau, **chi phí** khác nhau.
5. **CHƯA ĐIỀU TRA (ngoài ngân sách, gắn cờ cho parent):** `src/transcribe-tokenizer.cpp` (34 KB, nằm trong vòng decode mỗi token ở một số family), `src/transcribe-unicode.cpp` (32 KB, làm sạch text mỗi run), và lựa chọn kernel trong `src/conformer/`, `src/sanm/`.

---

## 13. Danh sách hành động xếp hạng cho pipeline này

| Ưu tiên | Thay đổi | Loại bỏ | Neo bằng chứng |
| --- | --- | --- | --- |
| **1** | **Thêm prefix/state reuse**: giữ KV cache + encoder output cho prefix đã transcribe và mở rộng, thay vì chạy lại trên toàn buffer đang lớn. Kể cả chưa làm được, hãy **debounce poll lên ≥1 s** và chỉ re-run tại ranh giới segment VAD (tức là wire lại Commit Manager — xem **A2** ở báo cáo chính) | Finding A (bùng nổ O(N²)) | `qwen3_asr/model.cpp:768-775`; không có API reuse ở `src/causal_lm/` |
| **2** | Gắn **persistent CPU threadpool** trong `configure_sched_n_threads` (code parakeet là template sẵn) | Finding B | `transcribe-batch-util.cpp:69-88` vs `parakeet/decoder.cpp:505-519` |
| **3** | Ngừng gọi `release_scratch()` trên đường offline cho session sẽ chạy lại ngay — hoặc tối thiểu chỉ release khi vượt ngưỡng kích thước | Finding C (~1–10 ms/run, thư viện tự đo) | `transcribe.cpp:2251-2253`; `transcribe-session.h:305-307` |
| **4** | Giữ encoder output **on-device** và feed thẳng prefill graph (bỏ round trip `enc_host`) | Finding E | `qwen3_asr/model.cpp:709` → `:802` |
| **5** | Cache causal mask `T_prompt²` + `build_sinusoid_pe` + `build_cu_seqlens_mask` qua các poll khi chỉ đuôi thay đổi | Finding D | `qwen3_asr/model.cpp:812-824`, `:656`, `:662` |
| **6** | Dùng persistent mel thread pool (hoặc gộp STFT vào ngân sách pool ggml như gigaam) | Finding G | `transcribe-mel.cpp:493-507`; đối chiếu `gigaam/model.cpp:718-723` |
| **7** | Tránh `from_buffer_copy` phía Python — truyền ctypes buffer writable đã cấp phát sẵn hoặc view `np.frombuffer` | Finding F | `__init__.py:466` |
| **8** | Batch session đồng thời qua `run_batch` (cho scale multi-session, không phải latency đơn luồng) | Finding L | `__init__.py:1136`; `qwen3_asr/model.cpp:1714` |
| **9** | Thêm chế độ replay buffer-đang-lớn + metric overhead `wall_ms − total_ms` vào `transcribe-bench` | Khoảng trống đo lường §9 | `tools/transcribe-bench/main.cpp:391-415`, `:448-452` |

---

## 14. Bảng nối finding C++ ↔ finding backend (báo cáo chính)

| Finding C++ | Khuếch đại bởi (backend) | Finding chính |
| --- | --- | --- |
| A — O(N²) mel+encoder+prefill | `asr/engine.py:358-363` poll lại toàn bộ câu mỗi 350 ms; `:381` nhịp = inference + 350 ms; Commit Manager không được wire (`:104` dead) nên câu có thể dài tới 50 s | **F-01, F-02, F-09** |
| B — threadpool dùng-một-lần | `n_threads=4` truyền từ `asr/engine.py:180`; không có API nào để giữ pool giữa các `run()` | mới — **F-31** (bổ sung) |
| C — sched/context recreate mỗi run | mỗi poll là một `run()` mới (offline path, vì qwen3_asr không có streaming hook) | mới — **F-32** (bổ sung) |
| D — O(T²) mask + 7 upload | cùng lý do | mới — **F-33** (bổ sung) |
| E — encoder D2H/H2D round trip | cùng lý do; trên wheel Vulkan đây là transfer qua PCIe thật | mới — **F-34** (bổ sung) |
| F — `from_buffer_copy` toàn buffer | trùng với `audio_buffer.get_slice` (`core/audio_buffer.py:130`) + `SpeechNormalizer` (~6 mảng) ⇒ **3 tầng copy toàn buffer mỗi poll** | **F-11, B3-1** |
| G — thread fan-out mỗi `compute()` | cộng vào `_EXECUTOR` (2 thread) + `_VAD_EXECUTOR` (1) + `_TRANS_EXECUTOR` (1) + TTS `to_thread` | **A1-tổng** |
| L — `run_batch` không dùng | `_shared_session` single (`asr/engine.py:48`) | **F-03, C1** |

---

*Phụ lục read-only. Không file nào trong `external/transcribe.cpp` được tạo, sửa hay xoá.*
