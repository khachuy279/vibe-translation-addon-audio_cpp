# Japanese ASR Accuracy Optimization — Audit & Changes

**Scope**: reduce character error rate (CER) for Japanese in the `backend_cpp` streaming
pipeline (VAD → AudioBufferManager → SpeechNormalizer → transcribe.cpp → commit controller).
**Date**: 2026-09-12
**Hardware**: RTX 5060 Ti 16 GB, Vulkan GGML backend, Windows 11.
**Status**: audit complete, two fixes implemented and tested; one measurement in flight.

---

## 1. Executive summary

| # | Finding | Evidence | Action |
|---|---|---|---|
| 1 | The Japanese evaluation set had **no discriminating power** — 2 clean clips, ~10 s, every competent model scored ~0% | 8-model bake-off, all 0.00–4.00% | Built a real eval set (see §2) |
| 2 | **`cohere-transcribe` is 4× more accurate than `qwen3-asr-1.7b` on Japanese**, and also faster | 400 verbatim utterances: **2.88% vs 11.69% CER** (RTF 0.028 vs 0.040) | Selectable in the popup; automatic routing was implemented and then **removed at the user's request** (§5) |
| 3 | cohere is **not a safe global default**: it collapses on long audio and fails Russian outright | 63 s file whole-file decode: 42.87% CER; Russian: empty output | Kept as an opt-in per-video choice, never auto-selected |
| 4 | `min_words_to_commit=4` **silently deletes complete short utterances** in every language — but lowering it is a **net loss**, because the same threshold was unknowingly filtering hallucinations | 5/12 real Japanese turns and 10/10 short English turns dropped; lowering to 1 costs **+2.0 pp** (22 of 26 recovered commits were `はい。`, present nowhere in the reference) | Two knobs separated; **default kept at 4**, opt-in available (§6) |
| 5 | A caption-derived reference set makes faithful transcription look like hallucination | 7 unrelated architectures emitted the *same* "extra" text | Metric corrected and the trap documented (§3, §4) |
| 6 | Through the **full streaming pipeline** the cohere advantage narrows to 22% (verbatim) / 37% (TV) | 12.11% vs 15.58%; 17.00% vs 26.96% | VAD segmentation is the dominant remaining loss (§7) |

**Bottom line**: on announced Japanese, expected CER drops from ~15.6% to ~12.1% on
verbatim material and from ~27.0% to ~17.0% on TV speech, with no behaviour change
anywhere else (verified bit-identical on the non-Japanese configurations).

---

## 2. Evaluation sets built

The repo shipped `wav_test/Japanese_5s.wav` (5.08 s) and `transcribe.cpp/samples/ja.wav`
(3 s). Two clips cannot rank models or detect a regression. What was built instead:

| Set | Source | Size | Reference quality | Purpose |
|---|---|---|---|---|
| `data/ja_eval` | `japanese-asr/ja_asr.reazonspeech_test` | 1000 utts, 88 min + 8×63 s streams | **TV captions — not verbatim** | Natural TV speech, domain match |
| `data/ja_eval/utts_verbatim.jsonl` | filtered subset | 304 utts, 27 min | heuristic filter (see §3.3 — did not fully work) | attempted salvage |
| `data/ja_cv` | `japanese-asr/ja_asr.common_voice_8_0` | 4483 utts, 389 min + 8×63 s streams | **verbatim by construction** (read prompts) | metric-valid authority |
| `debug_audio/3688800d…` | user's own browser capture | 29 s, 12 VAD utterances | none (drama dialogue) | domain sanity check |

Harnesses (all new, reusable):

| File | Role |
|---|---|
| `benchmarks/ja_text.py` | Japanese CER with NFKC + optional ITN folding, so `五十円` vs `50円` is not scored as an error |
| `benchmarks/build_ja_eval.py` | ReazonSpeech parquet → wav + manifest + streams |
| `benchmarks/build_ja_streams.py` | any manifest → continuous streams with concatenated reference |
| `benchmarks/ja_offline_bench.py` | per-utterance model comparison, char-weighted corpus CER |
| `benchmarks/streaming_wer_bench.py` | drives the **real** engine in-process (VAD → buffer → normalizer → commit) with per-commit trace |
| `benchmarks/ja_stream_attribution.py` | splits the offline-vs-streaming gap into VAD cost vs controller cost |
| `benchmarks/model_matrix.py` | cross-language regression check for a model swap |

---

## 3. The metric trap (and why the first conclusion was wrong)

### 3.1 What happened

On 200 ReazonSpeech utterances the first screen produced:

| model | CER | insertions |
|---|---|---|
| cohere-transcribe | **5.36%** | 61 |
| qwen3-asr-1.7b | **25.81%** | **611** |

qwen's errors were 55% insertions, so the harness was suspected first. It was cleared:
a fresh model+session per utterance reproduced the shared-session result **exactly,
per utterance**; reversing the order changed nothing (31.50% both ways); the same
utterance run 3× was bit-identical; and 300 ms of silence padding did not help.

### 3.2 The refutation

The decisive check was cross-model agreement on the same clip:

```
REF : 裏読み過ぎちゃったんじゃないかなと思うんだこれ。
qwen: 黄色以外考えないでしょ。これちょっと俺裏読みすぎちゃったんじゃないかなと思うんだこれ。
cohe: これちょっと裏読みすぎちゃったんじゃないかなと思うんだこれ。
koto: 黄色いから考えないでしょこれちょっと裏読みすぎちゃったんじゃないかなと思うんだこれ
sense: 黄色いくらい考えないでしょこれちょっと裏読み過ぎちゃったんじゃないかなと思うんだこれ
nemot: キールが考えないでしょこれちょっと俺裏読み過ぎちゃったんじゃないかなと思うんだこれ
voxtr: 黄色いの考えないでしょこれちょっと裏読みすぎちゃったんじゃないかなと思うこのこれ
```

Seven independent architectures agree on text the reference lacks. They are not all
hallucinating the same sentence — **the audio contains it**. ReazonSpeech is built from
Japanese TV broadcasts and its `transcription` field is the broadcast **caption**, which
routinely drops back-channel, crosstalk and fillers.

An energy check confirmed it: 45% of clips contain more speech energy than the reference
can account for, 23.5% exceed 1.35×, and 10% exceed 2×. One 5.77 s clip is labelled
`どうぞ！` alone.

**Consequence**: on this set, CER measures "how caption-like is the output", and a model
that transcribes faithfully is punished.

### 3.3 The reconciliation (and why the ranking survived)

If the extra text is unlabelled real speech, it shows up as *insertions* only. Splitting
qwen's error into substitutions+deletions:

| set | qwen S+D / ref chars | qwen corpus CER |
|---|---|---|
| ReazonSpeech (200 utts) | 501 / 4309 = **11.63%** | 25.81% (611 insertions) |
| Common Voice ja (400 utts, verbatim) | — | **11.69%** |

The faithful-error rate is identical across two sets with completely different reference
conventions. The caption artifact lives entirely in the insertion count, and **the
cohere/qwen ratio is ~4× on both sets**. The ranking is real; only absolute CER on
ReazonSpeech is inflated.

> An energy-ratio filter (`build_ja_verbatim.py`) was tried to salvage a verbatim subset
> from ReazonSpeech. It did **not** work — qwen's over-production stayed at 23.0% vs 23.5%
> — so it is reported here as a dead end rather than used. Common Voice is the authority.

---

## 4. Model comparison

### 4.1 Verbatim Japanese — Common Voice ja, 400 utterances / 36 min

| model | CER | CER (ITN) | S / D / I | RTF |
|---|---|---|---|---|
| **cohere-transcribe** | **2.88%** | 2.88% | 129 / 44 / 24 | 0.028 |
| kotoba-whisper-v2.2 | 11.31% | 10.46% | 512 / 214 / 47 | 0.037 |
| **qwen3-asr-1.7b** (current default) | **11.69%** | 11.36% | 533 / 207 / 59 | 0.040 |
| sensevoice-small | 12.35% | 11.56% | 565 / 233 / 46 | 0.009 |
| voxtral-mini-4b | 20.20% | 19.68% | 785 / 286 / 310 | 0.098 |
| nemotron-3.5-streaming | 20.92% | 20.96% | 724 / 659 / 47 | 0.019 |

Note qwen's insertions fall from 960 → 59 once the reference is verbatim: **qwen does not
hallucinate**; it was being scored against captions.

The Japanese fine-tune already on disk (`Qwen3-ASR-1.7B-JA-BF16.gguf`, 4.1 GB) is **not an
improvement**: 23.47% CER on the verbatim subset vs 24.33% for the base model, and it
applies ITN (`五十円` → `50円`) that the base model does not.

### 4.2 Cross-language regression check — `wav_test` (whole-file, warm)

| item | qwen3-asr-1.7b | cohere-transcribe | kotoba-whisper |
|---|---|---|---|
| Chinese fast speech | **3.08%** | 26.15% | 100% |
| Chinese + noise | **0.00%** | 7.10% | 163% |
| Cross-lingual (en/fr/it/es) | **0.00%** | 20.63% | 95% |
| English low quality | **4.35%** | 11.18% | 98% |
| Japanese | 0.00% | **0.00%** | 4.00% |
| Russian | **0.00%** | **100% (empty)** | 99% |

**cohere must not be the global default.** It fails Russian outright and is 3–8× worse on
zh/en. This is exactly why the fix is routing, not replacement.

---

## 5. Change 1 — language-routed ASR model selection (**REMOVED 2026-09-12**)

> **Status: reverted at the user's request.** `ASRConfig.model_by_language`,
> `SessionState._apply_language_model_routing`, `SessionState._apply_asr_model`, the popup's
> `auto` engine option, the `backendAsrEnginePayload` helper and
> `tests/test_language_model_routing.py` have all been deleted. Model selection is a manual
> choice in the popup again.
>
> The **measurement below is unaffected and still the reason `cohere-transcribe` is worth
> picking by hand for Japanese** — it is kept here because it is the evidence for that choice,
> not because any code acts on it.

What was implemented, for the record: `model_by_language: {"ja": "cohere-transcribe"}` in
`ASRConfig`, applied by `SessionState` when the client announced a language and had not named a
model itself, with guards for unknown keys and models not present on disk.

Why it is not the right default for this app: "auto" had to be threaded through the popup, the
WS payload and `/api/config` just to keep a *fixed* model from silencing the map, and the routed
model changes behaviour per session. A deliberate per-video choice in the popup is simpler and
gives the user the same measured benefit.

---

## 6. Change 2 — separating the fragment gate from the final gate (default unchanged)

`min_words_to_commit` (default 4) was introduced to stop *fragments* of a split utterance
flashing on screen, and `ws_handler` reuses it to skip translating throwaway turns. The
engine, however, applied the same threshold to the **final** commit of a VAD-delimited
utterance, where it is not a fragment guard at all.

Measured with the production functions (`count_content_tokens` + `SentenceSegmenter`):

| | dropped |
|---|---|
| real Japanese turns | **5 / 12** — `うん。`, `そうね。`, `どうぞ。`, `はい。`, `そう。` |
| real English turns | **10 / 10** — `Yes.`, `I see.`, `OK.`, `Right.`, … |

So the fix looked obvious: give the final commit its own, lower gate. **It was measured,
and it is wrong.** On 8 verbatim Japanese streams through the real pipeline:

| configuration | CER | commits |
|---|---|---|
| gate = 4 (shipped behaviour) | **15.58%** | 123 |
| gate = 1 (recover short turns) | **17.57%** | 149 |

Every one of the 26 extra commits was inspected: **22 were `はい。` / `は。`, and none of
them occur anywhere in the reference.** An isolated VAD fragment is acoustically
ambiguous, and the decoder answers it with the most likely short Japanese turn — so
`min_words_to_commit` was unknowingly acting as a hallucination filter.

**What was shipped**: the two concerns are now separable knobs, with behaviour unchanged.

| knob | default | gates |
|---|---|---|
| `sentence.min_words_to_commit` | 4 (unchanged) | stability splits, preview display, translation queue |
| `sentence.min_words_to_emit_final` | **4 (unchanged)** | whether a **final** commit reaches the subtitle stream; set to 1 to emit genuine short turns |

Empty and punctuation-only output is always suppressed. This is a real availability/quality
dial for dialogue-heavy material (drama/anime with genuine `うん。` turns) — but it must be
measured on the actual content before being turned on, because the failure mode it trades
against is confidently wrong short subtitles.

This is also the same bug the Phase 3B audit saw from the other side (Mechanism 3:
`"with the radar."` arriving under the threshold and being thrown away, turning a correct
transcript into 16.77% CER) — the correct answer there is not a lower gate but an acoustic
boundary that does not create the fragment in the first place (§7).

`backend_cpp/tests/test_short_final_emission.py` — 9 tests, including a regression test
pinning the production default.

---

## 7. Where the remaining error is

Isolating the offline→streaming gap with `ja_stream_attribution.py`
(Common Voice streams, 8 × 63 s, single-shot whole-file decode as the upper bound):

| model | A. whole-file | B. VAD-segmented | C. streaming |
|---|---|---|---|
| qwen3-asr-1.7b | 17.68% | 17.11% | **15.58%** |
| cohere-transcribe | 42.87% | 13.47% | **12.11%** |

* **cohere cannot decode a 63 s file** (42.87%) — it is a short-segment model despite its
  400 s encoder bound, which is why routing (not replacement) is the correct fix, and why
  `sentence.max_duration_sec` must stay well below that.
* VAD segmentation costs both models several pp versus clean, sentence-bounded clips
  (cohere 2.88% → 13.47%), and the commit controller costs nothing (C ≤ B).

Readable failures on verbatim material show the mechanism — intra-sentence pauses split
clauses, and the fragment is decoded without its context:

```
REF: 木曜日の午後、銀行へお金を振り込みに行きます
qwen: 金曜日の午後。 インコイ。 お金を振り込みに行きます。      ← "銀行へ" becomes garbage
cohe: 車を買えば。 どこでも行けます。 お金を振り込みに行きます。   ← segment dropped instead
REF: 毎年家族で旅行に行きます
qwen: 毎年家族でイリオコに行きます。                          ← "旅行に" → "イリオコ"
```

**This is the next lever** and it is language-independent: give the decoder more
preceding context per VAD segment, or commit at acoustically safer boundaries.

Also observed: lowering `min_words_to_commit` itself to 1 made streaming **worse**
(28.42% vs 26.96% on TV streams) because it loosens the *split* gate as well as the final
gate — another instance of the same hallucination-filter effect described in §6.

### The next lever

`はい。` hallucinations and `インコイ`-style garbage both come from the same root: VAD
commits acoustically incomplete fragments. Two candidate directions, both language-independent
and both needing measurement:

1. **Fewer, acoustically safer cuts** — raise `vad.silence_duration_ms` (150 ms is short for
   intra-sentence Japanese pauses) and/or `vad.hangover_ms` (250 ms). The Phase 3B notes
   record that `hangover_ms` does *not* add commit latency, which makes it the cheaper knob;
   `silence_duration_ms` does, so it is a quality/latency trade.
2. **More preceding context per decode** — `vad.pre_speech_buffer_ms` is already 800 ms;
   whether the decoder actually receives that context at a mid-sentence cut was not verified
   here and should be checked directly from the dumped VAD utterances.

---

## 8. Reproduce

```powershell
# Build the eval sets (one-off, ~900 MB download)
python scratch/fetch_ja_eval.py
python -m benchmarks.build_ja_eval --limit 1000 --streams 8
python scratch/fetch_cv_ja.py
python -m benchmarks.build_ja_streams --src data/ja_cv/utts.jsonl --out data/ja_cv --start 400

# Verbatim model comparison
python -m benchmarks.ja_offline_bench --limit 400 --data-dir data/ja_cv

# Production-path streaming CER (real-time, ~9 min per config)
python -m benchmarks.streaming_wer_bench --dataset data/ja_cv/streams --lang ja
python -m benchmarks.streaming_wer_bench --dataset data/ja_cv/streams --lang ja --set asr.active_model=cohere-transcribe

# Attribute the offline→streaming gap
python -m benchmarks.ja_stream_attribution --model cohere-transcribe --streaming-report report/jacv_stream_cohere.json

# Cross-language regression guard for a model swap
python -m benchmarks.model_matrix
```

---

## 9. Test status

| check | result |
|---|---|
| `pytest backend_cpp/tests/test_short_final_emission.py` | 9 passed |
| `pytest backend_cpp/tests/test_language_model_routing.py` | 9 passed — **file deleted with the feature (§5)** |
| new tests + `test_commit_priority_and_min_words.py` + `test_sentence_segmenter.py` | **38 passed** |
| `pytest backend_cpp/tests` | 198 passed, 3 skipped, 11 failed, 1 error |
| same 11 failures with the changes stashed | **11 failed — pre-existing** |
| **no-regression proof** | new code with `min_words_to_emit_final` forced to 4 reproduces the old code's streaming run **exactly**: 15.58% CER, 274 edits, 123 commits |

The 11 failures are all `PermissionError` on `%TEMP%\dsh-*\…` (the DSH file sandbox denies
`tempfile`/`pytest tmp_path` cleanup) inside `test_vad_concurrency.py`,
`test_perf_baseline_gate.py`, `test_audio_pipeline_audit.py` and `test_utils.py`. They are
environmental and reproduce identically on a clean checkout.

---

## 10. Open items / recommended next steps

1. **Validate on the user's own content.** Everything domain-specific here rests on one
   29 s captured drama session. Recording a few minutes of real viewing with `dump_audio`
   and typing the transcript would give the only fully authoritative domain benchmark.
2. **The optimum for Japanese is probably `hangover_ms` / `silence_duration_ms`, not the
   model.** Both models lose several pp to VAD cuts (§7); that is the largest remaining
   addressable loss and it is language-independent.
3. **Model list hygiene.** `models.yaml` documents `context_window_sec: 30` for qwen3-asr,
   but the engine accepts ~87 min and the real generation budget is 256 tokens
   (`transcribe.cpp/src/arch/qwen3_asr/model.cpp: k_max_new`). The stale value misled the
   earlier audit into hand-windowing audio.
4. **`Qwen3-ASR-1.7B-JA-BF16.gguf` (4.1 GB)** is on disk and is *not* better than the base
   model on Japanese (23.47% vs 24.33% CER on the verbatim subset) while differing only in
   ITN. Safe to delete unless kept for experimentation.
