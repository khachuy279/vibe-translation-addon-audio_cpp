# Japanese Segmentation Design — turning VAD from decider into proposer

**Question this answers**: if cutting is where the error comes from, should we turn VAD off
and find another way to break sentences? And for two-person dialogue — when do we cut, and
what happens when the two speakers run into each other?

**Date**: 2026-09-12 · Companion measurement report: `report/ja_asr_optimization.md`

---

## 1. Correcting the premise first

> **The strongest result is in §10.** A controlled direct-vs-pipeline test shows the pipeline
> costs +2.06 pp CER on 10 short utterances, that **VAD alone** causes it, and that the
> mechanism is *lost audio extent* (trailing tail and a mid-utterance discontinuity) — not
> misplaced cuts. The fixes in §10 are smaller and better evidenced than the architecture
> sketched in §7; do those first.

The earlier numbers do **not** say "VAD keeps CER low, streaming adds 10%". They say:

| stage | cohere | qwen |
|---|---|---|
| **A.** decode the whole 63 s file in one pass | **42.87%** | 17.68% |
| **B.** cut with VAD, then decode offline | **13.47%** | 17.11% |
| **C.** full streaming pipeline | **12.11%** | 15.58% |

`report/ja_attribution_{cohere,qwen}.json`

* The **streaming controller costs nothing** — C is *better* than B for both models.
* **Turning VAD off makes cohere 3.5× worse** (13.47% → 42.87%). cohere is a short-segment
  model; it needs the segmentation.
* The 2.88% figure is not "with VAD" — it is decoding the **original clips with their own
  natural sentence boundaries**. Same audio, same model, different boundaries:
  **2.88% vs 13.47%.**

So the target is not "no segmentation". It is: **make our boundaries as good as the ones
the audio already had.**

---

## 2. How bad are the current boundaries? (measured, not inferred)

`benchmarks/vad_boundary_audit.py` scores the segmenter against known ground truth: the
Common Voice streams are built by concatenating whole sentences with a 350 ms gap, so the
exact offset of every true sentence boundary is known, and the gaps are verifiably digital
silence (−120 dBFS).

| `silence_duration_ms` | utterances / sentence | sentences split | extra splits | merged |
|---|---|---|---|---|
| **150 (current)** | **1.57** | **44.2%** | **54** | 0 |
| 250 | 1.33 | 28.4% | 31 | 0 |
| 400 | 1.23 | 20.0% | 22 | 0 |
| 600 | 1.15 | 14.7% | 14 | 0 |
| 900 | 1.01 | 8.4% | 8 | **7** |

**At the shipped setting, 44% of sentences are cut into fragments.** Every fragment is
decoded without its context, which is exactly where `インコイ`-style garbage and `はい。`
hallucinations come from.

### The finding that rules out the obvious fix

The median pause before a **true** boundary is **0.996 s**; before a **spurious** cut it is
**1.284 s**. The spurious cuts sit on *longer* pauses than the true boundaries.

**Silence duration alone cannot separate a sentence boundary from an intra-sentence pause.**
Raising the threshold helps only by cutting less overall — it does not learn where sentences
end. (Caveat: the synthetic gaps are 350 ms while the read sentences contain longer internal
pauses, so this corpus cannot show the best a duration rule could ever do. It is enough to
show the rule is not sufficient.)

### A dead configuration knob

`VADProcessor` line 276 does `grace_hangover_ms = min(self.hangover_ms, silence_duration_ms)`.
With `silence_duration_ms = 150`, every `hangover_ms` ≥ 150 is clamped to 150 — the sweep
confirms 250 / 600 / 1200 ms produce **byte-identical segmentation**. The C06 note in
`config.py` claims "hangover 250 ms preserves trailing phonemes / codas"; only 150 ms is
ever retained. The clamp is *logically* right (you cannot retain tail audio you have not
received without waiting), but the config value is misleading and the recorded decision is
not in effect.

---

## 3. The design problem

One threshold (`silence_duration_ms = 150`) currently answers three different questions:

| question | what it wants | current answer |
|---|---|---|
| **1. What audio does the ASR decode?** | complete acoustic units → long | 150 ms → fragments |
| **2. When does a subtitle appear?** | immediately → short | 350 ms preview poll |
| **3. When did the turn change?** | speaker evidence → neither | 150 ms silence |

They are conflated, so tuning for latency destroys accuracy. They must be **decoupled**.

Crucially, question 2 is **already answered independently** by the preview poller — the
engine re-decodes the entire growing utterance every ~350 ms, so `silence_duration_ms` does
*not* control when text appears on screen. It only controls when a line stops being
tentative. That means **the acoustic span can grow across short pauses at almost no
perceived-latency cost.**

---

## 4. Proposed architecture — soft / hard boundaries

### 4.1 VAD proposes, the engine decides

VAD keeps detecting speech and silence, but a silence no longer closes the decode segment.
The engine accumulates and finalises only on a **hard boundary**:

```
hard boundary when ANY of:
  * silence >= HARD_SILENCE_MS                       (default 600-700 ms)
  * silence >= SOFT_SILENCE_MS  AND  the decoded preview text
    ends at a sentence-final form                    (default 250 ms)
  * segment duration >= MAX_SEGMENT_SEC              (15-20 s, existing machinery)
  * (optional) a confirmed speaker change

never: silence >= 150 ms alone
```

* `HARD_SILENCE_MS` costs **commit** latency only. The subtitle is already on screen from
  the preview poller; what is delayed is the moment the line becomes final and gets
  translated.
* Every decode — preview and final — runs on the **whole growing segment**, so a fragment
  is never decoded alone. This is where the 44% over-segmentation disappears.
* The existing `BoundaryState` / `FORCED_PENDING` / `boundary_candidate_silence_ms` machinery
  already implements the "wait for an acoustic dip before the forced cut" idea for
  `max_duration`. This proposal applies the same discipline to **every** boundary instead of
  only the 15 s one.

### 4.2 Use the decoded text as the sentence signal (especially for Japanese)

Japanese gives an unusually clean signal, and the engine already has the text in hand from
the preview:

* sentence-final forms: `です / ます / でした / ません / だ / だった / ね / よ / な / わ / か / の`
* sentence-final punctuation, which Qwen3-ASR emits reliably (`。！？`)
* a trailing particle with no continuation

Rule: **text may only delay a cut, never force one.** If the preview ends mid-phrase, wait
for more silence or more audio. This is the safe asymmetry: waiting costs latency, cutting
mid-phrase costs correctness.

This also fixes the Phase 3B Mechanism 3 case (`"with the radar."` dropped) from first
principles: the splitter would not have committed after `"...a problem."` because the next
phoneme had already started.

### 4.3 Two speakers running into each other

This is the case where no energy rule can work, and the honest answer is: **do not try.**

| situation | what to do | why |
|---|---|---|
| clear pause between turns (≥ hard threshold) | cut — normal case | both signals agree |
| short pause, next speaker starts | **merge, do not cut** | a wrong cut produces garbage; a merged line still has the right words |
| overlap (B starts before A finishes) | **merge** | separation is not possible from one mixed channel |

**Never cut on a guessed turn change.** A merged line costs speaker attribution; a wrong
cut costs words. For a subtitle tool, words win.

If speaker labels are actually wanted later, `transcribe.cpp` already ships streaming
diarization (`diar_streaming_sortformer_4spk-v2.1`, `Diarize` / `sortformer` options in the
Python binding), and the "hardware does not matter" constraint makes it affordable. It
should be used as a **labelling** signal feeding the boundary vote — never as a standalone
cut trigger, for the same reason.

### 4.4 Two-pass for maximum accuracy

Since this is a personal tool, the cheapest accuracy win is to stop treating the first
decode as final:

1. **Pass 1** — growing-segment decode drives the live subtitle (unchanged UX).
2. **Pass 2** — when a hard boundary closes, re-decode the complete segment once more and
   **replace** the displayed line. Optionally prepend ~1 s of the previous segment's audio
   as context and strip its text from the output; this directly targets the
   `旅行に → イリオコ` class of errors, where the model misreads a word that starts at a cut.

The engine already distinguishes preview from final, so this is a change of *policy*
(which decode is authoritative) rather than new machinery.

---

## 5. How bad are the boundaries? (see §2)

Measured above; the code mapping is in §8.

---

## 6. The falsification test — and why it came back inconclusive

The cheapest test of the whole hypothesis is to change only `silence_duration_ms`, which
fixes over-segmentation structurally, and see whether CER follows:

```
python -m benchmarks.streaming_wer_bench --dataset data/ja_cv/streams --lang ja \
    --set vad.silence_duration_ms=N
```

| stream | 150 ms (current) | 600 ms | 900 ms |
|---|---|---|---|
| stream00 | 10.23% | **6.82%** | **6.82%** |
| stream01 | **18.52%** | 25.93% | 22.84% |
| stream02 | **12.16%** | 13.51% | 14.41% |
| stream03 | 14.29% | **13.36%** | **13.36%** |
| stream04 | 14.91% | **14.55%** | 14.91% |
| stream05 | 11.74% | 11.74% | 11.74% |
| stream06 | **17.20%** | 28.00% | **17.20%** |
| stream07 | 25.11% | 21.59% | **20.26%** |
| **corpus** | **15.58%** (123 commits) | 17.00% (99) | 15.18% (92) |
| utterances / sentence | 1.57 | 1.15 | 1.01 |

**Verdict: the fragmentation hypothesis is NOT confirmed.** At 600 ms the segmentation is
almost perfect (99 commits against 95 true sentences) yet corpus CER *rose*. At 900 ms it
returned to roughly the 150 ms level. Per-stream swings of ±5 pp dwarf the ~1.5 pp corpus
difference across only 8 streams.

### Why the test is confounded

The stream corpus concatenates **random, semantically unrelated** Common Voice sentences
with a 350 ms gap. Merging two of them hands an LLM-based ASR an incoherent context, which
can be worse than decoding a fragment. Real video has coherent discourse, so that failure
mode does not exist there — but neither does the corpus let us measure the real trade-off.

Two further limits: the synthetic gaps (350 ms) are *shorter* than many intra-sentence
pauses in read speech, so the corpus cannot reveal the ceiling of any duration rule; and
structural ground truth says nothing about whether a *correct* boundary is also a
*linguistically useful* one.

### What stands, and what does not

| claim | status |
|---|---|
| The current setting splits 44% of sentences into fragments | **measured, solid** (structural ground truth) |
| `hangover_ms` is inert above `silence_duration_ms` | **measured, solid** (byte-identical segmentation at 250/600/1200 ms) |
| Silence duration alone cannot separate sentence boundaries from intra-sentence pauses | **measured, solid** (spurious cuts sit on *longer* median pauses: 1.284 s vs 0.996 s) |
| Reducing over-segmentation lowers CER | **NOT established** — falsified on this corpus, and the corpus is not valid for it |
| The 2.88% → 13.47% gap is caused by fragment decoding | **NOT established** — confounded with "isolated clean clips vs synthetic concatenated streams" |

**The blocking prerequisite is a real dialogue corpus** — coherent multi-turn speech with a
typed transcript. Without it none of the boundary policy below can be tuned or even
validated, and building it would be guesswork dressed up as a rationale.

---

## 7. The design, stated as a hypothesis to be tested

Everything below is **reasoning from first principles, not from measurement**. It is written
down because it is implementable and cheap to test *once a valid corpus exists*, not because
the evidence supports it yet.

### 7.1 Keep VAD; it is not optional

Turning VAD off makes cohere 3.5× worse (13.47% → 42.87% on a 63 s file). cohere is a
short-segment model, and qwen barely benefits from long audio either (17.68% whole-file vs
17.11% VAD-segmented). The lever is *where* the cuts go, not whether to cut.

### 7.2 Decouple the three questions one threshold currently answers

| question | what it wants | current answer |
|---|---|---|
| What audio does the ASR decode? | complete acoustic units → long | 150 ms → fragments |
| When does a subtitle appear? | immediately | 350 ms preview poll |
| When did the turn change? | speaker evidence | 150 ms silence |

The second is already answered independently by the preview poller, which re-decodes the
whole growing utterance every ~350 ms. `silence_duration_ms` therefore does **not** control
when text appears — only when a line stops being tentative. That is the room the design
exploits.

### 7.3 Soft / hard boundaries

VAD proposes; the engine decides. Finalise only on a hard boundary:

```
hard boundary when ANY of:
  * silence >= HARD_SILENCE_MS                      (600-700 ms)
  * silence >= SOFT_SILENCE_MS AND the decoded preview text ends
    at a sentence-final form                        (250 ms)
  * segment duration >= MAX_SEGMENT_SEC             (existing machinery)
  * (optional) a confirmed speaker change
never: silence alone at the current 150 ms
```

Every decode — preview and final — runs on the **whole growing segment**, so no fragment is
ever decoded in isolation. The existing `BoundaryState` / `FORCED_PENDING` /
`boundary_candidate_silence_ms` machinery already applies this discipline to `max_duration`;
the proposal extends it to all boundaries.

### 7.4 Japanese sentence-final forms as the linguistic signal

The engine already has the preview text. Japanese offers an unusually clean cue:

* `です / ます / でした / ません / だ / だった / ね / よ / な / わ / か / の`
* sentence-final punctuation, which Qwen3-ASR emits reliably (`。！？`)
* a trailing particle with no continuation

**Text may only delay a cut, never force one.** Waiting costs latency; cutting mid-phrase
costs words.

### 7.5 Two speakers running into each other

No energy rule can separate overlapping turns, and the honest answer is: **do not try.**

| situation | action | rationale |
|---|---|---|
| clear pause between turns | cut | both signals agree |
| short pause, next speaker starts | **merge** | a wrong cut produces garbage; a merged line still has the right words |
| overlap | **merge** | separation is impossible from one mixed channel |

If speaker labels are wanted later, `transcribe.cpp` ships streaming diarization
(`diar_streaming_sortformer_4spk-v2.1`; `Diarize` / `sortformer` in the Python binding). Use
it as a **voting** signal, never as a standalone cut trigger — a merged line costs
attribution, a wrong cut costs words.

### 7.6 Two-pass for maximum accuracy

Hardware is not a constraint here, so stop treating the first decode as final:

1. **Pass 1** — growing-segment decode drives the live subtitle (UX unchanged).
2. **Pass 2** — when a hard boundary closes, re-decode the complete segment and **replace**
   the line. Optionally prepend ~1 s of the previous segment as acoustic context and strip
   its text, targeting the `旅行に → イリオコ` class of errors where a word is misread
   because it starts at a cut.

---

## 8. Mapping onto the current code

| file | change |
|---|---|
| `backend_cpp/vad/vad_processor.py` | demote silence to a *candidate*; stop closing the segment on `silence_duration_ms` alone |
| `backend_cpp/asr/transcribe_engine.py` | own the boundary decision: hard/soft thresholds, sentence-final check on the preview text, commit only on a hard boundary; generalise `BoundaryState` beyond the max-duration path |
| `backend_cpp/asr/sentence_segmenter.py` | add `ends_at_sentence_final(text, language)`; keep stability as a secondary signal |
| `backend_cpp/config.py` | `hard_silence_ms`, `soft_silence_ms`; **fix the `hangover_ms` clamp** so the documented value is real, or delete the knob and state that tail retention is bounded by `silence_duration_ms` |

### The one change justified today, needing no new corpus

`hangover_ms` is inert above `silence_duration_ms` (§2). Either honour the configured value
by decoupling *decision* latency from *retention* length, or delete the knob and correct the
C06 decision note. As it stands the recorded decision is not in effect.

---

## 9. Recommended order of work

1. **Build a real dialogue corpus** — a few minutes of actual viewing with `dump_audio`, plus
   a typed transcript. This blocks everything else.
2. Re-run `benchmarks/streaming_wer_bench.py` on it for an absolute, trustworthy production
   CER for Japanese (none exists today).
3. Re-run the silence sweep **on that corpus** to see whether the fragmentation hypothesis
   holds for coherent discourse.
4. Only then implement §7.3–7.6, gated by `vad_boundary_audit.py` (structure) and
   `streaming_wer_bench.py` (accuracy).
5. Independently: fix or remove the `hangover_ms` clamp.

---

## 10. Transparency test — the strongest evidence (read this first)

A controlled experiment with one model (Qwen3-ASR-1.7B), 10 short utterances with verbatim
references, and VAD neutered (`silence_duration_ms = 60 s`) so no cut can occur:

```
Run 1  audio -> model                    -> CER 17.53%   (direct, no pipeline)
Run 2  audio -> real pipeline -> model   -> CER 19.59%   (+2.06 pp)
```

`benchmarks/pipeline_transparency_test.py`

**(1) < (2): the pipeline does introduce error, and one component owns all of it.**

| run | CER | Δ | utterances whose text changed |
|---|---|---|---|
| direct model | **17.53%** | — | — |
| pipeline, VAD neutered | 19.59% | **+2.06 pp** | 3 / 10 |
| − SpeechNormalizer | 19.59% | +2.06 pp | 3 / 10 |
| − short-commit drop filter | 19.59% | +2.06 pp | 3 / 10 |
| **− VAD (engine fed directly)** | **17.53%** | **+0.00 pp** | **1 / 10** |

The normalizer, ring buffer, int16 round-trip, `clean_transcript_text` and the whole commit
path are **transparent**. Removing VAD reproduces the direct result exactly. **VAD is the
only component that alters the outcome** — and note the neutered setting means this is *not*
about cutting at the wrong place; it is about the audio VAD delivers.

### Cause: audio extent, proven by replay

Replaying the exact float32 array the engine handed to `transcribe_cpp` reproduces the
pipeline transcript in every case, so it is an audio-integrity difference, not a difference
in how the model is called. Against the source file:

| utterance | file | fed to model | difference |
|---|---|---|---|
| cv00000 | 5.66 s | 5.520 s | −144 ms **tail only**; samples otherwise identical (max diff 3.1e-5 = 1 int16 LSB) |
| cv00008 | 4.46 s | 3.960 s | −504 ms **tail only**; samples otherwise identical |
| cv00006 | 3.58 s | 3.000 s | −576 ms **plus a mid-utterance discontinuity** (max diff 1.21) |
| cv00002 | 5.38 s | 3.960 s | −1.42 s: lead + interior dropped |

Model sensitivity to exactly that, measured directly:

| input | corpus CER | Δ vs full file |
|---|---|---|
| full file | 17.53% | — |
| −150 ms tail | 18.04% | **+0.52 pp** |
| −300 ms tail | 18.56% | **+1.03 pp** |
| −500 ms tail | 19.07% | **+1.55 pp** |
| +500 ms silence | 17.53% | +0.00 pp |
| +1000 ms silence | **17.01%** | **−0.52 pp** |

Truncation degrades monotonically, and the VAD's measured tail losses (144 ms, 504 ms) map
onto +0.52 pp and +1.55 pp — **most of the +2.06 pp penalty is the missing tail.**

### Root causes in `vad_processor.py`

**a. Trailing audio is capped by the silence threshold** (line 276):

```python
total_silence_limit_ms = float(self.silence_duration_ms)                          # 150
grace_hangover_ms      = min(float(self.hangover_ms), total_silence_limit_ms)     # -> 150
if silence_elapsed_ms <= grace_hangover_ms:
    feed(tail)     # only the first 150 ms of tail ever reaches the model
```

`hangover_ms` can never exceed `silence_duration_ms`, so the configured 250 ms — and the C06
decision recorded in `config.py` — are both inert. The model wants more tail than the
decision latency allows, and today those are the same number.

**b. Interior non-speech runs are gated out**, so the fed audio is not a contiguous span. On
cv00006 that produced a discontinuity which removed the first syllable:
`田中さんの右に…` → `中さんの右に…`.

**c. Onset confirmation can exceed `pre_speech_buffer_ms`:**

| utterance | pre=0 | pre=200 | pre=800 (current) | pre=1500 | pre=3000 |
|---|---|---|---|---|---|
| cv00002 | 1.088 s | 0.960 s | **0.384 s** | 0.000 s | 0.000 s |
| cv00007 | 1.152 s | 1.024 s | **0.448 s** | 0.000 s | 0.000 s |

FSMN-VAD onset confirmation reaches ~1.0–1.5 s, so with an 800 ms ring the first 0.13–0.45 s
of those utterances is permanently discarded. Raising the ring costs RAM and **zero latency**.

### Justified fixes, in order of value

1. **Decouple tail retention from the silence decision** — decide the end at
   `silence_duration_ms` as today, but keep collecting ~500 ms after the last speech frame
   before running commit inference. Costs ~350 ms of *commit* latency, not display latency
   (the preview poller already shows text continuously).
2. **Pad the decode window with 500–1000 ms of digital silence** — measured 0.00 pp at 500 ms
   and −0.52 pp at 1000 ms, at **zero latency cost**.
3. **Feed a contiguous span, not gated frames** — slice the raw buffer from
   `first_speech − pre_roll` to `last_speech + tail`; removes the discontinuity class.
4. **Raise `pre_speech_buffer_ms` to 1500–2000 ms.**

These four are smaller and far better evidenced than the soft/hard architecture in §7, and
should be done first. Note also that all measurements in this section used a **leap-second
free, deterministic** VAD: two passes over the same audio produced byte-identical spans, so
these are systematic losses, not noise.

---

## 11. VAD segmentation sweep — the shipped default was the worst setting (2026-09-12)

Test material: `data/ja_long3`, a 3-minute stream built by concatenating whole Common Voice
sentences with a 500 ms gap, so all 33 true sentence boundaries are known.

* **(1) ceiling** = decode each ground-truth sentence span separately: **10.68%** CER.
  Cached in `report/refs/stream00.json`; it depends only on the stream and the model, never on
  a VAD setting, so a sweep point costs ~3.8 min instead of ~6.5 min.
* **(3) pipeline** = the real path with VAD segmentation ON, `max_duration_sec = 8 s`.
* **Goal**: (3) → (1), with subtitle segments under 8 s.

| `silence_duration_ms` | (3) CER | gap vs (1) | segments | p95 length | > 8 s | S / D / I |
|---|---|---|---|---|---|---|
| **150 (old default)** | **12.46%** | **+1.78 pp** | 35 | 7.4 s | 2 | 46 / 32 / 6 |
| **250 (new default)** | **10.39%** | **−0.30 pp** | 33 | **7.5 s** | 2 | 39 / 25 / 6 |
| 400 | 9.35% | −1.33 pp | 32 | 8.4 s | 3 | 34 / 22 / 7 |
| 600 | 12.02% | +1.34 pp | 33 | 8.4 s | 3 | 54 / 12 / 15 |
| 300 + `hangover_ms=400` | 12.17% | +1.48 pp | 33 | 8.5 s | 3 | 52 / 11 / 19 |

**Findings**

1. `silence_duration_ms` is the dominant accuracy knob in the pipeline, and **150 ms was the
   worst value tested**. It is no longer the default: **250 ms** meets the goal (−0.30 pp, i.e.
   already better than a ground-truth-segmented decode) while keeping p95 segment length at
   7.5 s.
2. The 150 ms setting over-segments (35 segments for 33 sentences); its extra substitutions and
   deletions (46/32 vs 39/25) are fragments decoded without context.
3. Past ~400 ms the error turns back up because segments start *merging*, and the decoder
   inserts text to bridge the join (insertions 7 → 15 → 19). **Caveat**: this corpus
   concatenates *unrelated* sentences, so merging is punished harder here than it would be on
   coherent dialogue. On real video a longer value may win — the optimum measured here is a
   lower bound.
4. `hangover_ms` does not substitute for `silence_duration_ms`: 300+400 (an effective 700 ms
   window) behaves like 600, not like 400.
5. **Reproducibility achieved.** Two runs at 250 ms (with different `max_duration_grace_sec`)
   produced byte-identical CER (10.39%) and identical max segment length. This is what made the
   sweep interpretable, and it required disabling the wall-clock-dependent preview reuse (§12).

### The 8 s constraint is in tension with accuracy — and the resolution is to split the text

The ground-truth material itself contains a **9.4 s sentence**, so "every segment < 8 s" cannot
be satisfied without cutting speech. Measured cost of forcing it with
`max_duration_grace_sec`: at 400 ms, tightening grace from 2.0 s to 0.4 s moved max length only
from 9.60 s to 8.76 s and **cost 0.44 pp CER** (9.35% → 9.79%). The emergency cut has to slice
mid-word.

The measured direction of the trade is unambiguous — **longer decode units score better**
(9.35% at 400 ms vs 12.46% at 150 ms) — so shortening the *decode* unit to satisfy a
*readability* limit is the wrong lever. The right one is to **decode on the acoustically best
unit and split the displayed subtitle line by text**, which satisfies the 8 s reading limit
without touching the audio the model sees. That is the next change to make here.

---

## 12. Wall-clock nondeterminism removed

`reuse_stable_preview` (new config, **default off**) previously let a commit reuse the last
preview's text when no new audio had arrived since it was produced. Whether it fires depends on
poller wall-clock timing, and the normalizer's smoothed gain state differs between a preview
call and a commit call, so the reused text is not always what a fresh commit decode yields.
Measured on one identical stream: 10.31% vs 10.75% CER across runs — the same magnitude as every
effect being optimised, which made those effects unmeasurable. The commit is now always an
authoritative decode of the complete utterance. Set `asr.reuse_stable_preview = true` to trade
reproducibility for one saved inference per utterance.

---

## 13. Summary of changes made

| change | file | effect |
|---|---|---|
| `silence_duration_ms` 150 → **250** | `config.py` | gap vs (1) from **+1.78 pp to −0.30 pp** |
| tail no longer clamped; frames always delivered | `vad_processor.py` | removes mid-utterance discontinuities; `hangover_ms` becomes a real knob |
| `hangover_ms` 250 → **0** | `config.py` | keeps commit latency identical to before (the tail is now `silence_duration_ms` by construction) |
| `pre_speech_buffer_ms` 800 → **1500** | `config.py` | recovers up to 0.45 s of a sentence's opening audio that VAD onset confirmation was discarding |
| `reuse_stable_preview` **false** | `config.py`, `transcribe_engine.py` | removes wall-clock nondeterminism from commits |
| `min_words_to_emit_final` separated | `config.py`, `segmenter`, `engine` | knob exists; default unchanged at 4 because lowering it measured worse |

**Results**: pipeline vs ground-truth-segmented decode on verbatim material **+1.78 pp → −0.30 pp**.

> **Removed 2026-09-12 at the user's request**: per-language ASR model routing
> (`ASRConfig.model_by_language`, the popup's `auto` engine option, `_apply_language_model_routing`
> and its tests). The *measurement* stands and is kept in
> `report/ja_asr_optimization.md` §4 — `cohere-transcribe` scores 2.88% vs `qwen3-asr-1.7b`'s
> 11.69% CER on verbatim Japanese while being much worse on Chinese/English and failing Russian —
> but nothing switches models automatically any more. Selecting a model is a manual choice in the
> popup.

### Next

1. **Split displayed subtitle lines by text** so the decode unit can stay long (§11).
2. Re-measure the sweep on **coherent dialogue** — the optimum here is a lower bound because the
   corpus punishes merging artificially.
3. Delete or repurpose `Qwen3-ASR-1.7B-JA-BF16.gguf` (4.1 GB, not better than the base model).

---

## 14. Extension defaults aligned with the measurements

`extension_firefox/popup/popup.html`, `popup.js`, `content/content-script.js`

| setting | was | now | why |
|---|---|---|---|
| VAD Silence slider + `DEFAULT_SETTINGS` | `150` | **`250`** | measured optimum (§11): pipeline gap +1.78 pp → −0.30 pp |
| `hangoverMs` sent over WS | `250` (forced, and `\|\|`-defaulted so 0 was impossible) | **`0`** | with the current semantics it *adds* to `silenceDurationMs`, so 250 silently made the effective split threshold 500 ms, which measured worse |

`threshold` (0.20) and `minWordsToCommit` (4) were already correct and were left alone — lowering
the latter measured **worse** (§6).

The ASR model dropdown keeps `qwen3-asr-1.7b` as its default; `cohere-transcribe` remains available
as an explicit choice for Japanese.
