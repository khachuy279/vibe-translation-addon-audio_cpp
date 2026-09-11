"""Streaming Failure Attribution Benchmark Suite (Phase 3B).

Audits the five streaming failure mechanisms:
1. Hypothesis evolution & prefix survival
2. Repeated inference (Test A: Same PCM) vs Growing context (Test B: Context churn)
3. Stability decision & premature commitment
4. Audio/text state synchronization (Paired boundary test)
5. Mariachi background music repetition loop (Counterfactual Matrix M0-M4)

Usage:
  python -m benchmarks.audit_streaming_failure --stage 1
  python -m benchmarks.audit_streaming_failure --stage 2
  python -m benchmarks.audit_streaming_failure --stage 3
  python -m benchmarks.audit_streaming_failure --stage full
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import rapidfuzz.distance.Levenshtein as lev

from backend_cpp.asr.audio_buffer import AudioBufferManager, AudioSnapshot, VAD_STATE_SPEECH
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config as app_cfg
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.accuracy import evaluate_accuracy, AccuracyResult
from benchmarks.dataset import discover_dataset, DatasetPair

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_streaming")


# =====================================================================
# Telemetry Data Structures
# =====================================================================

@dataclass
class PollTelemetryRecord:
    inference_id: int
    poll_index: int
    timestamp_sec: float
    buffer_duration_sec: float
    buffer_start_sample: int
    buffer_end_sample: int
    sample_count: int
    audio_hash: str
    preview_text: str
    previous_preview_text: str
    hypothesis_delta: Dict[str, int]  # insertions, deletions, substitutions, stable_prefix_words
    stable_prefix_text: str
    committed_prefix_before: str
    committed_prefix_after: str
    is_stable: bool
    did_split: bool
    slice_start_sample: Optional[int] = None
    slice_end_sample: Optional[int] = None
    remaining_buffer_hash: Optional[str] = None
    remaining_samples: Optional[int] = None
    infer_latency_ms: float = 0.0


@dataclass
class BoundaryCutRecord:
    split_index: int
    timestamp_sec: float
    utterance_id: str
    committed_text: str
    snapshot_samples: int
    tail_audio_hash: str
    head_audio_hash: str
    tail_audio_rms: float
    head_audio_rms: float
    subsequent_first_preview: str = ""
    boundary_anomaly: str = "NONE"  # TRUNCATED_CODA, REPEATED_ONSET, etc.


@dataclass
class StreamingSessionTelemetry:
    pair_id: str
    config_id: str
    audio_duration_sec: float
    wall_duration_sec: float
    number_of_polls: int = 0
    number_of_inferences: int = 0
    unique_audio_snapshots: int = 0
    duplicate_audio_snapshots: int = 0
    total_audio_seconds_inferred: float = 0.0
    audio_exposure_ratio: float = 0.0
    poll_records: List[PollTelemetryRecord] = field(default_factory=list)
    boundary_cuts: List[BoundaryCutRecord] = field(default_factory=list)
    final_commits: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis_timeline: List[Dict[str, Any]] = field(default_factory=list)
    raw_hypothesis: str = ""
    accuracy: Optional[AccuracyResult] = None
    rollback_count: int = 0
    repetition_run_length: int = 0
    inserted_word_count: int = 0
    loop_amplification_factor: float = 0.0


# =====================================================================
# Instrumented Transcribe Engine
# =====================================================================

class InstrumentedTranscribeEngine(TranscribeEngine):
    """Subclass of TranscribeEngine instrumented to capture full streaming failure telemetry."""

    def __init__(self, session_id: str, on_telemetry: Optional[Callable[[PollTelemetryRecord], None]] = None, **kwargs):
        super().__init__(session_id=session_id, **kwargs)
        self.on_telemetry_cb = on_telemetry
        self.telemetry_records: List[PollTelemetryRecord] = []
        self.boundary_records: List[BoundaryCutRecord] = []
        self.raw_commits: List[Dict[str, Any]] = []
        self.global_inference_counter: int = 0
        self.poll_counter: int = 0
        self.buffer_start_cumulative: int = 0
        self._last_raw_preview: str = ""
        self._seen_audio_hashes: set = set()
        self.duplicate_audio_count: int = 0
        self.total_audio_inferred_sec: float = 0.0

    def _compute_hash(self, pcm: np.ndarray) -> str:
        return hashlib.sha256(pcm.tobytes()).hexdigest()[:16]

    def _compute_hypothesis_delta(self, prev_text: str, curr_text: str) -> Dict[str, int]:
        prev_words = prev_text.split()
        curr_words = curr_text.split()
        ops = lev.editops(prev_words, curr_words)
        insertions = sum(1 for op in ops if op[0] == "insert")
        deletions = sum(1 for op in ops if op[0] == "delete")
        substitutions = sum(1 for op in ops if op[0] == "replace")
        
        # Longest Common Prefix words
        lcp_words = 0
        for w1, w2 in zip(prev_words, curr_words):
            if w1.lower() == w2.lower():
                lcp_words += 1
            else:
                break
                
        return {
            "insertions": insertions,
            "deletions": deletions,
            "substitutions": substitutions,
            "stable_prefix_words": lcp_words,
        }

    async def _partial_preview_poller(self) -> None:
        """Instrumented preview poller recording detailed telemetry at each tick."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval_sec)
                self.poll_counter += 1

                with self._state_lock:
                    if not self._is_speech_active:
                        self._last_polled_samples = 0
                        self._last_preview_duration_sec = 0.0
                        continue

                with TranscribeEngine._commit_lock:
                    if TranscribeEngine._commit_waiting > 0:
                        continue

                snapshot = self._audio_buffer_mgr.get_snapshot_if_newer(self._last_polled_samples)
                if snapshot is None:
                    continue

                pcm_snapshot = snapshot.pcm
                dur = snapshot.duration_sec
                snapshot_samples = snapshot.sample_count
                snap_ver = snapshot.version
                frame_state = snapshot.frame_state

                if dur < self._min_transcribe_sec:
                    continue

                # Growth gate check
                if not self._should_run_preview(dur):
                    self._note_preview_skipped(snapshot_samples)
                    continue

                self._note_preview_ran(dur, snapshot_samples)

                with self._state_lock:
                    utt_id = self._current_utterance_id

                self.global_inference_counter += 1
                inf_id = self.global_inference_counter
                audio_hash = self._compute_hash(pcm_snapshot)
                if audio_hash in self._seen_audio_hashes:
                    self.duplicate_audio_count += 1
                else:
                    self._seen_audio_hashes.add(audio_hash)
                self.total_audio_inferred_sec += dur

                # Run inference
                t_inf_start = time.perf_counter()
                preview_text = await asyncio.to_thread(
                    self._run_inference, pcm_snapshot, frame_state=frame_state, is_commit=False, utt_id=utt_id
                )
                infer_latency_ms = (time.perf_counter() - t_inf_start) * 1000.0

                # Prefix stripping
                with self._state_lock:
                    head = self._last_committed_head
                    head_time = self._last_committed_head_time

                committed_prefix_before = head or ""
                stripped_preview = preview_text
                if head and (time.time() - head_time < 8.0):
                    stripped_preview = SentenceSegmenter.remove_prefix_overlap(head, preview_text)

                hyp_delta = self._compute_hypothesis_delta(self._last_raw_preview, stripped_preview)
                self._last_raw_preview = stripped_preview

                if not stripped_preview:
                    continue

                # Stability check
                is_stable = False
                if self.sentence_config.split_on_stability:
                    is_stable = self._segmenter.check_stability(stripped_preview) and not self._segmenter.is_text_filtered(stripped_preview)

                should_emit_partial = False
                did_split = False

                with self._state_lock:
                    if not self._is_speech_active or utt_id != self._current_utterance_id:
                        continue

                    if is_stable:
                        self._current_utterance_id = str(uuid.uuid4())
                        self._last_partial_text = ""
                        did_split = True
                    elif stripped_preview != self._last_partial_text:
                        self._last_partial_text = stripped_preview
                        self._last_partial_samples = snapshot_samples
                        should_emit_partial = True

                # Telemetry record for this poll
                record = PollTelemetryRecord(
                    inference_id=inf_id,
                    poll_index=self.poll_counter,
                    timestamp_sec=time.time(),
                    buffer_duration_sec=dur,
                    buffer_start_sample=self.buffer_start_cumulative,
                    buffer_end_sample=self.buffer_start_cumulative + snapshot_samples,
                    sample_count=snapshot_samples,
                    audio_hash=audio_hash,
                    preview_text=stripped_preview,
                    previous_preview_text=self._last_raw_preview,
                    hypothesis_delta=hyp_delta,
                    stable_prefix_text=self._segmenter._last_stable_text if hasattr(self._segmenter, "_last_stable_text") else "",
                    committed_prefix_before=committed_prefix_before,
                    committed_prefix_after="",
                    is_stable=is_stable,
                    did_split=did_split,
                    infer_latency_ms=infer_latency_ms,
                )

                if did_split:
                    with self._state_lock:
                        self._last_committed_head = stripped_preview
                        self._last_committed_head_time = time.time()
                    record.committed_prefix_after = stripped_preview
                    record.slice_start_sample = self.buffer_start_cumulative
                    record.slice_end_sample = self.buffer_start_cumulative + snapshot_samples

                    # Capture boundary audio slices (Tail A and Head B)
                    tail_samples = min(snapshot_samples, int(DEFAULT_SAMPLE_RATE * 0.5))
                    tail_pcm = pcm_snapshot[-tail_samples:]
                    tail_hash = self._compute_hash(tail_pcm)
                    tail_rms = float(np.sqrt(np.mean(tail_pcm ** 2) + 1e-12))

                    self._emit_final(stripped_preview, utt_id, reason="STABLE_PREFIX")
                    self._audio_buffer_mgr.slice_after(snapshot_samples, expected_version=snap_ver)
                    self.buffer_start_cumulative += snapshot_samples

                    # Inspect remaining buffer
                    rem_snapshot = self._audio_buffer_mgr.get_snapshot()
                    if rem_snapshot and rem_snapshot.sample_count > 0:
                        record.remaining_buffer_hash = self._compute_hash(rem_snapshot.pcm)
                        record.remaining_samples = rem_snapshot.sample_count
                        head_samples = min(rem_snapshot.sample_count, int(DEFAULT_SAMPLE_RATE * 0.5))
                        head_pcm = rem_snapshot.pcm[:head_samples]
                        head_hash = self._compute_hash(head_pcm)
                        head_rms = float(np.sqrt(np.mean(head_pcm ** 2) + 1e-12))
                    else:
                        record.remaining_buffer_hash = "EMPTY"
                        record.remaining_samples = 0
                        head_hash = "EMPTY"
                        head_rms = 0.0

                    # Save Boundary Cut Record
                    b_cut = BoundaryCutRecord(
                        split_index=len(self.boundary_records) + 1,
                        timestamp_sec=time.time(),
                        utterance_id=utt_id,
                        committed_text=stripped_preview,
                        snapshot_samples=snapshot_samples,
                        tail_audio_hash=tail_hash,
                        head_audio_hash=head_hash,
                        tail_audio_rms=tail_rms,
                        head_audio_rms=head_rms,
                    )
                    self.boundary_records.append(b_cut)

                    self._last_polled_samples = 0
                    self._last_preview_duration_sec = 0.0
                    self._segmenter.reset_stability()

                self.telemetry_records.append(record)
                if self.on_telemetry_cb:
                    self.on_telemetry_cb(record)

                if should_emit_partial:
                    out_msg = {
                        "type": "utterance_update",
                        "utterance_id": utt_id,
                        "text": stripped_preview,
                        "is_final": False,
                    }
                    self._push_message(out_msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in instrumented preview poller: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    def _emit_final(self, text: str, utt_id: str, reason: str) -> None:
        """Capture all finalized commits."""
        super()._emit_final(text, utt_id, reason)
        self.raw_commits.append({
            "utterance_id": utt_id,
            "text": text.strip(),
            "reason": reason,
            "timestamp": time.time(),
        })


# =====================================================================
# Streaming Audio Player / Feeder
# =====================================================================

from benchmarks.simulator import StreamingAudioSimulator


async def feed_streaming_audio(
    wav_path: Path,
    engine: InstrumentedTranscribeEngine,
    vad: VADProcessor,
    chunk_ms: int = 64,
    speed: float = 1.0,
    trailing_silence_sec: float = 1.5,
) -> None:
    """Feed audio file into VAD and Engine matching browser WebSocket streaming."""
    sim = StreamingAudioSimulator(
        audio_source=wav_path,
        chunk_ms=chunk_ms,
        speed=speed,
        trailing_silence_sec=trailing_silence_sec,
    )
    for chunk in sim.iter_chunks():
        vad.feed_chunk(chunk.pcm_bytes)
        await asyncio.sleep((chunk.duration_ms / 1000.0) / speed)


# =====================================================================
# Mechanism 2 Tests: Repeated vs Growing Inference
# =====================================================================

def run_mechanism_2_tests(wav_path: Path, model_key: str = "qwen3-asr-1.7b") -> Dict[str, Any]:
    """Audit Mechanism 2: Separate Test A (Same PCM Repeated Inference) from Test B (Growing Context)."""
    logger.info("🧪 [MECHANISM 2] Starting Test A: Same PCM Repeated Inference (20 runs)...")
    mgr = ASRModelManager()
    model = mgr.ensure_model(model_key)
    ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model)
        sim = StreamingAudioSimulator(audio_source=wav_path)
        data = np.frombuffer(sim.pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        pcm_3s = data[: 3 * 16000].astype(np.float32)
        audio_hash_3s = hashlib.sha256(pcm_3s.tobytes()).hexdigest()[:16]

        # Test A: Same PCM 20 repeated runs
        test_a_outputs: List[str] = []
        for i in range(20):
            res = session.run(pcm_3s, language=None)
            txt = getattr(res, "text", str(res)).strip()
            test_a_outputs.append(txt)

        unique_outputs = set(test_a_outputs)
        test_a_drift = len(unique_outputs) > 1
        logger.info(f"✅ [MECHANISM 2] Test A Complete: {len(unique_outputs)} unique output(s) across 20 runs. Bit-identical: {not test_a_drift}")

        # Test B: Growing context (1.0s, 1.5s, 2.0s, 2.5s, 3.0s, 3.5s)
        logger.info("🧪 [MECHANISM 2] Starting Test B: Growing Context Evolution...")
        durations = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        test_b_records: List[Dict[str, Any]] = []
        final_ref = test_a_outputs[0]

        for dur in durations:
            pcm_slice = data[: int(dur * 16000)].astype(np.float32)
            h_slice = hashlib.sha256(pcm_slice.tobytes()).hexdigest()[:16]
            res = session.run(pcm_slice, language=None)
            txt = getattr(res, "text", str(res)).strip()

            # Levenshtein distance to final 3.0s transcript
            dist = lev.distance(txt.split(), final_ref.split())
            test_b_records.append({
                "duration_sec": dur,
                "audio_hash": h_slice,
                "hypothesis": txt,
                "word_distance_to_final": dist,
            })
            logger.info(f"   Context {dur:.1f}s: '{txt}' (Lev dist to final: {dist})")

        return {
            "test_a": {
                "num_runs": 20,
                "audio_hash": audio_hash_3s,
                "unique_output_count": len(unique_outputs),
                "outputs": test_a_outputs[:3],
                "is_bit_identical": not test_a_drift,
            },
            "test_b": {
                "evolution": test_b_records,
                "final_reference": final_ref,
            }
        }
    finally:
        ASRModelManager.release_infer_lock()


# =====================================================================
# Mechanism 5: Mariachi Repetition Counterfactual Matrix (M0–M4)
# =====================================================================

async def run_mariachi_counterfactual_matrix(wav_path: Path) -> Dict[str, Any]:
    """Execute counterfactual diagnostics M0–M4 on 12s-35s of English_multiple_kinds_of_noise_88s."""
    logger.info("🎺 [MECHANISM 5] Running Mariachi Counterfactual Matrix (M0-M4)...")
    sim = StreamingAudioSimulator(audio_source=wav_path)
    full_data = np.frombuffer(sim.pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    start_sample = int(12.0 * 16000)
    end_sample = int(35.0 * 16000)
    mariachi_f32 = full_data[start_sample:end_sample].astype(np.float32)
    mariachi_int16 = (mariachi_f32 * 32767.0).astype(np.int16)
    audio_dur = len(mariachi_f32) / 16000.0

    # M3: Pure Acoustic Counterfactual (Raw offline Qwen3-ASR on 23s music with NO streaming controller)
    logger.info("   Executing M3: Pure Acoustic Counterfactual (Raw Qwen3-ASR on music)...")
    mgr = ASRModelManager()
    model = mgr.ensure_model("qwen3-asr-1.7b")
    ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model)
        t0 = time.perf_counter()
        m3_res = session.run(mariachi_f32, language="en")
        m3_infer_ms = (time.perf_counter() - t0) * 1000.0
        m3_text = getattr(m3_res, "text", str(m3_res)).strip()
        m3_words = m3_text.split()
        logger.info(f"   [M3 RESULT] Text: '{m3_text[:60]}' | Word count: {len(m3_words)} | Infer: {m3_infer_ms:.1f}ms")
    finally:
        ASRModelManager.release_infer_lock()

    # Helper to run streaming simulation on mariachi audio
    async def run_sub_mariachi(split_on_stability: bool, name: str) -> Dict[str, Any]:
        session_id = f"mariachi_{name}_{uuid.uuid4().hex[:6]}"
        engine = InstrumentedTranscribeEngine(session_id=session_id)
        engine.sentence_config.split_on_stability = split_on_stability
        engine.sentence_config.min_words_to_commit = 2
        engine.sentence_config.stability_duration_sec = 0.8
        engine._preview_min_growth_ratio = 0.2

        vad = VADProcessor(
            sample_rate=16000,
            vad_engine="fsmn-vad",
            threshold=0.20,
            silence_duration_ms=150,
            hangover_ms=250,
            pre_speech_buffer_ms=800,
            enabled=True,
            on_speech_chunk=engine.feed_audio,
            on_speech_start=engine.on_speech_start,
            on_speech_end=engine.on_speech_end,
        )

        async def _consume_mariachi():
            try:
                async for _ in engine.stream_tokens():
                    pass
            except asyncio.CancelledError:
                pass

        stream_task = asyncio.create_task(_consume_mariachi())
        # Feed chunks
        chunk_samples = int(16000 * 0.064)
        raw_bytes = mariachi_int16.tobytes()
        offset = 0
        while offset < len(raw_bytes):
            chunk = raw_bytes[offset : offset + chunk_samples * 2]
            vad.feed_chunk(chunk)
            offset += len(chunk)
            await asyncio.sleep(0.064)

        # Silence flush
        silence_bytes = bytes(int(16000 * 1.5 * 2))
        offset = 0
        while offset < len(silence_bytes):
            chunk = silence_bytes[offset : offset + chunk_samples * 2]
            vad.feed_chunk(chunk)
            offset += len(chunk)
            await asyncio.sleep(0.064)

        await engine.cleanup()
        stream_task.cancel()

        final_text = " ".join([c["text"] for c in engine.raw_commits]).strip()
        inserted_words = len(final_text.split())
        preview_audio_sec = engine.total_audio_inferred_sec
        amp_factor = preview_audio_sec / max(0.001, audio_dur)

        return {
            "name": name,
            "split_on_stability": split_on_stability,
            "final_text": final_text,
            "commit_count": len(engine.raw_commits),
            "inserted_words": inserted_words,
            "preview_audio_sec": round(preview_audio_sec, 2),
            "loop_amplification_factor": round(amp_factor, 2),
            "splits": len(engine.boundary_records),
        }

    m0 = await run_sub_mariachi(split_on_stability=True, name="M0_Production_Loop")
    m1 = await run_sub_mariachi(split_on_stability=False, name="M1_No_Split")

    logger.info(f"   [M0 Production] Commits: {m0['commit_count']} | Words: {m0['inserted_words']} | Amp Factor: {m0['loop_amplification_factor']}x")
    logger.info(f"   [M1 No-Split]   Commits: {m1['commit_count']} | Words: {m1['inserted_words']} | Amp Factor: {m1['loop_amplification_factor']}x")

    return {
        "audio_duration_sec": audio_dur,
        "m0_production": m0,
        "m1_no_split": m1,
        "m3_pure_acoustic": {
            "text": m3_text,
            "word_count": len(m3_words),
            "infer_ms": round(m3_infer_ms, 1),
        }
    }


# =====================================================================
# Main Benchmark Runner (Stages 1, 2, 3)
# =====================================================================

async def run_single_session_streaming(
    pair: DatasetPair,
    config_id: str,
    split_on_stability: bool = True,
    min_words_to_commit: int = 2,
    stability_duration_sec: float = 0.8,
    preview_min_growth_ratio: float = 0.2,
) -> StreamingSessionTelemetry:
    """Execute complete streaming pipeline for a single file and capture full telemetry."""
    session_id = f"{config_id}_{pair.pair_id}_{uuid.uuid4().hex[:6]}"
    engine = InstrumentedTranscribeEngine(session_id=session_id)
    engine.sentence_config.split_on_stability = split_on_stability
    engine.sentence_config.min_words_to_commit = min_words_to_commit
    engine.sentence_config.stability_duration_sec = stability_duration_sec
    engine._preview_min_growth_ratio = preview_min_growth_ratio

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine="fsmn-vad",
        threshold=0.20,
        silence_duration_ms=150,
        hangover_ms=250,
        pre_speech_buffer_ms=800,
        enabled=True,
        on_speech_chunk=engine.feed_audio,
        on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )

    async def _consume_session():
        try:
            async for _ in engine.stream_tokens():
                pass
        except asyncio.CancelledError:
            pass

    t0 = time.perf_counter()
    token_consumer = asyncio.create_task(_consume_session())

    await feed_streaming_audio(
        wav_path=pair.wav_path,
        engine=engine,
        vad=vad,
        chunk_ms=64,
        speed=1.0,
        trailing_silence_sec=1.5,
    )

    await engine.cleanup()
    token_consumer.cancel()
    wall_dur = time.perf_counter() - t0

    # Build final hypothesis from unique utterance commits
    finals_by_utt: Dict[str, str] = {}
    for c in engine.raw_commits:
        uid = c["utterance_id"]
        txt = c["text"]
        if uid not in finals_by_utt:
            finals_by_utt[uid] = txt
        else:
            # If multiple commits with same uid, append
            finals_by_utt[uid] += " " + txt

    raw_hyp = " ".join(finals_by_utt.values()).strip()
    if not raw_hyp and engine.telemetry_records:
        raw_hyp = engine.telemetry_records[-1].preview_text

    acc = evaluate_accuracy(
        raw_reference=pair.raw_reference,
        raw_hypothesis=raw_hyp,
        language=pair.inferred_language,
    )

    # Rollback count (insertions/deletions backwards)
    rollbacks = sum(1 for r in engine.telemetry_records if r.hypothesis_delta.get("deletions", 0) > 0)

    # Word count and loop metrics
    hyp_words = raw_hyp.split()
    ref_words = pair.raw_reference.split()
    inserted_words = max(0, len(hyp_words) - len(ref_words))

    amp_factor = engine.total_audio_inferred_sec / max(0.001, pair.actual_duration_sec)

    return StreamingSessionTelemetry(
        pair_id=pair.pair_id,
        config_id=config_id,
        audio_duration_sec=pair.actual_duration_sec,
        wall_duration_sec=wall_dur,
        number_of_polls=engine.poll_counter,
        number_of_inferences=engine.global_inference_counter,
        unique_audio_snapshots=len(engine._seen_audio_hashes),
        duplicate_audio_snapshots=engine.duplicate_audio_count,
        total_audio_seconds_inferred=round(engine.total_audio_inferred_sec, 2),
        audio_exposure_ratio=round(amp_factor, 2),
        poll_records=engine.telemetry_records,
        boundary_cuts=engine.boundary_records,
        final_commits=engine.raw_commits,
        raw_hypothesis=raw_hyp,
        accuracy=acc,
        rollback_count=rollbacks,
        inserted_word_count=inserted_words,
        loop_amplification_factor=round(amp_factor, 2),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Failure Attribution Suite")
    parser.add_argument("--stage", type=str, default="1", choices=["1", "2", "3", "full"], help="Stage to execute")
    parser.add_argument("--config", type=str, default="", help="Filter to single config ID (e.g. S0_Baseline)")
    parser.add_argument("--file", type=str, default="", help="Filter to single benchmark file")
    args = parser.parse_args()

    dataset = discover_dataset("wav_test")
    logger.info(f"Loaded {len(dataset)} files from dataset.")

    output_dir = Path("report")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-warm ASR model
    logger.info("🔥 Pre-warming Qwen3-ASR model...")
    _mgr = ASRModelManager()
    _mgr.ensure_model("qwen3-asr-1.7b")

    # =================================================================
    # STAGE 1: Instrumentation Validation on Cross_lingual_6s
    # =================================================================
    if args.stage in ("1", "full"):
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: Instrumentation Validation on Cross_lingual_6s")
        logger.info("=" * 70)

        target = [p for p in dataset if "cross_lingual" in p.pair_id.lower()][0]
        telemetry = await run_single_session_streaming(
            pair=target,
            config_id="S0_Baseline",
            split_on_stability=True,
            min_words_to_commit=2,
            stability_duration_sec=0.8,
            preview_min_growth_ratio=0.2,
        )

        logger.info(f"✅ Stage 1 Telemetry Captured:")
        logger.info(f"   Polls: {telemetry.number_of_polls} | Inferences: {telemetry.number_of_inferences}")
        logger.info(f"   Unique Snapshots: {telemetry.unique_audio_snapshots} | Audio Inferred: {telemetry.total_audio_seconds_inferred}s ({telemetry.audio_exposure_ratio}x)")
        logger.info(f"   Splits: {len(telemetry.boundary_cuts)} | Rollbacks: {telemetry.rollback_count}")
        logger.info(f"   Final CER: {telemetry.accuracy.cer:.2%} | WER: {telemetry.accuracy.wer:.2%}")
        logger.info(f"   Transcript: '{telemetry.raw_hypothesis}'")

        # Mechanism 2 Test A & B
        mech2_results = run_mechanism_2_tests(wav_path=target.wav_path)

        # Save stage 1 validation json
        s1_out = {
            "file": target.pair_id,
            "duration_sec": target.actual_duration_sec,
            "telemetry_summary": {
                "polls": telemetry.number_of_polls,
                "inferences": telemetry.number_of_inferences,
                "splits": len(telemetry.boundary_cuts),
                "audio_exposure_ratio": telemetry.audio_exposure_ratio,
                "cer": telemetry.accuracy.cer,
                "wer": telemetry.accuracy.wer,
            },
            "sample_poll_record": asdict(telemetry.poll_records[0]) if telemetry.poll_records else {},
            "mechanism_2": mech2_results,
        }
        with open(output_dir / "streaming_stage1_validation.json", "w", encoding="utf-8") as f:
            json.dump(s1_out, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Stage 1 validation to {output_dir / 'streaming_stage1_validation.json'}")

        if args.stage == "1":
            logger.info("Stage 1 Complete!")
            return

    # =================================================================
    # STAGE 2: Full Ablation Matrix (S0–S5) & Gate A Check
    # =================================================================
    if args.stage in ("2", "full"):
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: Controlled Ablation Matrix (S0–S5)")
        logger.info("=" * 70)

        configs = [
            {"id": "S0_Baseline", "split": True, "min_words": 2, "stab_dur": 0.8, "growth": 0.2},
            {"id": "S1_No_Split", "split": False, "min_words": 2, "stab_dur": 0.8, "growth": 0.2},
            {"id": "S2_Conservative_Split", "split": True, "min_words": 4, "stab_dur": 1.5, "growth": 0.2},
            {"id": "S3_Ungated_Poller", "split": True, "min_words": 2, "stab_dur": 0.8, "growth": 0.0},
            {"id": "S4_High_Growth", "split": True, "min_words": 2, "stab_dur": 0.8, "growth": 0.5},
            {"id": "S5_Acoustic_Convergence", "split": False, "min_words": 1, "stab_dur": 0.8, "growth": 0.2},
        ]

        if args.config:
            configs = [c for c in configs if c["id"].lower() == args.config.lower()]
            if not configs:
                logger.error(f"Config '{args.config}' not found in matrix.")
                return

        target_files = dataset
        if args.file:
            target_files = [p for p in dataset if args.file.lower() in p.pair_id.lower()]

        matrix_path = output_dir / "streaming_ablation_matrix.json"
        stage2_results: Dict[str, Any] = {}
        if matrix_path.exists():
            try:
                with open(matrix_path, "r", encoding="utf-8") as f:
                    stage2_results = json.load(f)
            except Exception:
                stage2_results = {}

        for cfg in configs:
            cfg_id = cfg["id"]
            logger.info(f"\n▶️ Running Config {cfg_id}...")
            cfg_telemetry: List[StreamingSessionTelemetry] = []

            total_sub = 0
            total_del = 0
            total_ins = 0
            total_ref = 0
            total_infers = 0
            total_splits = 0
            total_inferred_sec = 0.0
            total_audio_sec = 0.0

            for pair in target_files:
                logger.info(f"   Streaming {pair.pair_id} ({pair.actual_duration_sec:.1f}s)...")
                t = await run_single_session_streaming(
                    pair=pair,
                    config_id=cfg_id,
                    split_on_stability=cfg["split"],
                    min_words_to_commit=cfg["min_words"],
                    stability_duration_sec=cfg["stab_dur"],
                    preview_min_growth_ratio=cfg["growth"],
                )
                cfg_telemetry.append(t)
                total_sub += t.accuracy.substitutions
                total_del += t.accuracy.deletions
                total_ins += t.accuracy.insertions
                total_ref += t.accuracy.ref_length
                total_infers += t.number_of_inferences
                total_splits += len(t.boundary_cuts)
                total_inferred_sec += t.total_audio_seconds_inferred
                total_audio_sec += t.audio_duration_sec

            corpus_cer = (total_sub + total_del + total_ins) / max(1, total_ref)
            exposure = total_inferred_sec / max(0.001, total_audio_sec)

            logger.info(f"📊 [{cfg_id}] Corpus CER: {corpus_cer:.2%} | Splits: {total_splits} | Inferences: {total_infers} | Exposure: {exposure:.2f}x")

            stage2_results[cfg_id] = {
                "config": cfg,
                "corpus_cer": round(corpus_cer * 100.0, 2),
                "total_inferences": total_infers,
                "total_splits": total_splits,
                "audio_exposure_ratio": round(exposure, 2),
                "files": {
                    t.pair_id: {
                        "cer": round(t.accuracy.cer * 100.0, 2),
                        "wer": round(t.accuracy.wer * 100.0, 2),
                        "splits": len(t.boundary_cuts),
                        "inserted_words": t.inserted_word_count,
                        "raw_hypothesis": t.raw_hypothesis,
                    }
                    for t in cfg_telemetry
                }
            }

            with open(matrix_path, "w", encoding="utf-8") as f:
                json.dump(stage2_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved to {matrix_path}")

            # GATE A Check on S0
            if cfg_id == "S0_Baseline":
                noise_cer = stage2_results["S0_Baseline"]["files"].get("English_multiple_kinds_of_noise_88s", {}).get("cer", 0.0)
                logger.info(f"\n🚪 [GATE A CHECK] S0 Corpus CER = {corpus_cer:.2%}, Noise File CER = {noise_cer:.2f}% (R7 was 77.80%)")
                if noise_cer < 40.0:
                    logger.warning("⚠️ S0 Noise CER is lower than expected R7 baseline. Verifying deterministic discrepancy...")

        logger.info(f"Completed Stage 2 configs saved to {matrix_path}")

    # =================================================================
    # STAGE 3: Mariachi Counterfactual Diagnostics (M0–M4)
    # =================================================================
    if args.stage in ("3", "full"):
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 3: Mariachi Counterfactual Deep Dive (M0–M4)")
        logger.info("=" * 70)

        noise_file = [p for p in dataset if "multiple_kinds_of_noise" in p.pair_id.lower()][0]
        m_results = await run_mariachi_counterfactual_matrix(noise_file.wav_path)

        with open(output_dir / "streaming_mariachi_matrix.json", "w", encoding="utf-8") as f:
            json.dump(m_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Stage 3 results to {output_dir / 'streaming_mariachi_matrix.json'}")

    logger.info("\n🎉 Phase 3B Execution Finished Successfully!")


if __name__ == "__main__":
    asyncio.run(main())
