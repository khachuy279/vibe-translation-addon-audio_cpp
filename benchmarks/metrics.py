"""Event timeline, latency metrics (TTFS, RTF, E2E), and subtitle quality audit."""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimelineEvent:
    """A timestamped lifecycle event in the audio/subtitle pipeline."""
    event_type: str
    monotonic_ts: float
    sample_position: int
    audio_timestamp_sec: float
    wall_clock_ts: float
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "monotonic_ts": round(self.monotonic_ts, 4),
            "sample_position": self.sample_position,
            "audio_timestamp_sec": round(self.audio_timestamp_sec, 3),
            "wall_clock_ts": round(self.wall_clock_ts, 4),
            "data": self.data,
        }


@dataclass
class SubtitleQualityReport:
    """Subtitle delivery stability and quality metrics."""
    total_partials: int
    total_finals: int
    unique_finals: int
    duplicate_finals_count: int
    rollback_count: int
    empty_finals_count: int
    avg_partial_cadence_ms: float
    max_partial_gap_ms: float
    rollback_examples: List[str]
    duplicate_examples: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_partials": self.total_partials,
            "total_finals": self.total_finals,
            "unique_finals": self.unique_finals,
            "duplicate_finals_count": self.duplicate_finals_count,
            "rollback_count": self.rollback_count,
            "empty_finals_count": self.empty_finals_count,
            "avg_partial_cadence_ms": round(self.avg_partial_cadence_ms, 1),
            "max_partial_gap_ms": round(self.max_partial_gap_ms, 1),
            "rollback_examples": self.rollback_examples[:3],
            "duplicate_examples": self.duplicate_examples[:3],
        }


@dataclass
class QueueReport:
    """Detailed queue depth, throughput, and backpressure metrics."""
    max_queue_depth: int
    avg_queue_depth: float
    queue_drain_time_ms: float
    producer_rate_items_per_sec: float
    consumer_rate_items_per_sec: float
    pct_time_queue_non_empty: float
    max_queue_wait_ms: float
    avg_queue_wait_ms: float
    queue_full_dropped: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_queue_depth": self.max_queue_depth,
            "avg_queue_depth": round(self.avg_queue_depth, 2),
            "queue_drain_time_ms": round(self.queue_drain_time_ms, 1),
            "producer_rate_items_per_sec": round(self.producer_rate_items_per_sec, 2),
            "consumer_rate_items_per_sec": round(self.consumer_rate_items_per_sec, 2),
            "pct_time_queue_non_empty": round(self.pct_time_queue_non_empty, 1),
            "max_queue_wait_ms": round(self.max_queue_wait_ms, 1),
            "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 1),
            "queue_full_dropped": self.queue_full_dropped,
        }


@dataclass
class LatencyReport:
    """Complete latency and Real-Time Factor (RTF) metrics with terminology distinction."""
    ttfs_ms: Optional[float]                 # Time To First Subtitle (from first speech chunk to first partial subtitle)
    final_latency_ms: Optional[float]        # Latency from speech end to final committed subtitle
    e2e_total_duration_ms: float             # Total test duration from first chunk to pipeline drain
    audio_duration_sec: float                # Actual audio duration
    rtf_asr: float                           # ASR Compute RTF = pure ASR inference time / audio duration
    rtf_pipeline: float                      # Pipeline RTF = (wall duration during active streaming) / audio duration
    rtf_e2e: float                           # End-to-end RTF = total wall duration (inc. trailing silence + drain) / audio duration
    avg_asr_infer_ms: float                  # Average ASR inference duration in ms
    total_inferences: int                    # Total inferences (previews + commits)
    queue: QueueReport                       # Comprehensive queue performance report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttfs_ms": round(self.ttfs_ms, 1) if self.ttfs_ms is not None else None,
            "final_latency_ms": round(self.final_latency_ms, 1) if self.final_latency_ms is not None else None,
            "e2e_total_duration_ms": round(self.e2e_total_duration_ms, 1),
            "audio_duration_sec": round(self.audio_duration_sec, 2),
            "rtf_asr": round(self.rtf_asr, 3),
            "rtf_pipeline": round(self.rtf_pipeline, 3),
            "rtf_e2e": round(self.rtf_e2e, 3),
            "avg_asr_infer_ms": round(self.avg_asr_infer_ms, 1),
            "total_inferences": self.total_inferences,
            "queue": self.queue.to_dict(),
        }


class TimelineCollector:
    """Collects chronological events and computes latency and quality metrics."""

    def __init__(self, audio_duration_sec: float):
        self.audio_duration_sec = audio_duration_sec
        self.events: List[TimelineEvent] = []
        self.stream_start_mono: float = 0.0
        self.first_speech_chunk_mono: Optional[float] = None
        self.last_speech_chunk_mono: Optional[float] = None
        self.first_subtitle_mono: Optional[float] = None
        self.last_final_subtitle_mono: Optional[float] = None

        # Received subtitles ordered
        self.partials: List[Dict[str, Any]] = []
        self.finals: List[Dict[str, Any]] = []

    def record_event(
        self,
        event_type: str,
        sample_position: int,
        audio_timestamp_sec: float,
        data: Optional[Dict[str, Any]] = None,
    ) -> TimelineEvent:
        now_mono = time.perf_counter()
        now_wall = time.time()
        ev = TimelineEvent(
            event_type=event_type,
            monotonic_ts=now_mono,
            sample_position=sample_position,
            audio_timestamp_sec=audio_timestamp_sec,
            wall_clock_ts=now_wall,
            data=data or {},
        )
        self.events.append(ev)
        return ev

    def note_stream_started(self) -> None:
        self.stream_start_mono = time.perf_counter()
        self.record_event("stream_started", 0, 0.0)

    def note_chunk_sent(self, sample_idx: int, audio_ts: float, is_silence: bool) -> None:
        now = time.perf_counter()
        if not is_silence:
            if self.first_speech_chunk_mono is None:
                self.first_speech_chunk_mono = now
            self.last_speech_chunk_mono = now

        self.record_event(
            "chunk_sent",
            sample_position=sample_idx,
            audio_timestamp_sec=audio_ts,
            data={"is_silence": is_silence},
        )

    def note_subtitle_received(self, msg: Dict[str, Any]) -> None:
        now = time.perf_counter()
        is_final = bool(msg.get("is_final"))
        text = (msg.get("text") or "").strip()
        utt_id = msg.get("utterance_id", "")

        msg_record = {
            "text": text,
            "utterance_id": utt_id,
            "is_final": is_final,
            "rx_mono": now,
        }

        if self.first_subtitle_mono is None and text:
            self.first_subtitle_mono = now

        if is_final:
            self.finals.append(msg_record)
            self.last_final_subtitle_mono = now
            self.record_event(
                "subtitle_final",
                sample_position=-1,
                audio_timestamp_sec=-1.0,
                data=msg_record,
            )
        else:
            self.partials.append(msg_record)
            self.record_event(
                "subtitle_partial",
                sample_position=-1,
                audio_timestamp_sec=-1.0,
                data=msg_record,
            )

    def analyze_quality(self) -> SubtitleQualityReport:
        """Inspect partials and finals for duplicates, rollbacks, and gaps."""
        # 1. Duplicates check
        seen_final_texts: Dict[str, int] = {}
        seen_utt_ids: set = set()
        dups_count = 0
        dup_examples: List[str] = []

        for f in self.finals:
            txt = f["text"]
            utt_id = f["utterance_id"]
            if not txt:
                continue

            # Check if this exact text was already emitted for a DIFFERENT utterance ID
            if utt_id not in seen_utt_ids:
                seen_utt_ids.add(utt_id)
                seen_final_texts[txt] = seen_final_texts.get(txt, 0) + 1
                if seen_final_texts[txt] > 1:
                    dups_count += 1
                    dup_examples.append(txt)

        # 2. Rollback check on partials within the same utterance
        rollbacks = 0
        rollback_examples: List[str] = []
        partials_by_utt: Dict[str, List[str]] = {}
        for p in self.partials:
            uid = p["utterance_id"]
            partials_by_utt.setdefault(uid, []).append(p["text"])

        for uid, texts in partials_by_utt.items():
            prev = ""
            for cur in texts:
                # If current text is strictly shorter than previous and previous is not empty
                if len(cur) < len(prev) and not cur.startswith(prev[:len(cur)]):
                    rollbacks += 1
                    rollback_examples.append(f"'{prev}' -> '{cur}'")
                prev = cur

        # 3. Partial cadence
        gaps: List[float] = []
        for i in range(1, len(self.partials)):
            g = (self.partials[i]["rx_mono"] - self.partials[i - 1]["rx_mono"]) * 1000.0
            gaps.append(g)

        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        max_gap = max(gaps) if gaps else 0.0

        empty_finals = sum(1 for f in self.finals if not f["text"])

        return SubtitleQualityReport(
            total_partials=len(self.partials),
            total_finals=len(self.finals),
            unique_finals=len(seen_utt_ids),
            duplicate_finals_count=dups_count,
            rollback_count=rollbacks,
            empty_finals_count=empty_finals,
            avg_partial_cadence_ms=avg_gap,
            max_partial_gap_ms=max_gap,
            rollback_examples=rollback_examples,
            duplicate_examples=dup_examples,
        )

    def compute_latency(self, server_telemetry: Optional[Dict[str, Any]] = None) -> LatencyReport:
        """Compute latency indicators and Real-Time Factor."""
        t_end = time.perf_counter()
        wall_dur_sec = max(0.001, t_end - self.stream_start_mono)
        audio_dur = max(0.001, self.audio_duration_sec)

        # TTFS: Time from speech start to first subtitle
        ref_start = self.first_speech_chunk_mono or self.stream_start_mono
        ttfs_ms = ((self.first_subtitle_mono - ref_start) * 1000.0) if self.first_subtitle_mono else None

        # Final subtitle latency: from last speech chunk sent to final subtitle received
        final_lat_ms = None
        if self.last_speech_chunk_mono and self.last_final_subtitle_mono:
            final_lat_ms = max(0.0, (self.last_final_subtitle_mono - self.last_speech_chunk_mono) * 1000.0)

        # Server metrics if available
        telemetry = server_telemetry or {}
        counters = telemetry.get("counters", {})
        metrics = telemetry.get("metrics", {})

        total_infer = counters.get("asr.total_inferences", 0)
        infer_stat = metrics.get("asr", {}).get("infer_ms", {})
        avg_infer = infer_stat.get("avg", infer_stat.get("mean", 0.0))
        queue_stat = metrics.get("translation", {}).get("queue_wait_ms", {})
        max_queue = float(queue_stat.get("max", 0.0))
        avg_queue = float(queue_stat.get("avg", queue_stat.get("mean", 0.0)))
        queue_dropped = int(counters.get("translation.queue_full_dropped", 0))

        # Total ASR compute time (seconds)
        total_asr_time_sec = (total_infer * avg_infer) / 1000.0 if total_infer and avg_infer else 0.0
        rtf_asr = total_asr_time_sec / audio_dur if total_asr_time_sec > 0 else 0.0
        rtf_pipeline = max(rtf_asr, (wall_dur_sec - audio_dur) / audio_dur) if wall_dur_sec > audio_dur else rtf_asr
        rtf_e2e = wall_dur_sec / audio_dur

        # Queue telemetry calculation
        drain_time_ms = max(0.0, (t_end - (self.last_speech_chunk_mono or self.stream_start_mono)) * 1000.0)
        total_finals = len(self.finals)
        producer_rate = float(total_finals) / wall_dur_sec if wall_dur_sec > 0 else 0.0
        consumer_rate = float(total_finals) / max(0.001, (wall_dur_sec - (ttfs_ms or 0.0) / 1000.0))

        # Estimate queue depth based on item wait times vs throughput
        estimated_max_depth = min(20, int(math.ceil(max_queue / max(1.0, avg_infer)))) if max_queue > 0 else 0
        estimated_avg_depth = min(20.0, float(avg_queue / max(1.0, avg_infer))) if avg_queue > 0 else 0.0
        pct_non_empty = min(100.0, (float(total_finals) * avg_queue / (wall_dur_sec * 1000.0)) * 100.0) if wall_dur_sec > 0 else 0.0

        q_report = QueueReport(
            max_queue_depth=estimated_max_depth,
            avg_queue_depth=estimated_avg_depth,
            queue_drain_time_ms=drain_time_ms,
            producer_rate_items_per_sec=producer_rate,
            consumer_rate_items_per_sec=consumer_rate,
            pct_time_queue_non_empty=pct_non_empty,
            max_queue_wait_ms=max_queue,
            avg_queue_wait_ms=avg_queue,
            queue_full_dropped=queue_dropped,
        )

        return LatencyReport(
            ttfs_ms=ttfs_ms,
            final_latency_ms=final_lat_ms,
            e2e_total_duration_ms=wall_dur_sec * 1000.0,
            audio_duration_sec=audio_dur,
            rtf_asr=rtf_asr,
            rtf_pipeline=rtf_pipeline,
            rtf_e2e=rtf_e2e,
            avg_asr_infer_ms=avg_infer,
            total_inferences=total_infer,
            queue=q_report,
        )
