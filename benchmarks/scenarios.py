"""Benchmark test configurations: Baseline, Low Latency, High Accuracy, Balanced, and Stress."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioConfig:
    """Benchmark test execution scenario parameters."""
    scenario_id: str
    name: str
    description: str
    asr_model: str
    vad_engine: str
    vad_threshold: float
    silence_duration_ms: int
    hangover_ms: int
    chunk_ms: int
    speed: float = 1.0
    preview_min_growth_ratio: float = 0.0
    normalize_speech: bool = True
    normalize_target_rms: float = 0.10
    split_on_stability: bool = True
    min_words_to_commit: int = 2
    extra_session_config: Dict[str, Any] = field(default_factory=dict)

    def to_ws_config(self) -> Dict[str, Any]:
        """Convert scenario parameters to backend set_config WebSocket payload."""
        cfg = {
            "modelId": self.asr_model,
            "vadEngine": self.vad_engine,
            "vadThreshold": self.vad_threshold,
            "silenceDurationMs": self.silence_duration_ms,
            "hangoverMs": self.hangover_ms,
            "minWordsToCommit": self.min_words_to_commit,
            "splitOnStability": self.split_on_stability,
            "preview_min_growth_ratio": self.preview_min_growth_ratio,
            "ttsEnabled": False,
            **self.extra_session_config,
        }
        return cfg


def get_standard_scenarios() -> Dict[str, ScenarioConfig]:
    """Return standard test scenarios conforming to specifications."""
    return {
        "baseline": ScenarioConfig(
            scenario_id="baseline",
            name="Test A — Baseline (preview_growth=0.2)",
            description="Current production config: qwen3-asr-1.7b, fsmn-vad, silence=150ms, chunk=64ms, preview_growth=0.2",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=150,
            hangover_ms=400,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
            normalize_speech=True,
            normalize_target_rms=0.10,
        ),
        "baseline_ungated": ScenarioConfig(
            scenario_id="baseline_ungated",
            name="Test A0 — Baseline Ungated (preview_growth=0.0)",
            description="Pre-optimization ungated baseline: preview_growth=0.0",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=150,
            hangover_ms=400,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.0,
            normalize_speech=True,
            normalize_target_rms=0.10,
        ),
        "low_latency": ScenarioConfig(
            scenario_id="low_latency",
            name="Test B — Low Latency",
            description="Aggressive low latency: silero-vad, silence=100ms, chunk=40ms, sensevoice-small",
            asr_model="sensevoice-small",
            vad_engine="silero-vad",
            vad_threshold=0.35,
            silence_duration_ms=100,
            hangover_ms=200,
            chunk_ms=40,
            speed=1.0,
            preview_min_growth_ratio=0.3,
            normalize_speech=True,
            normalize_target_rms=0.10,
        ),
        "high_accuracy": ScenarioConfig(
            scenario_id="high_accuracy",
            name="Test C — High Accuracy",
            description="High accuracy: qwen3-asr-1.7b, firered-vad, silence=300ms, chunk=64ms",
            asr_model="qwen3-asr-1.7b",
            vad_engine="firered-vad",
            vad_threshold=0.5,
            silence_duration_ms=300,
            hangover_ms=400,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.0,
            normalize_speech=True,
            normalize_target_rms=0.10,
        ),
        "balanced": ScenarioConfig(
            scenario_id="balanced",
            name="Test D — Balanced",
            description="Optimal trade-off: qwen3-asr-1.7b, fsmn-vad, silence=150ms, preview growth ratio 0.2",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=150,
            hangover_ms=300,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
            normalize_speech=True,
            normalize_target_rms=0.10,
        ),
        # Silence duration matrix benchmarks (100 / 120 / 150 / 180 / 200 ms)
        "silence_100ms": ScenarioConfig(
            scenario_id="silence_100ms",
            name="Silence Matrix — 100ms",
            description="Silence duration 100ms: evaluates early commit and fragmentation",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=100,
            hangover_ms=200,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
        ),
        "silence_120ms": ScenarioConfig(
            scenario_id="silence_120ms",
            name="Silence Matrix — 120ms",
            description="Silence duration 120ms: candidate low latency commit",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=120,
            hangover_ms=240,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
        ),
        "silence_150ms": ScenarioConfig(
            scenario_id="silence_150ms",
            name="Silence Matrix — 150ms",
            description="Silence duration 150ms: current baseline reference",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=150,
            hangover_ms=300,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
        ),
        "silence_180ms": ScenarioConfig(
            scenario_id="silence_180ms",
            name="Silence Matrix — 180ms",
            description="Silence duration 180ms: conversational pause tolerance",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=180,
            hangover_ms=360,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
        ),
        "silence_200ms": ScenarioConfig(
            scenario_id="silence_200ms",
            name="Silence Matrix — 200ms",
            description="Silence duration 200ms: maximum conversational stability",
            asr_model="qwen3-asr-1.7b",
            vad_engine="fsmn-vad",
            vad_threshold=0.4,
            silence_duration_ms=200,
            hangover_ms=400,
            chunk_ms=64,
            speed=1.0,
            preview_min_growth_ratio=0.2,
        ),
    }
