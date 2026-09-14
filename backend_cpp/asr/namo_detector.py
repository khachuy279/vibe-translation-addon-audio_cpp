"""Namo Turn Detector v1 integration module for backend_cpp.

Provides semantic end-of-utterance (EOU) detection using VideoSDK's Namo Turn Detector
Multilingual ONNX model (mmBERT architecture).

Enables the ASR streaming pipeline to commit complete grammatical sentences immediately
upon semantic completion, prior to VAD acoustic silence timeouts.
"""

import logging
from pathlib import Path
import threading
import time
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from backend_cpp.asr.sentence_segmenter import count_content_tokens

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NAMO_DIR = PROJECT_ROOT / "backend_cpp" / "models" / "namo"
DEFAULT_NAMO_REPO = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
DEFAULT_NAMO_MODEL_FILE = "model_quant.onnx"
DEFAULT_NAMO_MAX_LENGTH = 8192
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MIN_TOKENS = 3

REQUIRED_NAMO_FILES = (
    "model_quant.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class NamoTurnDetector:
    """Manages ONNX inference session and tokenization for Namo Turn Detector v1."""

    _shared_instance: Optional["NamoTurnDetector"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        repo_id: str = DEFAULT_NAMO_REPO,
        model_dir: Union[str, Path] = DEFAULT_NAMO_DIR,
        model_filename: str = DEFAULT_NAMO_MODEL_FILE,
        max_length: int = DEFAULT_NAMO_MAX_LENGTH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ) -> None:
        self.repo_id = repo_id
        self.model_dir = Path(model_dir).resolve()
        self.model_filename = model_filename
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.min_tokens = min_tokens

        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._infer_lock: threading.Lock = threading.Lock()

        self._load_model()

    def _load_model(self) -> None:
        """Download (if needed) into local model_dir and initialize tokenizer and ONNX Runtime session."""
        try:
            t0 = time.perf_counter()
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[NamoTurnDetector] Loading model from '{self.model_dir}' (repo='{self.repo_id}')...")

            model_path = self.model_dir / self.model_filename

            # Ensure all required model and tokenizer artifacts are present in local directory
            for fname in REQUIRED_NAMO_FILES:
                target_path = self.model_dir / fname
                if not target_path.exists():
                    logger.info(f"[NamoTurnDetector] Downloading {fname} to {self.model_dir}...")
                    hf_hub_download(
                        repo_id=self.repo_id,
                        filename=fname,
                        local_dir=str(self.model_dir),
                    )

            # Load tokenizer directly from local model directory
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            load_dur = time.perf_counter() - t0
            logger.info(f"[NamoTurnDetector] Initialized successfully from {self.model_dir} in {load_dur:.2f}s.")
            self._warmup()
        except Exception as e:
            logger.error(f"[NamoTurnDetector] Failed to load model from {self.model_dir}: {e}", exc_info=True)
            self._session = None
            self._tokenizer = None

    def _warmup(self) -> None:
        """Run a warmup inference to eliminate first-token cold-start latency."""
        if self._session is None or self._tokenizer is None:
            return
        try:
            t0 = time.perf_counter()
            self.predict_eou("Hello world, this is a warmup check.")
            warmup_dur = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"[NamoTurnDetector] Warmup completed in {warmup_dur:.1f}ms.")
        except Exception as e:
            logger.warning(f"[NamoTurnDetector] Warmup failed: {e}")

    @classmethod
    def get_shared_instance(
        cls,
        repo_id: str = DEFAULT_NAMO_REPO,
        model_dir: Optional[Union[str, Path]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ) -> "NamoTurnDetector":
        """Thread-safe singleton accessor for shared detector instance."""
        with cls._init_lock:
            resolved_dir = Path(model_dir).resolve() if model_dir else DEFAULT_NAMO_DIR
            if cls._shared_instance is None or cls._shared_instance.model_dir != resolved_dir:
                cls._shared_instance = cls(
                    repo_id=repo_id,
                    model_dir=resolved_dir,
                    confidence_threshold=confidence_threshold,
                    min_tokens=min_tokens,
                )
            else:
                cls._shared_instance.confidence_threshold = confidence_threshold
                cls._shared_instance.min_tokens = min_tokens
            return cls._shared_instance

    def predict_eou(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
        min_tokens: Optional[int] = None,
    ) -> Tuple[bool, float]:
        """Predict whether the input text marks the semantic End of Utterance (EOU).

        Args:
            text: Partial or candidate transcript text from ASR.
            confidence_threshold: Optional override for decision threshold.
            min_tokens: Optional override for minimum content tokens.

        Returns:
            Tuple[bool, float]: (is_eou, eou_probability)
                - is_eou: True if semantic end of utterance detected with confidence >= threshold.
                - eou_probability: Model confidence score for class 1 (End of Turn), range [0.0, 1.0].
        """
        if not text or not text.strip():
            return False, 0.0

        clean_text = text.strip()
        effective_min_tokens = min_tokens if min_tokens is not None else self.min_tokens
        token_count = count_content_tokens(clean_text)
        if token_count < effective_min_tokens:
            return False, 0.0

        if self._session is None or self._tokenizer is None:
            return False, 0.0

        effective_thresh = (
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )

        try:
            with self._infer_lock:
                inputs = self._tokenizer(
                    clean_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np",
                )
                feed_dict = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }
                outputs = self._session.run(None, feed_dict)

            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            pred_label = int(np.argmax(probs))
            p_eou = float(probs[1]) if len(probs) > 1 else 0.0

            is_eou = (pred_label == 1 and p_eou >= effective_thresh)
            return is_eou, p_eou

        except Exception as e:
            logger.error(f"[NamoTurnDetector] Inference error: {e}")
            return False, 0.0
