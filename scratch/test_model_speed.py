import time
from pathlib import Path
import asyncio
from benchmarks.namo_benchmark import run_single_benchmark, parse_ground_truth, DEFAULT_WAV, DEFAULT_TXT

turns = parse_ground_truth(DEFAULT_TXT)
for model in ['sensevoice-small', 'nemotron-3.5-streaming']:
    t0 = time.perf_counter()
    print(f'Starting test for {model}...')
    res = asyncio.run(run_single_benchmark(
        DEFAULT_WAV, turns,
        vad_engine='fsmn-vad', threshold=0.45, silence_ms=500, hangover_ms=300, pre_speech_ms=120,
        namo_enabled=True, namo_threshold=0.70, namo_require_silence_ms=120,
        model_name=model,
    ))
    dur = time.perf_counter() - t0
    print(f'Done {model} in {dur:.1f}s | CER Strict: {res["cer_strict"]}% | Namo commits: {res["namo_commits"]}/{res["total_commits"]} ({res["namo_ratio_pct"]}%)')
