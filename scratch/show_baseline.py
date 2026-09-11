"""Print the key W2.1 baseline numbers in a compact, readable form."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
baseline = json.loads((PROJECT_ROOT / "report" / "baseline_perf.json").read_text(encoding="utf-8"))

print("=" * 78)
print("W2.1 BASELINE SUMMARY")
print("=" * 78)
env = baseline["environment"]
print(f"ASR={env.get('asr_model')} backend={env.get('asr_backend')} threads={env.get('asr_threads')}")
print(f"VAD={env.get('vad_engine')} poll={env.get('poll_interval_ms')}ms "
      f"translation={env.get('translation_base')} max_sessions={env.get('max_sessions')}")
print(f"GPU={env.get('gpu')} VRAM={env.get('vram_total_mb')}MB torch={env.get('torch')}")

micro = baseline.get("micro_benchmarks", {})

print("\n--- A/C (CPU side): work done before every ASR inference ---")
snap = micro.get("snapshot_and_normalize", {})
for row in snap.get("rows", []):
    print(f"  {row['audio_sec']:>4.0f}s | snapshot {row['snapshot']['median_ms']:>6.2f} ms "
          f"({row['snapshot']['peak_alloc_bytes']/1024:>8.1f} KiB) | "
          f"normalize {row['normalize']['median_ms']:>5.2f} ms "
          f"({row['normalize']['peak_alloc_bytes']/1024:>8.1f} KiB) | "
          f"pre-infer {row['pre_infer_ms']:>5.2f} ms")
print("  scaling:", json.dumps(snap.get("scaling", {})))

print("\n--- C: copy chain (1s of audio) ---")
for stage in micro.get("copy_chain", {}).get("stages", []):
    tag = " [reference]" if stage.get("is_reference_only") else ""
    print(f"  {stage['stage']:<52} {stage['median_ms']:>7.2f} ms{tag}")

print("\n--- VAD: real cost per engine (this is the CPU-heavy stage) ---")
for row in micro.get("vad_cost", {}).get("rows", []):
    print(f"  {row['engine']:<12} frame={row['native_frame_samples']:>4} "
          f"({row['frames_per_audio_sec']:>5.1f} fps) | {row['ms_per_audio_sec']:>6.1f} ms per audio-sec "
          f"| RTF {row['rtf']:.3f} | speech frames -> ASR: {row['speech_frames_passed_to_asr']}")

e2e = baseline.get("end_to_end_scenarios", {})
print("\n--- A (end-to-end): ASR cost vs utterance length ---")
pc = e2e.get("preview_cost", {})
print(f"  validity: {json.dumps(pc.get('validity', {}))}")
print(f"  client events: {json.dumps(pc.get('client_events', {}))}")
for row in pc.get("rows", []):
    s = row["infer_ms"]
    if s.get("count"):
        print(f"  {row['audio_sec']:>4.0f}s | n={s['count']:>3} | infer avg {s['avg']:>8.1f} ms "
              f"p50 {s['p50']:>8.1f} p90 {s['p90']:>8.1f} max {s['max']:>8.1f} | "
              f"rtf avg {row['rtf'].get('avg', 0):.3f}")
    else:
        print(f"  {row['audio_sec']:>4.0f}s | no samples")
print("  infer cost scaling (longest/shortest):", pc.get("infer_cost_scaling_longest_over_shortest"))
print("  counters:", json.dumps(pc.get("counters", {})))

print("\n--- D: backpressure ---")
bp = e2e.get("backpressure", {})
print(f"  validity: {json.dumps(bp.get('validity', {}))}")
print(f"  delivery: {json.dumps(bp.get('delivery', {}))}")
print(f"  resources: {json.dumps(bp.get('resources', {}))}")
print(f"  counter delta: {json.dumps(bp.get('counter_delta', {}))}")

print("\n--- G: session scaling ---")
sc = e2e.get("session_scaling", {})
for row in sc.get("rows", []):
    lw = row["lock_wait_ms"]
    inf = row["infer_ms"]
    print(f"  sessions={row['sessions']} | wall {row['wall_sec']:>5.1f}s | "
          f"lock_wait n={lw.get('count', 0):>3} p50 {lw.get('p50', 0):>7.1f} "
          f"p99 {lw.get('p99', 0):>7.1f} max {lw.get('max', 0):>7.1f} | "
          f"infer avg {inf.get('avg', 0):>7.1f} | rss {row['ram_rss_mb']} MB | "
          f"valid={row.get('validity', {}).get('valid')}")

print("\n--- R: REAL SPEECH end-to-end (the only delivery-capable scenario) ---")
ra = e2e.get("real_audio", {})
if ra:
    print(f"  audio: {ra.get('audio_file')}  ({ra.get('audio_sec')}s @ {ra.get('speed')}x, tts={ra.get('tts_enabled')})")
    print(f"  wall: {ra.get('wall_sec')}s | validity: {json.dumps(ra.get('validity', {}))}")
    print(f"  delivery: {json.dumps(ra.get('delivery', {}))}")
    a = ra.get("asr", {})
    t = ra.get("translation", {})
    print(f"  asr infer_ms:   {json.dumps(a.get('infer_ms', {}))}")
    print(f"  asr rtf:        {json.dumps(a.get('rtf', {}))}")
    print(f"  asr lock_wait:  {json.dumps(a.get('lock_wait_ms', {}))}")
    print(f"  translation queue_wait: {json.dumps(t.get('queue_wait_ms', {}))}")
    print(f"  translation infer_ms:   {json.dumps(t.get('infer_ms', {}))}")
    print(f"  resources: {json.dumps(ra.get('resources', {}))}")
    print(f"  counters: {json.dumps(ra.get('counters', {}))}")
    print("  reference transcripts (W2.2 equivalence gate baseline):")
    for item in ra.get("transcripts", [])[:16]:
        print(f"    - {item.get('text', '')!r}")
else:
    print("  (not run)")
print("=" * 78)
