"""Does min_words_to_commit silently drop short Japanese utterances?

`count_content_tokens` counts CJK characters individually. Japanese drama/anime
dialogue is full of very short complete turns ("うん。", "そうね。", "どうした。",
"いや大丈夫。"). If the commit filter requires 4 content tokens, those turns are
dropped outright -- a guaranteed deletion, independent of model quality.

This exercises the real production functions, not a reimplementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens as cct  # noqa: E402
from backend_cpp.config import config  # noqa: E402

REAL_TURNS = [
    "うん。",
    "そうね。",
    "どうした。",
    "いや大丈夫。",
    "はい、ありがとう。",
    "どうぞ。",
    "いってらっしゃい！",
    "そう。",
    "はい。",
    "なるほど。",
    "まあ今日はゆっくりしね。",
    "親父も結局のところお袋のことが好きでしょうがないんよな。",
]

EN_TURNS = [
    "Yes.",
    "I see.",
    "OK.",
    "I'm fine.",
    "Yes, thank you.",
    "Let's go.",
    "Right.",
    "Sure.",
    "Good morning.",
    "That's fine.",
]

min_words = config.sentence.min_words_to_commit
seg = SentenceSegmenter(min_words_to_commit=min_words)

print(f"config.sentence.min_words_to_commit = {min_words}")
print(f"config.asr.language                 = {config.asr.language!r}")
print()

print(f"{'Japanese turn':<28}{'tokens':>8}  dropped?")
dropped_ja = 0
for t in REAL_TURNS:
    n = cct(t)
    d = seg.is_text_filtered(t)
    dropped_ja += d
    print(f"{t:<28}{n:>8}  {'DROPPED' if d else 'kept'}")
print(f"-> {dropped_ja}/{len(REAL_TURNS)} real Japanese turns dropped")

print()
print(f"{'English turn':<28}{'tokens':>8}  dropped?")
dropped_en = 0
for t in EN_TURNS:
    n = cct(t)
    d = seg.is_text_filtered(t)
    dropped_en += d
    print(f"{t:<28}{n:>8}  {'DROPPED' if d else 'kept'}")
print(f"-> {dropped_en}/{len(EN_TURNS)} real English turns dropped")
