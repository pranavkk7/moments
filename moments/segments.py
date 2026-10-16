"""Cutting a transcript into overlapping time windows, the units that search finds and answers cite.

A window is about 40 seconds: long enough to hold one idea with its context, short enough that a
timestamp lands near the point. Windows end at a pause in the speech where one is close, so a sentence
is rarely cut in half, and each window starts about 10 seconds before the previous one ended, so an
idea that straddles a cut is still whole in one of them.
"""
from dataclasses import asdict, dataclass

from polysub.cues import join_words

TARGET_SECONDS = 40.0
MAX_SECONDS = 55.0  # how far a window may stretch to reach a pause
OVERLAP_SECONDS = 10.0
PAUSE_SECONDS = 0.6


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str

    def to_dict(self):
        return asdict(self)


def build_segments(words, language, target=TARGET_SECONDS, longest=MAX_SECONDS,
                   overlap=OVERLAP_SECONDS, pause=PAUSE_SECONDS):
    words = [w for w in words if w.text.strip()]
    segments = []
    first = 0
    while first < len(words):
        start = words[first].start
        last = first
        while last + 1 < len(words) and words[last].end - start < target:
            last += 1
        # Past the target length, end at the first pause before the window gets too long.
        cut = last
        for k in range(last, len(words) - 1):
            if words[k].end - start > longest:
                break
            if words[k + 1].start - words[k].end >= pause:
                cut = k
                break
        segments.append(Segment(len(segments), round(start, 2), round(words[cut].end, 2),
                                join_words(words[first:cut + 1], language)))
        if cut == len(words) - 1:
            break
        # The next window starts `overlap` seconds before this one ended, but always moves forward.
        resume = words[cut].end - overlap
        nxt = next((k for k in range(first + 1, cut + 1) if words[k].start >= resume), cut + 1)
        first = max(nxt, first + 1)
    return segments


def format_time(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
