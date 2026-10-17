"""The sentence in a result that best answers the question, shown on the result instead of the start of
a 40-second window or a 150-word passage.

Sentences are compared with the question by the same multilingual embedding model that search uses, so
an English question still picks the right sentence in a Spanish document. For a video in another
language the sentence comes from its English translation, which is what the page shows first.
"""
import re

import numpy as np

# Sentence ends: Latin and Indian full stops, question and exclamation marks, and their full-width forms.
SENTENCE_END = re.compile(r"(?<=[.!?।؟])\s+|(?<=[。！？])")
MIN_CHARS = 12  # shorter pieces ("Yes.", "No.") are joined to the sentence before them
# A full stop after one of these does not end the sentence: "It is used, e.g. by MIT."
ABBREVIATION = re.compile(r"(?:\b(?:e\.g|i\.e|etc|cf|vs|approx|no|nr|mr|mrs|ms|dr|st|fig|p|pp)\.)$", re.IGNORECASE)


def sentences(text):
    """The sentences of a passage, in order. A piece that follows an abbreviation, or is very short,
    is joined to the sentence before it."""
    out = []
    for piece in SENTENCE_END.split(text or ""):
        piece = piece.strip()
        if not piece:
            continue
        if out and (len(piece) < MIN_CHARS or ABBREVIATION.search(out[-1])):
            out[-1] += " " + piece
        else:
            out.append(piece)
    return out


def best_sentences(embedder, question, texts):
    """For each text, the sentence closest in meaning to the question ("" for an empty text). All the
    sentences are embedded in one batch."""
    split = [sentences(t) for t in texts]
    flat = [s for group in split for s in group]
    if not flat:
        return ["" for _ in texts]
    scores = embedder.passages(flat) @ embedder.query(question)
    best, start = [], 0
    for group in split:
        if not group:
            best.append("")
            continue
        best.append(group[int(np.argmax(scores[start:start + len(group)]))])
        start += len(group)
    return best
