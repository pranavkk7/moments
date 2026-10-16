"""Transcribes the same file twice and reports whether the words match exactly.
    python eval/repeatability.py FILE LANG"""
import sys

from moments.transcribe import transcribe

runs = [transcribe(sys.argv[1], language=sys.argv[2]) for _ in range(2)]
texts = [" ".join(w.text for w in r.words) for r in runs]
print(runs[0].engine, "| identical:", texts[0] == texts[1], "| words:", [len(r.words) for r in runs])
