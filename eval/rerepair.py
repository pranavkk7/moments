"""Re-applies the script repair to transcripts already in a library (no new transcription), then
re-indexes them.   python eval/rerepair.py DATA_DIR"""
import json
import sys
from pathlib import Path

from polysub.cues import Word

from moments.library import Library
from moments.scripts import repair_words
from moments.transcribe import Transcript

library = Library(sys.argv[1])
for meta in library.videos():
    folder = library.root / "videos" / meta["id"]
    words = [Word(**w) for w in json.loads((folder / "transcript.json").read_text("utf-8"))]
    words, repaired = repair_words(words, meta["language"])
    if repaired:
        transcript = Transcript(meta["language"], meta["duration"], meta["engine"], words, True)
        library.add(meta["source"], transcript, title=meta["title"])
        print("repaired", meta["id"])
