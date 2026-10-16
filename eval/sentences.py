"""Prints a video's transcript one sentence per line with its start and end time, for writing the
test set's answer ranges.   python eval/sentences.py VIDEO_ID"""
import json
import sys
from pathlib import Path

ENDS = tuple(".?!。？！।؟")

video_id = sys.argv[1]
words = json.loads(Path(f"data/videos/{video_id}/transcript.json").read_text("utf-8"))
no_spaces = json.loads(Path(f"data/videos/{video_id}/meta.json").read_text("utf-8"))["language"] in ("ja", "zh")
sentence = []
for w in words:
    sentence.append(w)
    if w["text"].strip().endswith(ENDS) or w is words[-1]:
        glue = "" if no_spaces else " "
        text = glue.join(x["text"].strip() for x in sentence)
        print(f"{sentence[0]['start']:7.1f} {sentence[-1]['end']:7.1f}  {text}")
        sentence = []

# Transcripts with little punctuation (Hindi here) also get short fixed-size lines: python eval/sentences.py ID 12
if len(sys.argv) > 2:
    size = int(sys.argv[2])
    print("\n--- in lines of", size, "words ---")
    for i in range(0, len(words), size):
        chunk = words[i:i + size]
        print(f"{chunk[0]['start']:7.1f} {chunk[-1]['end']:7.1f}  " + " ".join(w["text"].strip() for w in chunk))
