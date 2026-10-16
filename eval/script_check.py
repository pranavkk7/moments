"""Which Whisper model writes Malayalam in Malayalam script? Transcribes the first minutes of a file with
each model and reports the share of letters per script.   python eval/script_check.py FILE LANG MINUTES"""
import sys
import unicodedata

from faster_whisper import WhisperModel
from polysub.audio import SAMPLE_RATE, load_audio

path, language, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
audio = load_audio(path)[: int(minutes * 60 * SAMPLE_RATE)]
for name in ("large-v3", "large-v2"):
    model = WhisperModel(name, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio, language=language, beam_size=5, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments)
    counts = {}
    for ch in text:
        if ch.isalpha():
            script = unicodedata.name(ch, "?").split(" ")[0]
            counts[script] = counts.get(script, 0) + 1
    total = sum(counts.values()) or 1
    share = ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    print(f"{name}: {share}\n  {text[:400]}\n")
    del model
