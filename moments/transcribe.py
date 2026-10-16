"""Speech to timed words, using PolySub's speech engines.

PolySub already chose the more accurate engine per language by measurement (Qwen3-ASR for English and
the other languages its aligner can time, Whisper large-v3 for the rest), so Moments reuses that choice
instead of making its own. Every word keeps its start and end time, which is what lets an answer point
to an exact moment in the video.
"""
from dataclasses import dataclass, field

from polysub import languages
from polysub.asr import EnginePool
from polysub.audio import SAMPLE_RATE, load_audio


@dataclass
class Transcript:
    language: str
    duration: float
    engine: str = ""
    words: list = field(default_factory=list)  # polysub.cues.Word: text, start, end (seconds)


def transcribe(path, pool=None, language=None, progress=None):
    """Timed words for the speech in an audio or video file. `language` is a code such as "ml", or None
    to let Whisper detect it from the first 30 seconds."""
    pool = pool or EnginePool()
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    if language is None:
        language, _ = pool.get("whisper").detect_language(audio)
    else:
        language = languages.resolve(language)
    engine = pool.get(languages.engine_for(language))
    result = engine.transcribe(audio, language, progress=progress)
    return Transcript(language=language, duration=duration, engine=engine.label, words=result.words)
