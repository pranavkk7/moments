"""Speech to timed words, using PolySub's speech engines.

PolySub already chose the more accurate engine per language by measurement (Qwen3-ASR for English and
the other languages its aligner can time, Whisper large-v3 for the rest), so Moments reuses that choice
instead of making its own. Every word keeps its start and end time, which is what lets an answer point
to an exact moment in the video.
"""
from dataclasses import dataclass, field

from polysub import languages
from polysub.asr import ENGINES, EnginePool, WhisperEngine
from polysub.audio import SAMPLE_RATE, load_audio

from .scripts import repair_words


SEED = 0


class SeededWhisperEngine(WhisperEngine):
    """PolySub's Whisper engine, made repeatable. When Whisper gets stuck repeating itself it retries with
    random sampling, so two runs on the same file differed: on the Malayalam sample one run dropped a
    whole sentence. Fixing the random seed keeps those retries but makes every run the same."""

    def transcribe(self, audio, language, **kwargs):
        import ctranslate2

        ctranslate2.set_random_seed(SEED)
        return super().transcribe(audio, language, **kwargs)


class WhisperV2Engine(SeededWhisperEngine):
    """Whisper large-v2, kept as an option per language: on Malayalam it writes Kannada letters, which
    map onto Malayalam far more completely than large-v3's Gurmukhi (see moments.scripts)."""

    label = "Whisper large-v2"

    def __init__(self, device="cuda"):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel("large-v2", device=device.split(":")[0], compute_type=compute_type)


FACTORIES = {**ENGINES, "whisper": SeededWhisperEngine, "whisper-v2": WhisperV2Engine}
# Engine per language where our own measurements beat PolySub's default choice (see the README). On the
# Malayalam sample, large-v2 + script repair found the right moment first for 70% of English questions,
# large-v3 for 30%.
ENGINE_FOR = {"ml": "whisper-v2"}


def engine_for(language):
    return ENGINE_FOR.get(language) or languages.engine_for(language)


@dataclass
class Transcript:
    language: str
    duration: float
    engine: str = ""
    words: list = field(default_factory=list)  # polysub.cues.Word: text, start, end (seconds)
    script_repaired: bool = False  # letters were moved into the language's own script


def translate(path, language, pool=None, progress=None):
    """English words with their timings, translated from speech by Whisper (task "translate"), on the
    GPU and without any API. Uses the language's own Whisper version when it has one (large-v2 for
    Malayalam)."""
    own_pool = pool is None
    pool = pool or EnginePool(factories=FACTORIES)
    try:
        audio = load_audio(path)
        name = "whisper-v2" if ENGINE_FOR.get(language) == "whisper-v2" else "whisper"
        return pool.get(name).transcribe(audio, language, progress=progress, task="translate").words
    finally:
        if own_pool:
            pool.close()


def transcribe(path, pool=None, language=None, progress=None, engine_name=None):
    """Timed words for the speech in an audio or video file. `language` is a code such as "ml", or None
    to let Whisper detect it from the first 30 seconds."""
    language = languages.resolve(language)  # "Malayalam" -> "ml"; None or "auto" -> detect
    own_pool = pool is None
    pool = pool or EnginePool(factories=FACTORIES)
    try:
        audio = load_audio(path)
        duration = len(audio) / SAMPLE_RATE
        if language is None:
            language, _ = pool.get("whisper").detect_language(audio)
        engine = pool.get(engine_name or engine_for(language))
        result = engine.transcribe(audio, language, progress=progress)
        words, repaired = repair_words(result.words, language)
        return Transcript(language=language, duration=duration, engine=engine.label, words=words,
                          script_repaired=repaired)
    finally:
        # A pool made here is freed at once: the embedding model needs the GPU next, and 8 GB cannot
        # hold both. A pool passed in (the web app's) stays loaded for the next file.
        if own_pool:
            pool.close()
