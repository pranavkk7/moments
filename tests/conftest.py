"""Stand-ins for the heavy parts, so the tests run without a GPU, a model download or an API key."""
import hashlib

import numpy as np
import pytest
from polysub.cues import Word

from moments.library import Library
from moments.transcribe import Transcript

# A tiny "multilingual model": words with the same meaning share a vector, whatever the language.
SAME_MEANING = {"പരീക്ഷ": "exam", "exams": "exam", "ഫീസ്": "fee", "fees": "fee", "試験": "exam"}


class FakeEmbedder:
    name = "fake-embedder"

    def _vector(self, text):
        v = np.zeros(64, dtype=np.float32)
        # Split on spaces, not \w: Malayalam vowel signs are not "word" characters to a regex.
        for word in text.lower().split():
            word = word.strip(".,?!")
            word = SAME_MEANING.get(word, word)
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def passages(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def query(self, text):
        return self._vector(text)


def words_from(sentences, gap=1.0, word_seconds=0.5):
    """Timed words: each sentence is spoken word by word, with a pause between sentences."""
    words, t = [], 0.0
    for sentence in sentences:
        for token in sentence.split():
            words.append(Word(token, round(t, 2), round(t + word_seconds - 0.05, 2)))
            t += word_seconds
        t += gap
    return words


def transcript(sentences, language="en"):
    words = words_from(sentences)
    duration = (words[-1].end if words else 0) + 1
    return Transcript(language=language, duration=duration, engine="fake", words=words)


@pytest.fixture
def library(tmp_path):
    return Library(tmp_path / "data", embedder=FakeEmbedder())


@pytest.fixture
def media(tmp_path):
    """Files to add: search only needs them to exist and differ."""
    def make(name, content=b"x"):
        path = tmp_path / name
        path.write_bytes(content)
        return path
    return make
