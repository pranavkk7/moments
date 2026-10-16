"""Text to vectors that sit close together when they mean the same thing, in any language.

Multilingual embedding models map "When does the course start?" and its Hindi translation to nearby
vectors, which is what lets an English question find a Hindi moment without translating first.
Which model does this best on our videos is measured, not assumed (see moments.evaluate), so the model
is a setting.

Some models expect the text to say whether it is a question or a passage; the prefixes below follow
each model's own instructions.
"""
import os

import numpy as np

DEFAULT_MODEL = os.environ.get("MOMENTS_EMBEDDING_MODEL", "BAAI/bge-m3")

# (query prefix, passage prefix) for models trained with them.
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


class Embedder:
    def __init__(self, model=DEFAULT_MODEL, device=None):
        self.name = model
        self.query_prefix, self.passage_prefix = PREFIXES.get(model, ("", ""))
        self.device = device
        self._model = None  # loaded on first use, so importing this module needs no GPU or download

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def passages(self, texts):
        return self._encode([self.passage_prefix + t for t in texts])

    def query(self, text):
        return self._encode([self.query_prefix + text])[0]

    def _encode(self, texts):
        # Normalised vectors, so the dot product is the cosine similarity.
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=16,
                                            show_progress_bar=False), dtype=np.float32)
