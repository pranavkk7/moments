"""English translations of documents, made on your own machine.

Whisper translates speech, so videos get their English while they are transcribed; documents need a
text translation model instead. Moments uses the University of Helsinki's Opus-MT models (MarianMT,
about 300 MB each, Apache-2.0), one per language into English. Passages are translated sentence by
sentence, which is what these models are trained on, and joined again.
"""
from .highlight import sentences

MODELS = {lang: f"Helsinki-NLP/opus-mt-{lang}-en" for lang in ("es", "hi", "ar", "ja")}
BATCH = 16
MAX_TOKENS = 400  # a sentence longer than this is cut; rare in prose


def can_translate(language):
    return language in MODELS


def model_folder(language):
    """The model's files on disk, downloaded once if missing. Loading from the folder, rather than by the
    model's name, keeps transformers from contacting Hugging Face each time and from fetching a second,
    converted copy of the weights in the background."""
    from huggingface_hub import snapshot_download

    files = ["pytorch_model.bin", "*.json", "*.spm"]
    try:
        return snapshot_download(MODELS[language], allow_patterns=files, local_files_only=True)
    except Exception:  # not downloaded yet
        return snapshot_download(MODELS[language], allow_patterns=files)


class TextTranslator:
    """Keeps one Opus-MT model loaded at a time (they are loaded on first use)."""

    def __init__(self, device=None):
        self.device = device
        self.language = None
        self.model = None
        self.tokenizer = None

    def _load(self, language):
        if self.language == language:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        folder = model_folder(language)
        self.tokenizer = MarianTokenizer.from_pretrained(folder)
        self.model = MarianMTModel.from_pretrained(folder).to(device).eval()
        self.language, self.device = language, device

    def translate(self, texts, language, progress=None):
        """English for each text, keeping their order."""
        import torch

        self._load(language)
        split = [sentences(t) for t in texts]
        flat = [s for group in split for s in group]
        english = []
        for start in range(0, len(flat), BATCH):
            batch = self.tokenizer(flat[start:start + BATCH], return_tensors="pt", padding=True, truncation=True,
                                   max_length=MAX_TOKENS).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**batch, max_new_tokens=MAX_TOKENS)
            english += self.tokenizer.batch_decode(out, skip_special_tokens=True)
            if progress:
                progress(min((start + BATCH) / len(flat), 1.0))
        joined, at = [], 0
        for group in split:
            joined.append(" ".join(english[at:at + len(group)]))
            at += len(group)
        return joined
