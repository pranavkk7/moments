"""Documents in the library: PDF, Word and plain text, cut into passages that keep their page number.

A passage is about 150 words and never crosses a page, so a search result can say "page 12" and open the
PDF there, the way a video result opens the video at a timestamp. Passages overlap by about 30 words so
a sentence cut at a boundary is whole in one of them. Word files and plain text have no fixed pages, so
their passages are numbered in order instead.

Scanned PDFs (pictures of pages, no text layer) yield no text; Moments does not do OCR and says so.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .scripts import script_of
from .segments import Segment

DOCUMENT_TYPES = {".pdf", ".docx", ".txt", ".md"}
PASSAGE_WORDS = 150
OVERLAP_WORDS = 30
PASSAGE_CHARS = 300  # for languages written without spaces, which are cut by characters

# A few of the commonest words per language, enough to tell Latin-script languages apart in a document.
STOPWORDS = {
    "en": {"the", "and", "of", "to", "is", "in", "that", "it", "for", "with", "as", "are", "this", "was"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "las", "es", "por", "una", "del", "se", "con"},
    "fr": {"le", "la", "de", "et", "les", "des", "est", "une", "du", "en", "que", "pour", "dans", "pas"},
    "de": {"der", "die", "und", "das", "ist", "nicht", "mit", "ein", "eine", "den", "zu", "von", "sie"},
    "pt": {"o", "a", "de", "que", "e", "do", "da", "em", "um", "uma", "os", "para", "com", "não"},
    "it": {"il", "di", "che", "e", "la", "per", "un", "una", "non", "sono", "del", "della", "gli"},
}
SCRIPT_LANGUAGE = {"DEVANAGARI": "hi", "ARABIC": "ar", "HANGUL": "ko", "CYRILLIC": "ru", "BENGALI": "bn",
                   "TAMIL": "ta", "TELUGU": "te", "KANNADA": "kn", "MALAYALAM": "ml", "GUJARATI": "gu",
                   "GURMUKHI": "pa", "THAI": "th", "HEBREW": "he", "GREEK": "el"}


class DocumentError(Exception):
    pass


@dataclass
class Page:
    number: int  # 1-based; None for formats without pages
    text: str


def read_pages(path):
    """The text of a document, page by page."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            return [Page(n, page.extract_text() or "") for n, page in enumerate(reader.pages, start=1)]
        except Exception as error:  # pypdf raises several types for damaged or encrypted files
            raise DocumentError(f"That PDF could not be read: {error}") from error
    if suffix == ".docx":
        from docx import Document

        try:
            paragraphs = [p.text for p in Document(str(path)).paragraphs]
        except Exception as error:
            raise DocumentError(f"That Word file could not be read: {error}") from error
        return [Page(None, "\n".join(paragraphs))]
    if suffix in (".txt", ".md"):
        return [Page(None, path.read_text("utf-8", errors="replace"))]
    raise DocumentError(f"Unsupported document type: {suffix}")


def detect_language(text):
    """A language code for a document. The script decides for most languages; Latin-script text is told
    apart by its commonest words; Japanese is recognised by its kana among Chinese characters."""
    script = script_of(text)
    if script == "CJK":
        has_kana = any(unicodedata.name(ch, "").startswith(("HIRAGANA", "KATAKANA")) for ch in text[:5000])
        return "ja" if has_kana else "zh"
    if script in SCRIPT_LANGUAGE:
        return SCRIPT_LANGUAGE[script]
    words = re.findall(r"[^\W\d_]+", text[:20000].lower())
    votes = {lang: sum(1 for w in words if w in stop) for lang, stop in STOPWORDS.items()}
    return max(votes, key=votes.get) if any(votes.values()) else "en"


def clean(text):
    """Joins words split across lines with a hyphen and collapses the line breaks of PDF text."""
    text = re.sub(r"(\w) ?-\n\s*(\w)", r"\1\2", text)  # "commer-" or "commer -" at a line end
    return re.sub(r"\s+", " ", text).strip()


def passages(pages, language, words=PASSAGE_WORDS, overlap=OVERLAP_WORDS):
    """Overlapping passages that each stay within one page."""
    no_spaces = language in ("ja", "zh")
    out = []
    for page in pages:
        text = clean(page.text)
        if not text:
            continue
        units = list(text) if no_spaces else text.split(" ")
        size = PASSAGE_CHARS if no_spaces else words
        step = size - (overlap * 2 if no_spaces else overlap)
        start = 0
        while start < len(units):
            chunk = units[start:start + size]
            # A last piece much shorter than the overlap is already inside the previous passage.
            if out and start > 0 and len(chunk) <= (overlap * 2 if no_spaces else overlap):
                break
            out.append(Segment(len(out), 0.0, 0.0, ("" if no_spaces else " ").join(chunk), page=page.number))
            start += step
    return out
