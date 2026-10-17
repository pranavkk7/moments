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


def remove_running_lines(pages, edge=2):
    """Drops headers and footers: lines among the first or last `edge` of a page that recur, ignoring
    digits and punctuation, on several pages ("3. The Creative Commons licencing scheme 31" on every page
    of a chapter). Left in, they would start passages with noise and add the same words to all of them."""
    def key(line):
        return re.sub(r"[\W\d_]+", "", line.lower())

    numbered = [p for p in pages if p.number]
    if len(numbered) < 3:
        return pages
    edges = []
    for page in numbered:
        lines = [l for l in page.text.split("\n") if l.strip()]
        edges.append({key(l) for l in lines[:edge] + lines[-edge:] if key(l)})
    counts = {}
    for found in edges:
        for k in found:
            counts[k] = counts.get(k, 0) + 1
    running = {k for k, n in counts.items() if n >= max(3, len(numbered) // 20)}
    if not running:
        return pages
    out = []
    for page in pages:
        lines = page.text.split("\n")
        kept_idx = [i for i, l in enumerate(lines) if l.strip()]
        edge_idx = set(kept_idx[:edge] + kept_idx[-edge:])
        text = "\n".join(l for i, l in enumerate(lines) if not (i in edge_idx and key(l) in running))
        out.append(Page(page.number, text))
    return out


MIXED_CASE = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b|\b[A-Z]{2,}[a-z][A-Za-z]*\b")


def fix_small_caps(text, vocabulary):
    """Some PDF fonts set headings in small capitals, which text extraction returns as "CreATive" or
    "COmmOns". A mixed-case word is put back to normal case when the document also uses it in plain
    lower case elsewhere ("licencing"); names like "OpenCourseWare" and plurals like "PDFs" never appear
    that way, so they are left alone."""
    def fix(match):
        word = match.group(0)
        if word.lower() not in vocabulary:
            return word
        return word[0] + word[1:].lower() if word[0].isupper() else word.lower()

    return MIXED_CASE.sub(fix, text)


def lowercase_words(pages):
    return {w for p in pages for w in re.findall(r"\b[a-z]{3,}\b", p.text)}


def clean(text):
    """Joins words split across lines with a hyphen and collapses the line breaks of PDF text."""
    text = re.sub(r"(\w) ?-\n\s*(\w)", r"\1\2", text)  # "commer-" or "commer -" at a line end
    return re.sub(r"\s+", " ", text).strip()


def passages(pages, language, words=PASSAGE_WORDS, overlap=OVERLAP_WORDS):
    """Overlapping passages that each stay within one page, without the pages' headers and footers."""
    no_spaces = language in ("ja", "zh")
    pages = remove_running_lines(pages)
    vocabulary = lowercase_words(pages)
    out = []
    for page in pages:
        text = fix_small_caps(clean(page.text), vocabulary)
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
