"""The library of videos and documents: their windows and passages on disk, vectors in ChromaDB, and
hybrid search over all of them at once.

Search combines two rankings, because each misses what the other finds:
- meaning (dense vectors): finds a Hindi moment for an English question, and paraphrases;
- keywords (BM25 on the transcript): finds exact names, numbers and rare terms that vectors blur.
The two lists are merged with reciprocal rank fusion, which needs no tuning of score scales: a window
ranked high by either method rises, and one ranked high by both rises most. Keyword matches only count
when the question is in the same language as the text they matched; across languages, meaning decides.

Layout under the data folder:
    videos/<id>/meta.json        title, source file, kind ("video" or "document"), language, ...
    videos/<id>/transcript.json  every timed word
    videos/<id>/segments.json    the time windows that search returns
    videos/<id>/vectors.npz      the windows' vectors, Moments' own copy (see _meaning_ranking)
    index/                       ChromaDB, one collection per embedding model
"""
import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polysub import languages
from polysub.audio import AudioError

from .documents import DOCUMENT_TYPES, DocumentError, detect_language, passages, read_pages
from .embed import Embedder
from .highlight import best_sentences
from .scripts import script_of
from .segments import Segment, attach_english, build_segments
from .transcribe import transcribe, translate
from .translate_text import TextTranslator, can_translate

RRF_K = 60  # the usual constant: damps the gap between rank 1 and rank 2
CANDIDATES = 30  # how many windows each method contributes before fusion
# How much keyword matches count when the question is in a different language from the text they matched.
# Matching the words of an English question against a Japanese or Spanish transcript is mostly noise, and
# on the test set it pulled the right moment down (see the README); only meaning connects two languages.
CROSS_LANGUAGE_KEYWORD_WEIGHT = 0.0
# The rankings search can fuse: meaning and keywords in the video's own language, and the same two over
# the English translation of videos in other languages.
RANKINGS = ("meaning", "keyword", "meaning_en", "keyword_en")
MODES = {"hybrid": RANKINGS, "meaning": ("meaning",), "keyword": ("keyword",)}


class LibraryError(Exception):
    pass


@dataclass
class Hit:
    video_id: str
    segment: Segment
    score: float
    meaning_rank: int = None  # 1-based rank in the dense list, None if absent
    keyword_rank: int = None
    language: str = ""
    ranks: dict = None  # 1-based rank in every ranking that found this window
    best: str = ""  # the sentence that best answers the question (see Library.highlight)


def video_id_for(path):
    """A readable, stable id: the file name plus a short hash of its size and first megabyte, so the same
    file added twice is recognised and two files with the same name are not confused."""
    path = Path(path)
    digest = hashlib.sha1(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(1 << 20))
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40] or "video"
    return f"{slug}-{digest.hexdigest()[:8]}"


def fold_accents(text):
    """"púlsar" -> "pulsar", for Latin-script text only: in Indian scripts the same Unicode marks are
    vowel signs, which carry meaning."""
    decomposed = unicodedata.normalize("NFKD", text)
    out, base_latin = [], False
    for ch in decomposed:
        if unicodedata.combining(ch):
            if not base_latin:
                out.append(ch)
            continue
        base_latin = unicodedata.name(ch, "").startswith("LATIN")
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def tokens(text, language):
    """Terms for keyword search, lower-cased and without accents on Latin letters. Languages written
    without spaces are split into character pairs."""
    text = fold_accents(text.lower())
    if not languages.uses_spaces(language):
        chars = [c for c in text if not c.isspace() and c.isalnum()]
        return [a + b for a, b in zip(chars, chars[1:])] or chars
    return [t for t in re.split(r"[\s.,!?;:\"'()\[\]{}…\-–—/]+", text) if t]


class Library:
    def __init__(self, root="data", embedder=None):
        self.root = Path(root)
        self.embedder = embedder or Embedder()
        self._collection = None
        self._english = None
        self._text_translator = None  # loaded only when a document needs translating
        self._scripts = {}  # video id -> writing system of its transcript
        self._vocabulary = {}  # video id -> (terms of the transcript, terms of its English translation)

    # ---------- storage ----------

    def _dir(self, video_id):
        return self.root / "videos" / video_id

    def _open(self, prefix):
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(self.root / "index"), settings=Settings(anonymized_telemetry=False))
        name = prefix + re.sub(r"[^a-zA-Z0-9._-]+", "-", self.embedder.name)[:50]
        # We always pass our own vectors, so Chroma must not load an embedding model of its own.
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    @property
    def collection(self):
        """Vectors of the windows in the video's own language."""
        if self._collection is None:
            self._collection = self._open("segments-")
        return self._collection

    @property
    def english(self):
        """Vectors of the windows' English translations (videos in other languages only)."""
        if self._english is None:
            self._english = self._open("english-")
        return self._english

    def items(self):
        """Every video and document in the library, by title."""
        folder = self.root / "videos"
        if not folder.exists():
            return []
        metas = [json.loads((d / "meta.json").read_text("utf-8")) for d in folder.iterdir()
                 if (d / "meta.json").exists()]
        return sorted(metas, key=lambda m: m["title"].lower())

    def meta(self, video_id):
        path = self._dir(video_id) / "meta.json"
        if not path.exists():
            raise LibraryError(f"Nothing with id {video_id!r} in the library.")
        return json.loads(path.read_text("utf-8"))

    def update_meta(self, video_id, **fields):
        """Adds or replaces fields in a video's meta.json (for example where a sample can be streamed from)."""
        meta = {**self.meta(video_id), **fields}
        (self._dir(video_id) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        return meta

    def script(self, video_id):
        """The writing system of a video's transcript, worked out once and remembered."""
        if video_id not in self._scripts:
            self._scripts[video_id] = script_of(" ".join(s.text for s in self.segments(video_id)[:5]))
        return self._scripts[video_id]

    def vocabulary(self, video_id, language):
        """The keyword terms of a video's transcript and of its English translation, worked out once."""
        if video_id not in self._vocabulary:
            segments = self.segments(video_id)
            self._vocabulary[video_id] = (set(tokens(" ".join(s.text for s in segments), language)),
                                          set(tokens(" ".join(s.english for s in segments), "en")))
        return self._vocabulary[video_id]

    def question_language(self, query, meta):
        """Whether a question is in the video's own language ("own") or in English ("en"), judged by
        which of the two texts more of its words appear in. Spanish and English share an alphabet, so the
        script alone cannot tell an English question from a Spanish one."""
        if script_of(query) != self.script(meta["id"]):
            return "en" if script_of(query) == "LATIN" else "other"
        own, english = self.vocabulary(meta["id"], meta["language"])
        words = set(tokens(query, meta["language"]))
        if not english:
            # No translation (an English video, or a document): the question is in the text's language
            # when most of its words appear in the text.
            return "own" if words and len(words & own) >= len(words) / 2 else "other"
        return "own" if len(words & own) > len(words & english) else "en"

    def segments(self, video_id):
        data = json.loads((self._dir(video_id) / "segments.json").read_text("utf-8"))
        return [Segment(**s) for s in data]

    # ---------- adding videos ----------

    def ingest(self, path, pool=None, language=None, title=None, progress=None, engine_name=None,
               on_stage=None, english=True):
        """Transcribes a file and adds it, then translates it into English when it is in another
        language. Returns the video id. `engine_name` overrides the engine chosen for the language
        (used to compare engines); `on_stage` hears "transcribing" and "translating"."""
        path = Path(path)
        if not path.exists():
            raise LibraryError(f"File not found: {path}")
        if path.suffix.lower() in DOCUMENT_TYPES:
            if on_stage:
                on_stage("reading")
            doc_id = self.add_document(path, title=title)
            if english and can_translate(self.meta(doc_id)["language"]):
                if on_stage:
                    on_stage("translating")
                self.translate(doc_id, progress=progress)
            return doc_id
        if on_stage:
            on_stage("transcribing")
        try:
            transcript = transcribe(path, pool=pool, language=language, progress=progress, engine_name=engine_name)
        except (ValueError, AudioError) as error:  # an unknown language, or a file FFmpeg cannot read
            raise LibraryError(str(error)) from error
        video_id = self.add(path, transcript, title=title)
        if english and transcript.language != "en":
            if on_stage:
                on_stage("translating")
            self.translate(video_id, pool=pool, progress=progress)
        return video_id

    def remove(self, video_id):
        """Takes a video out of the library: its vectors and its folder. The original file is untouched."""
        self.meta(video_id)  # raises if there is no such video
        self.collection.delete(where={"video_id": video_id})
        self.english.delete(where={"video_id": video_id})
        shutil.rmtree(self._dir(video_id))
        self._scripts.pop(video_id, None)

    def translate(self, video_id, pool=None, progress=None):
        """Adds an English translation to every window or passage of an item in another language, and
        indexes it. Shown above the original so a viewer can read it, and searched as well. Videos are
        translated from their speech by Whisper; documents by a text translation model, for the
        languages it covers."""
        meta = self.meta(video_id)
        if meta["language"] == "en":
            return False
        if meta.get("kind") == "document":
            if not can_translate(meta["language"]):
                return False
            if self._text_translator is None:
                self._text_translator = TextTranslator()
            texts = self._text_translator.translate([s.text for s in self.segments(video_id)], meta["language"],
                                                    progress=progress)
            self.set_english_texts(video_id, texts)
            return True
        words = translate(meta["source"], meta["language"], pool=pool, progress=progress)
        self.set_english(video_id, words)
        return True

    def set_english(self, video_id, words):
        """A video's English, from translated words with their timings."""
        segments = attach_english(self.segments(video_id), words)
        (self._dir(video_id) / "transcript_en.json").write_text(
            json.dumps([{"text": w.text, "start": w.start, "end": w.end} for w in words], ensure_ascii=False), "utf-8")
        self._store_english(video_id, segments)

    def set_english_texts(self, item_id, texts):
        """A document's English: one translated text per passage, in order."""
        segments = self.segments(item_id)
        for segment, text in zip(segments, texts):
            segment.english = text
        self._store_english(item_id, segments)

    def _store_english(self, video_id, segments):
        folder = self._dir(video_id)
        (folder / "segments.json").write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=1), "utf-8")
        self.update_meta(video_id, english=True)
        self._vocabulary.pop(video_id, None)
        self.english.delete(where={"video_id": video_id})
        translated = [s for s in segments if s.english]
        if translated:
            vectors = self.embedder.passages([s.english for s in translated])
            saved = dict(np.load(folder / "vectors.npz")) if (folder / "vectors.npz").exists() else {}
            np.savez(folder / "vectors.npz", **saved, english=vectors, english_index=[s.index for s in translated])
            self.english.add(
                ids=[f"{video_id}:{s.index}" for s in translated],
                embeddings=vectors.tolist(),
                documents=[s.english for s in translated],
                metadatas=[{"video_id": video_id, "index": s.index, "start": s.start, "end": s.end}
                           for s in translated],
            )

    def add(self, path, transcript, title=None):
        path = Path(path)
        segments = build_segments(transcript.words, transcript.language)
        if not segments:
            raise LibraryError("No speech was found in this file.")
        video_id = video_id_for(path)
        folder = self._dir(video_id)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"id": video_id, "title": title or path.stem, "source": str(path.resolve()), "kind": "video",
                "language": transcript.language, "engine": transcript.engine,
                "script_repaired": transcript.script_repaired,
                "duration": round(transcript.duration, 2), "segments": len(segments)}
        words = [{"text": w.text, "start": w.start, "end": w.end} for w in transcript.words]
        (folder / "transcript.json").write_text(json.dumps(words, ensure_ascii=False), "utf-8")
        (folder / "segments.json").write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=1), "utf-8")
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")

        self._index(video_id, segments, transcript.language)
        return video_id

    def add_document(self, path, title=None):
        """Reads a PDF, Word or text file and adds its passages. Returns its id."""
        path = Path(path)
        try:
            pages = read_pages(path)
        except DocumentError as error:
            raise LibraryError(str(error)) from error
        text = " ".join(p.text for p in pages)
        if not text.strip():
            raise LibraryError("No text was found in this document. Scanned PDFs need OCR, which Moments doesn't do.")
        language = detect_language(text)
        segments = passages(pages, language)
        doc_id = video_id_for(path)
        folder = self._dir(doc_id)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"id": doc_id, "title": title or path.stem, "source": str(path.resolve()), "kind": "document",
                "format": path.suffix.lower().lstrip("."), "language": language,
                "pages": max((p.number or 0) for p in pages) or None, "duration": 0, "segments": len(segments)}
        (folder / "segments.json").write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=1), "utf-8")
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        self._index(doc_id, segments, language)
        return doc_id

    def _index(self, item_id, segments, language):
        """Vectors for an item's windows or passages: into ChromaDB, and Moments' own copy beside them.
        Re-adding an item replaces its vectors rather than duplicating them; a new transcript also makes
        any earlier translation stale."""
        self._scripts.pop(item_id, None)
        self._vocabulary.pop(item_id, None)
        self.collection.delete(where={"video_id": item_id})
        self.english.delete(where={"video_id": item_id})
        vectors = self.embedder.passages([s.text for s in segments])
        np.savez(self._dir(item_id) / "vectors.npz", original=vectors, original_index=[s.index for s in segments])
        self.collection.add(
            ids=[f"{item_id}:{s.index}" for s in segments],
            embeddings=vectors.tolist(),
            documents=[s.text for s in segments],
            metadatas=[{"video_id": item_id, "index": s.index, "start": s.start, "end": s.end,
                        "page": s.page or 0, "language": language} for s in segments],
        )

    # ---------- search ----------

    def search(self, query, video_id=None, k=5, mode="hybrid", rankings=None, average=True):
        """The k best windows for a question. `mode` is "hybrid" (the default), or "meaning" / "keyword" to
        use one ranking alone; `rankings` picks any set of RANKINGS, which is how the evaluation measures
        what each contributes."""
        if mode not in MODES:
            raise ValueError(f"Unknown search mode: {mode}")
        rankings = tuple(rankings or MODES[mode])
        query = (query or "").strip()
        if not query:
            raise LibraryError("Type a question first.")
        targets = [self.meta(video_id)] if video_id else self.items()
        if not targets:
            raise LibraryError("The library is empty. Add a video or a document first.")
        windows = {(m["id"], s.index): s for m in targets for s in self.segments(m["id"])}

        only_one = len(rankings) == 1
        asked_in = {m["id"]: self.question_language(query, m) for m in targets}
        lists, weights = {}, {}
        if "meaning" in rankings:
            lists["meaning"] = self._meaning_ranking(self.collection, query, video_id, len(windows), targets, "original")
            weights["meaning"] = lambda vid: 1.0
        if "keyword" in rankings:
            lists["keyword"] = self._keyword_ranking(query, targets, windows)
            # Keyword matches count only for a question in the video's own language.
            weights["keyword"] = lambda vid: 1.0 if only_one or asked_in[vid] == "own" else CROSS_LANGUAGE_KEYWORD_WEIGHT
        translated = [m for m in targets if m.get("english")]
        if translated and "meaning_en" in rankings:
            lists["meaning_en"] = self._meaning_ranking(self.english, query, video_id, len(windows), targets, "english")
            weights["meaning_en"] = lambda vid: 1.0
        if translated and "keyword_en" in rankings:
            english = {key: Segment(s.index, s.start, s.end, s.english) for key, s in windows.items()
                       if s.english}
            lists["keyword_en"] = self._keyword_ranking(query, [{**m, "language": "en"} for m in translated], english)
            # ...and matches in the English translation only for a question in English.
            weights["keyword_en"] = lambda vid: 1.0 if only_one or asked_in[vid] == "en" else CROSS_LANGUAGE_KEYWORD_WEIGHT

        scores = {}
        for name, ranking in lists.items():
            for rank, key in enumerate(ranking, start=1):
                weight = weights[name](key[0])
                if weight and key in windows:
                    scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)
        # Each score is averaged over the rankings its item can appear in. A translated video sits in four
        # rankings and a document in two, so plain sums favoured translated videos whenever the whole
        # library was searched (measured in the README). Within one item every window has the same
        # rankings, so this changes nothing there.
        open_to = {m["id"]: sum(1 for name in lists if weights[name](m["id"])
                                and (not name.endswith("_en") or m.get("english"))) for m in targets}
        if average:  # `average=False` restores plain sums, for the evaluation to compare
            scores = {key: score / max(1, open_to[key[0]]) for key, score in scores.items()}
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        language_of = {m["id"]: m["language"] for m in targets}

        def rank_in(name, key):
            ranking = lists.get(name, [])
            return ranking.index(key) + 1 if key in ranking else None

        return [Hit(key[0], windows[key], round(scores[key], 5), rank_in("meaning", key), rank_in("keyword", key),
                    language_of[key[0]], {name: rank_in(name, key) for name in lists if rank_in(name, key)})
                for key in best]

    def highlight(self, query, hits):
        """Fills in each hit's best sentence: from the English translation when there is one, since
        that is what the page shows first, otherwise from the original words."""
        for hit, best in zip(hits, best_sentences(self.embedder, query,
                                                  [h.segment.english or h.segment.text for h in hits])):
            hit.best = best
        return hits

    def _meaning_ranking(self, collection, query, video_id, available, targets, kind):
        vector = self.embedder.query(query)
        try:
            if collection.count() == 0:
                return []
            result = collection.query(query_embeddings=[vector.tolist()], n_results=max(1, min(CANDIDATES, available)),
                                      where={"video_id": video_id} if video_id else None, include=["metadatas"])
            return [(m["video_id"], m["index"]) for m in result["metadatas"][0]]
        except Exception as error:  # chromadb.errors.InternalError
            # Right after a video's vectors are replaced, ChromaDB's index is sometimes not readable yet
            # ("Error creating hnsw segment reader: Nothing found on disk"), seen about once in 25 test
            # runs. Moments keeps its own copy of every video's vectors, so it ranks those exactly instead:
            # the same answer, and quick at the size of a personal video library.
            if "hnsw" not in str(error).lower():
                raise
            return self._exact_ranking(vector, targets, kind)

    def _exact_ranking(self, vector, targets, kind):
        keys, rows = [], []
        for meta in targets:
            path = self._dir(meta["id"]) / "vectors.npz"
            if not path.exists():
                continue
            saved = np.load(path)
            if kind not in saved:
                continue
            keys += [(meta["id"], int(i)) for i in saved[f"{kind}_index"]]
            rows.append(saved[kind])
        if not rows:
            return []
        scores = np.concatenate(rows) @ vector
        return [keys[i] for i in np.argsort(-scores)[:CANDIDATES]]

    def _keyword_ranking(self, query, targets, windows):
        from rank_bm25 import BM25Okapi

        keys, corpus = [], []
        language_of = {m["id"]: m["language"] for m in targets}
        for (vid, index), segment in windows.items():
            keys.append((vid, index))
            corpus.append(tokens(segment.text, language_of[vid]) or ["_"])
        # The question is tokenised the way each video's language is, so a Japanese question can match
        # a Japanese transcript (character pairs) as well as a spaced one (words).
        query_tokens = set().union(*(tokens(query, lang) for lang in set(language_of.values())))
        scores = BM25Okapi(corpus).get_scores(list(query_tokens))
        ranked = sorted(range(len(keys)), key=lambda i: scores[i], reverse=True)
        return [keys[i] for i in ranked[:CANDIDATES] if scores[i] > 0]
