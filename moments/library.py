"""The video library: transcripts and windows on disk, vectors in ChromaDB, and hybrid search over both.

Search combines two rankings, because each misses what the other finds:
- meaning (dense vectors): finds a Malayalam moment for an English question, and paraphrases;
- keywords (BM25 on the transcript): finds exact names, numbers and rare terms that vectors blur.
The two lists are merged with reciprocal rank fusion, which needs no tuning of score scales: a window
ranked high by either method rises, and one ranked high by both rises most. Keyword matches only count
when the question is written in the same script as the video; across languages, meaning alone decides.

Layout under the data folder:
    videos/<id>/meta.json        title, source file, language, engine, duration
    videos/<id>/transcript.json  every timed word
    videos/<id>/segments.json    the time windows that search returns
    index/                       ChromaDB, one collection per embedding model
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from polysub import languages
from polysub.audio import AudioError

from .embed import Embedder
from .segments import Segment, build_segments
from .transcribe import transcribe

RRF_K = 60  # the usual constant: damps the gap between rank 1 and rank 2
CANDIDATES = 30  # how many windows each method contributes before fusion
# How much keyword matches count when the question and the video are in different scripts. Matching the
# words of an English question against a Japanese transcript is mostly noise, and on the test set it
# pulled the right moment down (see the README); only meaning connects the two languages.
CROSS_SCRIPT_KEYWORD_WEIGHT = 0.0


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


def video_id_for(path):
    """A readable, stable id: the file name plus a short hash of its size and first megabyte, so the same
    file added twice is recognised and two files with the same name are not confused."""
    path = Path(path)
    digest = hashlib.sha1(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(1 << 20))
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40] or "video"
    return f"{slug}-{digest.hexdigest()[:8]}"


def script_of(text):
    """The writing system most of the letters in `text` belong to: "LATIN", "CJK", "DEVANAGARI",
    "MALAYALAM", "ARABIC", ... Japanese kana and Chinese characters count as one script, "CJK"."""
    counts = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "").split(" ")[0]
            name = "CJK" if name in ("CJK", "HIRAGANA", "KATAKANA") else name
            counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def tokens(text, language):
    """Words for keyword search. Languages written without spaces are split into character pairs."""
    text = text.lower()
    if not languages.uses_spaces(language):
        chars = [c for c in text if not c.isspace() and c.isalnum()]
        return [a + b for a, b in zip(chars, chars[1:])] or chars
    return [t for t in re.split(r"[\s.,!?;:\"'()\[\]{}…\-–—/]+", text) if t]


class Library:
    def __init__(self, root="data", embedder=None):
        self.root = Path(root)
        self.embedder = embedder or Embedder()
        self._collection = None
        self._scripts = {}  # video id -> writing system of its transcript

    # ---------- storage ----------

    def _dir(self, video_id):
        return self.root / "videos" / video_id

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(path=str(self.root / "index"),
                                               settings=Settings(anonymized_telemetry=False))
            name = "segments-" + re.sub(r"[^a-zA-Z0-9._-]+", "-", self.embedder.name)[:50]
            # We always pass our own vectors, so Chroma must not load an embedding model of its own.
            self._collection = client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
        return self._collection

    def videos(self):
        folder = self.root / "videos"
        if not folder.exists():
            return []
        metas = [json.loads((d / "meta.json").read_text("utf-8")) for d in folder.iterdir()
                 if (d / "meta.json").exists()]
        return sorted(metas, key=lambda m: m["title"].lower())

    def meta(self, video_id):
        path = self._dir(video_id) / "meta.json"
        if not path.exists():
            raise LibraryError(f"No video with id {video_id!r} in the library.")
        return json.loads(path.read_text("utf-8"))

    def script(self, video_id):
        """The writing system of a video's transcript, worked out once and remembered."""
        if video_id not in self._scripts:
            self._scripts[video_id] = script_of(" ".join(s.text for s in self.segments(video_id)[:5]))
        return self._scripts[video_id]

    def segments(self, video_id):
        data = json.loads((self._dir(video_id) / "segments.json").read_text("utf-8"))
        return [Segment(**s) for s in data]

    # ---------- adding videos ----------

    def ingest(self, path, pool=None, language=None, title=None, progress=None):
        """Transcribes a file and adds it. Returns the video id."""
        path = Path(path)
        if not path.exists():
            raise LibraryError(f"File not found: {path}")
        try:
            transcript = transcribe(path, pool=pool, language=language, progress=progress)
        except (ValueError, AudioError) as error:  # an unknown language, or a file FFmpeg cannot read
            raise LibraryError(str(error)) from error
        return self.add(path, transcript, title=title)

    def add(self, path, transcript, title=None):
        path = Path(path)
        segments = build_segments(transcript.words, transcript.language)
        if not segments:
            raise LibraryError("No speech was found in this file.")
        video_id = video_id_for(path)
        folder = self._dir(video_id)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"id": video_id, "title": title or path.stem, "source": str(path.resolve()),
                "language": transcript.language, "engine": transcript.engine,
                "duration": round(transcript.duration, 2), "segments": len(segments)}
        words = [{"text": w.text, "start": w.start, "end": w.end} for w in transcript.words]
        (folder / "transcript.json").write_text(json.dumps(words, ensure_ascii=False), "utf-8")
        (folder / "segments.json").write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=1), "utf-8")
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")

        self._scripts.pop(video_id, None)
        # Re-adding a video replaces its windows rather than duplicating them.
        self.collection.delete(where={"video_id": video_id})
        self.collection.add(
            ids=[f"{video_id}:{s.index}" for s in segments],
            embeddings=self.embedder.passages([s.text for s in segments]).tolist(),
            documents=[s.text for s in segments],
            metadatas=[{"video_id": video_id, "index": s.index, "start": s.start, "end": s.end,
                        "language": transcript.language} for s in segments],
        )
        return video_id

    # ---------- search ----------

    def search(self, query, video_id=None, k=5, mode="hybrid"):
        """The k best windows for a question. `mode` is "hybrid" (the default), or "meaning" / "keyword" to
        use one ranking alone, which is how the evaluation measures what each contributes."""
        if mode not in ("hybrid", "meaning", "keyword"):
            raise ValueError(f"Unknown search mode: {mode}")
        query = (query or "").strip()
        if not query:
            raise LibraryError("Type a question first.")
        targets = [self.meta(video_id)] if video_id else self.videos()
        if not targets:
            raise LibraryError("The library is empty. Add a video first.")
        windows = {(m["id"], s.index): s for m in targets for s in self.segments(m["id"])}

        meaning = self._meaning_ranking(query, video_id, len(windows)) if mode != "keyword" else []
        keyword = self._keyword_ranking(query, targets, windows) if mode != "meaning" else []

        question_script = script_of(query)
        keyword_weight = {m["id"]: 1.0 if mode == "keyword" or self.script(m["id"]) == question_script
                          else CROSS_SCRIPT_KEYWORD_WEIGHT for m in targets}
        scores = {}
        for rank, key in enumerate(meaning, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        for rank, key in enumerate(keyword, start=1):
            if keyword_weight[key[0]]:
                scores[key] = scores.get(key, 0.0) + keyword_weight[key[0]] / (RRF_K + rank)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        language_of = {m["id"]: m["language"] for m in targets}
        return [Hit(key[0], windows[key], round(scores[key], 5),
                    meaning.index(key) + 1 if key in meaning else None,
                    keyword.index(key) + 1 if key in keyword else None,
                    language_of[key[0]])
                for key in best if key in windows]

    def _meaning_ranking(self, query, video_id, available):
        result = self.collection.query(
            query_embeddings=[self.embedder.query(query).tolist()],
            n_results=max(1, min(CANDIDATES, available)),
            where={"video_id": video_id} if video_id else None,
            include=["metadatas"],
        )
        return [(m["video_id"], m["index"]) for m in result["metadatas"][0]]

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
