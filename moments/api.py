"""The web API and page: upload videos and documents, follow their processing, ask questions, and open
the moments (a video at a timestamp, a PDF at a page).

    uvicorn moments.api:app --port 8000

Transcription uses the GPU, which holds one speech model at a time, so uploads wait in a queue and a
single worker processes them in order. The page polls the job for progress. Videos are streamed with
range requests, so the player can jump straight to a cited moment without downloading the whole file.

With MOMENTS_DEMO=1 (a hosted demo without a GPU) uploads are turned off and the page searches the
library it was given; sample videos stream from where they are published (meta "media_url").

Written answers are optional. Without an LLM API key (or with MOMENTS_ANSWERS=off) Moments is a search
product: a question returns the moments that answer it, which is everything the evaluation measures.
"""
import logging
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .answer import AnswerError, answer
from .library import Library, LibraryError

PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / ".env")
log = logging.getLogger("moments")

WEB = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GB
MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".ogv": "video/ogg", ".mkv": "video/x-matroska",
               ".mov": "video/quicktime", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
               ".ogg": "audio/ogg", ".opus": "audio/ogg", ".flac": "audio/flac",
               ".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
               ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


class Jobs:
    """Upload jobs, processed one at a time by a background worker."""

    def __init__(self, library, pool=None):
        self.library = library
        self.pool = pool
        self.state = {}
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, path, language=None, title=None):
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.state[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "title": title or path.stem,
                                  "video_id": None, "error": None}
        self.queue.put((job_id, path, language, title))
        return job_id

    def get(self, job_id):
        with self.lock:
            return dict(self.state[job_id]) if job_id in self.state else None

    def _update(self, job_id, **changes):
        with self.lock:
            self.state[job_id].update(changes)

    def _work(self):
        while True:
            job_id, path, language, title = self.queue.get()
            self._update(job_id, status="transcribing")
            try:
                if self.pool is None:
                    from polysub.asr import EnginePool

                    from .transcribe import FACTORIES

                    self.pool = EnginePool(factories=FACTORIES)
                video_id = self.library.ingest(path, pool=self.pool, language=language, title=title,
                                               progress=lambda f: self._update(job_id, progress=round(f, 3)),
                                               on_stage=lambda stage: self._update(job_id, status=stage, progress=0.0))
                self._update(job_id, status="done", progress=1.0, video_id=video_id)
            except LibraryError as error:
                self._update(job_id, status="failed", error=str(error))
            except Exception as error:  # a crash in one job must not stop the worker for the next
                self._update(job_id, status="failed", error=f"Something went wrong: {error}")


class Question(BaseModel):
    question: str
    video_id: str | None = None


def answers_enabled():
    return os.environ.get("MOMENTS_ANSWERS", "").lower() != "off" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def create_app(library=None, jobs=None, llm=None, demo=None, answers=None):
    library = library or Library(os.environ.get("MOMENTS_DATA", PROJECT / "data"))
    demo = os.environ.get("MOMENTS_DEMO") == "1" if demo is None else demo
    answers = (llm is not None or answers_enabled()) if answers is None else answers
    jobs = jobs or (None if demo else Jobs(library))
    app = FastAPI(title="Moments")
    uploads = library.root / "uploads"

    @app.get("/api/config")
    def config():
        return {"uploads": not demo, "answers": answers}

    @app.get("/api/items")
    def videos():
        return library.items()

    @app.post("/api/items", status_code=202)
    def upload(file: UploadFile = File(...), language: str = Form(""), title: str = Form("")):
        if demo:
            raise HTTPException(403, "Uploads are turned off in this demo. Try the sample videos.")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in MEDIA_TYPES:
            raise HTTPException(400, "That file type isn't supported. Upload a video, audio, PDF, Word or text file.")
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"{uuid.uuid4().hex[:8]}-{Path(file.filename).name}"
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out, length=1 << 20)
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            path.unlink()
            raise HTTPException(413, "That file is larger than 2 GB.")
        return {"job_id": jobs.submit(path, language=language or None, title=title or None)}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        state = jobs.get(job_id) if jobs else None
        if state is None:
            raise HTTPException(404, "No such job.")
        return state

    @app.get("/api/items/{video_id}/segments")
    def segments(video_id: str):
        try:
            return [s.to_dict() for s in library.segments(library.meta(video_id)["id"])]
        except LibraryError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/ask")
    def ask(body: Question):
        try:
            titles = {m["id"]: m["title"] for m in library.items()}
            hits = library.highlight(body.question, library.search(body.question, video_id=body.video_id, k=6))
        except LibraryError as error:
            raise HTTPException(400, str(error)) from error

        kinds = {m["id"]: m.get("kind", "video") for m in library.items()}

        def cite(video_id, start, end, text, english="", page=None, part=None, best=""):
            return {"video_id": video_id, "title": titles.get(video_id, video_id), "kind": kinds.get(video_id, "video"),
                    "start": start, "end": end, "page": page, "part": part, "text": text, "english": english,
                    "best": best}

        moments = [cite(h.video_id, h.segment.start, h.segment.end, h.segment.text, h.segment.english,
                        h.segment.page, h.segment.index + 1, h.best) for h in hits[:3]]
        if not answers:
            return {"answer": "", "found": bool(moments), "written": False, "citations": moments}

        try:
            result = answer(body.question, hits, titles, llm=llm)
        except AnswerError as error:
            # Search still works without the model: show the closest moments rather than failing the
            # whole question. The reason goes to the server log, not to visitors.
            log.warning("No written answer: %s", error)
            return {"answer": "A written answer isn't available right now. These are the closest moments:",
                    "found": False, "written": False, "citations": moments}
        return {"answer": result.text, "found": result.found, "written": True,
                "citations": [cite(c.video_id, c.start, c.end, c.text, c.english, c.page, c.part, c.best)
                              for c in result.citations]}

    @app.get("/media/{video_id}")
    def media(video_id: str):
        try:
            meta = library.meta(video_id)
        except LibraryError as error:
            raise HTTPException(404, str(error)) from error
        if meta.get("media_url"):
            # A published copy every browser can play (Chrome and Edge no longer play Ogg Theora).
            return RedirectResponse(meta["media_url"])
        source = Path(meta["source"])
        if not source.exists():
            raise HTTPException(404, "The original file has been moved or deleted.")
        # FileResponse answers range requests, which is what lets the player seek to a moment.
        return FileResponse(source, media_type=MEDIA_TYPES.get(source.suffix.lower(), "application/octet-stream"))

    @app.get("/")
    def page():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def __getattr__(name):
    # `uvicorn moments.api:app` builds the app on first use, so importing this module (as the tests do)
    # starts no worker thread and touches no data folder.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
