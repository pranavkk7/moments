"""The web API and page: upload a video, follow its transcription, ask questions, play the moments.

    uvicorn moments.api:app --port 8000

Transcription uses the GPU, which holds one speech model at a time, so uploads wait in a queue and a
single worker processes them in order. The page polls the job for progress. Videos are streamed with
range requests, so the player can jump straight to a cited moment without downloading the whole file.
"""
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .answer import AnswerError, answer
from .library import Library, LibraryError

PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / ".env")

WEB = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GB
MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".ogv": "video/ogg", ".mkv": "video/x-matroska",
               ".mov": "video/quicktime", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
               ".ogg": "audio/ogg", ".opus": "audio/ogg", ".flac": "audio/flac"}


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
                                               progress=lambda f: self._update(job_id, progress=round(f, 3)))
                self._update(job_id, status="done", progress=1.0, video_id=video_id)
            except LibraryError as error:
                self._update(job_id, status="failed", error=str(error))
            except Exception as error:  # a crash in one job must not stop the worker for the next
                self._update(job_id, status="failed", error=f"Something went wrong: {error}")


class Question(BaseModel):
    question: str
    video_id: str | None = None


def create_app(library=None, jobs=None, llm=None):
    library = library or Library(os.environ.get("MOMENTS_DATA", PROJECT / "data"))
    jobs = jobs or Jobs(library)
    app = FastAPI(title="Moments")
    uploads = library.root / "uploads"

    @app.get("/api/videos")
    def videos():
        return library.videos()

    @app.post("/api/videos", status_code=202)
    def upload(file: UploadFile = File(...), language: str = Form(""), title: str = Form("")):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in MEDIA_TYPES:
            raise HTTPException(400, "That file type isn't supported. Upload a video or audio file.")
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
        state = jobs.get(job_id)
        if state is None:
            raise HTTPException(404, "No such job.")
        return state

    @app.get("/api/videos/{video_id}/segments")
    def segments(video_id: str):
        try:
            return [s.to_dict() for s in library.segments(library.meta(video_id)["id"])]
        except LibraryError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/ask")
    def ask(body: Question):
        try:
            titles = {m["id"]: m["title"] for m in library.videos()}
            hits = library.search(body.question, video_id=body.video_id, k=6)
            result = answer(body.question, hits, titles, llm=llm)
        except LibraryError as error:
            raise HTTPException(400, str(error)) from error
        except AnswerError as error:
            raise HTTPException(502, str(error)) from error
        return {
            "answer": result.text,
            "found": result.found,
            "citations": [{"video_id": c.video_id, "title": titles.get(c.video_id, c.video_id),
                           "start": c.start, "end": c.end, "text": c.text} for c in result.citations],
        }

    @app.get("/media/{video_id}")
    def media(video_id: str):
        try:
            source = Path(library.meta(video_id)["source"])
        except LibraryError as error:
            raise HTTPException(404, str(error)) from error
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
