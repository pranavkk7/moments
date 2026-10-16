import time

import pytest
from fastapi.testclient import TestClient

from moments.api import Jobs, create_app

from .conftest import transcript
from .test_answer import FakeLLM


class FakePool:
    """Never used for real work: the library's transcriber is replaced below."""

    def close(self):
        pass


@pytest.fixture
def client(library, monkeypatch):
    monkeypatch.setattr("moments.library.transcribe",
                        lambda path, **kw: transcript(["intro words here"] * 100 + ["the exam is in march"] * 3))
    llm = FakeLLM({"found": True, "answer": "In March.", "moments": [1]})
    app = create_app(library=library, jobs=Jobs(library, pool=FakePool()), llm=llm)
    return TestClient(app)


def wait_for(client, job_id):
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_transcribe_and_ask(client, tmp_path):
    assert client.get("/api/videos").json() == []
    response = client.post("/api/videos", files={"file": ("lecture.mp4", b"fake video bytes", "video/mp4")},
                           data={"title": "Lecture one"})
    assert response.status_code == 202
    job = wait_for(client, response.json()["job_id"])
    assert job["status"] == "done" and job["video_id"]

    videos = client.get("/api/videos").json()
    assert [v["title"] for v in videos] == ["Lecture one"]
    assert client.get(f"/api/videos/{job['video_id']}/segments").json()[0]["start"] == 0.0

    body = client.post("/api/ask", json={"question": "when is the exam", "video_id": job["video_id"]}).json()
    assert body["answer"] == "In March." and body["found"]
    assert body["citations"][0]["title"] == "Lecture one"
    assert "exam is in march" in body["citations"][0]["text"]


def test_media_is_streamed_with_range_requests(client):
    job_id = client.post("/api/videos", files={"file": ("clip.mp4", b"0123456789" * 100, "video/mp4")}).json()["job_id"]
    video_id = wait_for(client, job_id)["video_id"]
    response = client.get(f"/media/{video_id}", headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == b"0123456789"
    assert response.headers["content-type"] == "video/mp4"


def test_unsupported_files_and_unknown_ids_are_refused(client):
    assert client.post("/api/videos", files={"file": ("notes.pdf", b"%PDF", "application/pdf")}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/media/nope").status_code == 404
    assert client.post("/api/ask", json={"question": "anything"}).status_code == 400  # empty library


def test_the_page_and_its_assets_are_served(client):
    assert "Moments" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_without_a_working_model_the_closest_moments_are_still_shown(library, monkeypatch):
    from moments.answer import AnswerError

    class BrokenLLM:
        label = "broken"

        def complete(self, *args):
            raise AnswerError("No API key: set ANTHROPIC_API_KEY in .env to get written answers.")

    monkeypatch.setattr("moments.library.transcribe",
                        lambda path, **kw: transcript(["intro words here"] * 100 + ["the exam is in march"] * 3))
    client = TestClient(create_app(library=library, jobs=Jobs(library, pool=FakePool()), llm=BrokenLLM()))
    job_id = client.post("/api/videos", files={"file": ("a.mp4", b"x", "video/mp4")}).json()["job_id"]
    wait_for(client, job_id)
    body = client.post("/api/ask", json={"question": "when is the exam"}).json()
    assert body["written"] is False and "isn't available" in body["answer"]
    assert "API" not in body["answer"]  # the technical reason is logged, not shown to visitors
    assert len(body["citations"]) == 3 and "exam is in march" in body["citations"][0]["text"]


def test_demo_mode_turns_uploads_off_and_streams_published_copies(library, media):
    video_id = library.add(media("sample.ogv"), transcript(["hello there"] * 20))
    library.update_meta(video_id, media_url="https://upload.example.org/sample.webm",
                        credit={"author": "A. Author", "license": "CC BY-SA 4.0", "page": "https://example.org"})
    client = TestClient(create_app(library=library, llm=FakeLLM({}), demo=True))
    assert client.get("/api/config").json()["uploads"] is False
    refused = client.post("/api/videos", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert refused.status_code == 403 and "turned off" in refused.json()["detail"]
    redirect = client.get(f"/media/{video_id}", follow_redirects=False)
    assert redirect.status_code == 307 and redirect.headers["location"].endswith("sample.webm")
    assert client.get("/api/videos").json()[0]["credit"]["license"] == "CC BY-SA 4.0"


def test_without_an_api_key_moments_is_a_search_product(library, monkeypatch):
    monkeypatch.setattr("moments.library.transcribe",
                        lambda path, **kw: transcript(["intro words here"] * 100 + ["the exam is in march"] * 3))
    client = TestClient(create_app(library=library, jobs=Jobs(library, pool=FakePool()), answers=False))
    assert client.get("/api/config").json()["answers"] is False
    job_id = client.post("/api/videos", files={"file": ("a.mp4", b"x", "video/mp4")}).json()["job_id"]
    wait_for(client, job_id)
    body = client.post("/api/ask", json={"question": "when is the exam"}).json()
    assert body == {"answer": "", "found": True, "written": False, "citations": body["citations"]}
    assert "exam is in march" in body["citations"][0]["text"]


def test_answers_are_off_without_a_key_or_when_switched_off(monkeypatch):
    from moments.api import answers_enabled

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOMENTS_ANSWERS", raising=False)  # a developer's own .env may set it
    assert not answers_enabled()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert answers_enabled()
    monkeypatch.setenv("MOMENTS_ANSWERS", "off")
    assert not answers_enabled()


def test_moments_carry_their_english_translation(library, media):
    from polysub.cues import Word

    ja = library.add(media("ja.mp4", b"j"), transcript(["MIT の 講義 です"] * 20, language="ja"))
    library.set_english(ja, [Word("A", 1.0, 1.2), Word("lecture", 1.3, 1.8), Word("at", 1.9, 2.0), Word("MIT.", 2.1, 2.5)])
    client = TestClient(create_app(library=library, answers=False, demo=True))
    body = client.post("/api/ask", json={"question": "MIT lecture", "video_id": ja}).json()
    assert body["citations"][0]["english"] == "A lecture at MIT."
    assert "MIT" in body["citations"][0]["text"]
