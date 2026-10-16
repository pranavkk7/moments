import io

from moments import cli

from .conftest import transcript


def run(library, *argv):
    out = io.StringIO()
    code = cli.main(list(argv), library=library, out=out)
    return code, out.getvalue()


def test_list_and_search(library, media):
    code, text = run(library, "list")
    assert code == 0 and "empty" in text
    library.add(media("lecture.mp4"), transcript(["intro words here"] * 20 + ["the exam is in march"] * 3),
                title="Lecture one")
    code, text = run(library, "list")
    assert "Lecture one" in text and "[en, " in text
    code, text = run(library, "search", "when is the exam", "-k", "1")
    assert code == 0
    assert "Lecture one" in text and "exam is in march" in text


def test_add_uses_the_transcriber(library, media, monkeypatch):
    monkeypatch.setattr("moments.library.transcribe",
                        lambda path, **kw: transcript(["a short talk about clay"] * 10))
    code, text = run(library, "add", str(media("clay.mp4")), "--title", "Clay talk")
    assert code == 0 and "Added Clay talk (en" in text


def test_errors_are_reported_without_a_traceback(library, capsys):
    code, _ = run(library, "search", "anything")
    assert code == 1
    assert "library is empty" in capsys.readouterr().err
