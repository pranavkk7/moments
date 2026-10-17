import json

import pytest
from fastapi.testclient import TestClient

from moments.api import create_app
from moments.documents import Page, clean, detect_language, passages, read_pages
from moments.evaluate import evaluate, load_cases
from moments.library import LibraryError

from .conftest import transcript


def make_pdf(path, pages):
    """A minimal real PDF with one line of Helvetica text per page, written by hand so the tests need no
    PDF-making library."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 11 Tf 40 760 Td ({text}) Tj ET".encode("latin-1")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream")
        content = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R "
                       f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


SOLAR = ["The Sun is a medium sized star at the edge of the Milky Way.",
         "Jupiter is the largest planet and has a thin cloud of sodium.",
         "A neutron star is so dense that a spoonful weighs a billion tonnes."]


def test_pdf_text_is_read_page_by_page(tmp_path):
    pages = read_pages(make_pdf(tmp_path / "solar.pdf", SOLAR))
    assert [p.number for p in pages] == [1, 2, 3]
    assert "Jupiter is the largest planet" in pages[1].text


def test_word_and_text_files_have_no_pages(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("First paragraph about stars.")
    doc.add_paragraph("Second paragraph about planets.")
    doc.save(tmp_path / "notes.docx")
    pages = read_pages(tmp_path / "notes.docx")
    assert len(pages) == 1 and pages[0].number is None and "Second paragraph" in pages[0].text
    (tmp_path / "notes.txt").write_text("Plain text notes.", "utf-8")
    assert read_pages(tmp_path / "notes.txt")[0].text == "Plain text notes."


def test_language_is_detected_from_script_or_common_words():
    assert detect_language("The star is the closest one to the Sun and it is bright.") == "en"
    assert detect_language("La estrella más cercana es la que está en el centro de la galaxia.") == "es"
    assert detect_language("सेंगोल संसद में स्थापित किया गया") == "hi"
    assert detect_language("الجص مادة تعود إلى أصلها") == "ar"
    assert detect_language("クリエイティブ・コモンズとは何ですか") == "ja"


def test_passages_stay_on_their_page_and_overlap():
    words = " ".join(f"w{i}" for i in range(400))
    result = passages([Page(1, words), Page(2, "a short second page")], "en", words=150, overlap=30)
    on_first = [p for p in result if p.page == 1]
    assert len(on_first) == 4 and all(len(p.text.split()) <= 150 for p in on_first)
    assert on_first[-1].text.endswith("w399")  # the end of the page is covered
    assert on_first[0].text.split()[-30:] == on_first[1].text.split()[:30]  # overlap
    assert result[-1].page == 2 and result[-1].text == "a short second page"
    assert [p.index for p in result] == list(range(len(result)))


def test_pdf_line_breaks_and_hyphenation_are_cleaned():
    assert clean("a neu-\ntron star\n\nis  dense") == "a neutron star is dense"
    assert clean("commer -\ncial use") == "commercial use"  # some PDFs put a space before the hyphen


def test_documents_are_searched_together_with_videos(library, media, tmp_path):
    library.add(media("lecture.mp4"), transcript(["intro words here"] * 60 + ["the exam is in march"] * 3))
    doc = library.add_document(make_pdf(tmp_path / "solar.pdf", SOLAR), title="Solar system notes")
    meta = library.meta(doc)
    assert meta["kind"] == "document" and meta["pages"] == 3 and meta["language"] == "en"
    hit = library.search("which planet is the largest", k=1)[0]
    assert hit.video_id == doc and hit.segment.page == 2
    assert {m["kind"] for m in library.items()} == {"video", "document"}


def test_ingest_reads_documents_without_transcribing(library, tmp_path, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("documents must not be transcribed")

    monkeypatch.setattr("moments.library.transcribe", never)
    stages = []
    doc = library.ingest(make_pdf(tmp_path / "solar.pdf", SOLAR), on_stage=stages.append)
    assert stages == ["reading"] and library.meta(doc)["kind"] == "document"
    assert library.translate(doc) is False  # Whisper translates speech, not text


def test_a_document_without_text_is_refused_clearly(library, tmp_path):
    with pytest.raises(LibraryError, match="OCR"):
        library.add_document(make_pdf(tmp_path / "scan.pdf", [""]))
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(LibraryError):
        library.add_document(tmp_path / "broken.pdf")


def test_document_results_point_to_their_page(library, tmp_path):
    doc = library.add_document(make_pdf(tmp_path / "solar.pdf", SOLAR), title="Solar system notes")
    client = TestClient(create_app(library=library, answers=False, demo=True))
    body = client.post("/api/ask", json={"question": "How dense is a neutron star?"}).json()
    top = body["citations"][0]
    assert top["kind"] == "document" and top["page"] == 3 and "spoonful" in top["text"]
    pdf = client.get(f"/media/{doc}")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"


def test_document_questions_are_scored_by_page(library, tmp_path):
    library.add_document(make_pdf(tmp_path / "solar.pdf", SOLAR))
    questions = {"videos": [{"file": "solar.pdf", "questions": [
        {"question": "Which planet is the largest?", "language": "en", "pages": [2]}]}]}
    qpath = tmp_path / "q.json"
    qpath.write_text(json.dumps(questions), "utf-8")
    results = evaluate(library, load_cases(qpath, library), k=3)
    assert results["hybrid"][("all", "")]["hit@1"] == 1.0
    assert ("en document", "same language") in results["hybrid"]


def test_mixed_search_averages_over_the_rankings_each_item_can_appear_in(library, media, tmp_path):
    from polysub.cues import Word

    ja = library.add(media("ja.mp4", b"j"), transcript(["土 の 話 を します"] * 40, language="ja"))
    library.set_english(ja, [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(("talk about soil " * 40).split())])
    library.add_document(make_pdf(tmp_path / "solar.pdf", SOLAR))
    for hit in library.search("Which planet is the largest?", k=10):
        # An English question: the Japanese video can appear in its meaning ranking and the two English
        # (translation) rankings, the English document in its own meaning and keyword rankings.
        lists = 3 if hit.video_id == ja else 2
        assert hit.score == pytest.approx(sum(1 / (60 + r) for r in hit.ranks.values()) / lists, abs=1e-5)
    top = library.search("Which planet is the largest?", k=1)[0]
    assert top.segment.page == 2  # the document's answer is not outranked by a video in more rankings


def test_running_headers_and_footers_are_removed():
    from moments.documents import remove_running_lines

    topics = ["stars", "planets", "comets", "moons", "galaxies", "nebulae"]
    pages = [Page(n, f"3. The Creative Commons scheme {n}\nA page about {topic}.\nWhy {topic} matter.\n{n}")
             for n, topic in zip(range(10, 16), topics)]
    cleaned = remove_running_lines(pages)
    assert all("Creative Commons scheme" not in p.text for p in cleaned)
    assert all(f"A page about {topic}." in p.text for p, topic in zip(cleaned, topics))
    short = [Page(1, "Title\nbody"), Page(2, "Title\nother")]  # too few pages to tell a header from text
    assert remove_running_lines(short) == short


def test_small_capitals_are_put_back_only_for_words_the_document_also_writes_normally():
    from moments.documents import fix_small_caps

    vocabulary = {"creative", "commons", "licencing", "scheme"}
    text = "The CreATive COmmOns liCenCing sCheme, used by OpenCourseWare, with PDFs and YouTube."
    assert fix_small_caps(text, vocabulary) == \
        "The Creative Commons licencing scheme, used by OpenCourseWare, with PDFs and YouTube."


class FakeTextTranslator:
    """Translates by a word list, so the tests need no translation model."""
    WORDS = {"la": "the", "tierra": "earth", "es": "is", "donde": "where", "vivimos": "we live",
             "marte": "mars", "rojo": "red", "planeta": "planet", "júpiter": "jupiter", "grande": "big",
             "saturno": "saturn", "tiene": "has", "anillos": "rings", "venus": "venus", "caliente": "hot"}

    def translate(self, texts, language, progress=None):
        return [" ".join(self.WORDS.get(w.strip(".").lower(), w) for w in t.split()) for t in texts]


PLANETAS = ["La Tierra es donde vivimos.", "Marte es el planeta rojo.", "Jupiter es grande.",
            "Saturno tiene anillos.", "Venus es caliente."]


def test_documents_in_other_languages_are_translated_into_english(library, tmp_path):
    library._text_translator = FakeTextTranslator()
    stages = []
    doc = library.ingest(make_pdf(tmp_path / "planetas.pdf", PLANETAS), on_stage=stages.append)
    assert library.meta(doc)["language"] == "es" and library.meta(doc)["english"] is True
    assert stages == ["reading", "translating"]
    assert library.segments(doc)[1].english.startswith("mars is el planet red")
    # The English is searched too: the question's words appear in the translation, not the Spanish.
    hit = library.search("Which planet is red?", video_id=doc, k=1)[0]
    assert hit.segment.page == 2 and set(hit.ranks) & {"meaning_en", "keyword_en"}


def test_document_results_show_their_english_first(library, tmp_path):
    library._text_translator = FakeTextTranslator()
    library.ingest(make_pdf(tmp_path / "planetas.pdf", PLANETAS))
    client = TestClient(create_app(library=library, answers=False, demo=True))
    top = client.post("/api/ask", json={"question": "Which planet is red?"}).json()["citations"][0]
    assert top["page"] == 2 and top["english"].startswith("mars") and top["text"].startswith("Marte")


def test_documents_are_served_from_the_library_even_with_a_published_copy(library, tmp_path):
    # Wikimedia asks that its files not be framed, and the page shows PDFs in a frame.
    doc = library.add_document(make_pdf(tmp_path / "solar.pdf", SOLAR))
    library.update_meta(doc, media_url="https://upload.example.org/solar.pdf")
    client = TestClient(create_app(library=library, answers=False, demo=True))
    local = client.get(f"/media/{doc}", follow_redirects=False)
    assert local.status_code == 200 and local.headers["content-type"] == "application/pdf"
    (tmp_path / "solar.pdf").unlink()  # without the file (a hosted demo), the published copy is used
    assert client.get(f"/media/{doc}", follow_redirects=False).status_code == 307
