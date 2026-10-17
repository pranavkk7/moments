from fastapi.testclient import TestClient
from polysub.cues import Word

from moments.api import create_app
from moments.highlight import best_sentences, sentences

from .conftest import FakeEmbedder, transcript


def test_sentences_are_split_in_several_scripts():
    assert sentences("The exam is in March. Bring a pen! Any questions?") == \
        ["The exam is in March.", "Bring a pen!", "Any questions?"]
    assert sentences("試験は三月です。ペンを持ってきてください。") == ["試験は三月です。", "ペンを持ってきてください。"]
    assert sentences("परीक्षा मार्च में है। सभी छात्र अपना कलम लाइए।") == ["परीक्षा मार्च में है।", "सभी छात्र अपना कलम लाइए।"]
    assert sentences("") == []


def test_abbreviations_do_not_end_a_sentence():
    text = "This licence is used, e.g. by the Massachusetts Institute of Technology (MIT). It is the most restrictive."
    assert sentences(text) == ["This licence is used, e.g. by the Massachusetts Institute of Technology (MIT).",
                               "It is the most restrictive."]


def test_very_short_pieces_are_joined_to_the_sentence_before():
    assert sentences("It is used, e.g. by MIT. Next sentence here.") == \
        ["It is used, e.g. by MIT.", "Next sentence here."]


def test_the_sentence_closest_to_the_question_is_chosen_for_each_text():
    texts = ["Welcome to the course. The exam is in the last week of March. We start with clay.",
             "The office opens at nine. Exam results come in April.", ""]
    assert best_sentences(FakeEmbedder(), "when is the exam", texts) == \
        ["The exam is in the last week of March.", "Exam results come in April.", ""]


def test_results_carry_their_best_sentence(library, media):
    talk = ["intro words here"] * 60 + ["welcome back everyone. the exam is in march. bring a pen."] * 3
    library.add(media("lecture.mp4"), transcript(talk))
    client = TestClient(create_app(library=library, answers=False, demo=True))
    top = client.post("/api/ask", json={"question": "when is the exam"}).json()["citations"][0]
    assert top["best"] == "the exam is in march."


def test_videos_in_other_languages_get_their_best_sentence_from_the_english(library, media):
    ja = library.add(media("ja.mp4", b"j"), transcript(["試験 は 三月 です"] * 20, language="ja"))
    english = "Welcome to the class. The exam is in March. Bring a pen."
    library.set_english(ja, [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(english.split())])
    hit = library.highlight("when is the exam", library.search("when is the exam", video_id=ja, k=1))[0]
    assert hit.best == "The exam is in March."
