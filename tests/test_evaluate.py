import json

from moments.evaluate import evaluate, first_hit_rank, load_cases, overlaps, summarise
from moments.library import Hit
from moments.segments import Segment

from .conftest import transcript


def seg(start, end):
    return Segment(0, start, end, "")


def test_a_window_counts_when_it_overlaps_the_answer_enough():
    assert overlaps(seg(30, 70), [(60, 80)])  # 10 s of overlap
    assert not overlaps(seg(30, 61), [(60, 80)])  # 1 s is not enough
    assert overlaps(seg(30, 62), [(60, 61.5)])  # a short answer only needs to be covered whole
    assert overlaps(seg(0, 40), [(100, 120), (35, 50)])  # any of several answer ranges


def test_rank_of_the_first_hit():
    hits = [Hit("v", seg(0, 40), 0), Hit("v", seg(200, 240), 0), Hit("v", seg(100, 140), 0)]
    assert first_hit_rank(hits, [(110, 120)]) == 3
    assert first_hit_rank(hits, [(500, 510)]) is None


def test_summary_metrics():
    s = summarise([1, 2, None, 4])
    assert s["questions"] == 4
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5
    assert abs(s["mrr"] - (1 + 0.5 + 0.25) / 4) < 1e-9


def test_end_to_end_on_a_small_library(library, media, tmp_path):
    talk = ["intro words here"] * 60 + ["the exam is in march"] * 3 + ["more filler words"] * 60
    path = media("lecture.mp4")
    library.add(path, transcript(talk))
    exam_start = 60 * 3 * 0.5 + 60 * 1.0  # 60 three-word sentences of 0.5 s words, 1 s pauses
    questions = {"videos": [{"file": "lecture.mp4", "questions": [
        {"question": "when is the exam", "language": "en", "answer": [[exam_start, exam_start + 6]]}]}]}
    qpath = tmp_path / "questions.json"
    qpath.write_text(json.dumps(questions), "utf-8")
    cases = load_cases(qpath, library)
    assert cases[0]["video_language"] == "en" and cases[0]["ranges"] == [(exam_start, exam_start + 6)]
    results = evaluate(library, cases, k=3)
    assert results["hybrid"][("all", "")]["hit@3"] == 1.0
    assert set(results) == {"hybrid", "meaning", "keyword"}


def test_written_answers_are_scored_for_citations_and_for_admitting_gaps(library, media, tmp_path):
    from moments.evaluate import evaluate_answers

    from .test_answer import FakeLLM

    talk = ["intro words here"] * 60 + ["the exam is in march"] * 3 + ["more filler words"] * 60
    library.add(media("lecture.mp4"), transcript(talk))
    exam_start = 60 * 3 * 0.5 + 60 * 1.0
    questions = {"videos": [{"file": "lecture.mp4", "questions": [
        {"question": "when is the exam", "language": "en", "answer": [[exam_start, exam_start + 6]]},
        {"question": "where is the car park", "language": "en", "answer": []}]}]}
    qpath = tmp_path / "q.json"
    qpath.write_text(json.dumps(questions), "utf-8")
    cases = load_cases(qpath, library)
    assert evaluate(library, cases, k=3)["hybrid"][("all", "")]["questions"] == 1  # the gap is not searched for

    class ByQuestion(FakeLLM):
        def complete(self, system, prompt, schema):
            if "car park" in prompt:
                return {"found": False, "answer": "The video does not cover that.", "moments": []}
            line = next(x for x in prompt.splitlines() if "exam is in march" in x)
            return {"found": True, "answer": "In March.", "moments": [int(line[1:line.index("]")])]}

    scores = evaluate_answers(library, cases, llm=ByQuestion({}))
    assert scores["cited right"] == 1.0 and scores["said not covered"] == 1.0
    assert scores["answerable"] == 1 and scores["not covered"] == 1
