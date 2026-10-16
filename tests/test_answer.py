import json
from types import SimpleNamespace

import pytest

from moments.answer import AnswerError, AnthropicLLM, answer, build_prompt
from moments.library import Hit
from moments.segments import Segment


class FakeLLM:
    label = "fake-model"

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, system, prompt, schema):
        self.prompts.append(prompt)
        return self.reply


def hits():
    return [
        Hit("lecture-1", Segment(0, 0.0, 40.0, "welcome to the course"), 0.03, language="en"),
        Hit("lecture-1", Segment(5, 600.0, 640.0, "the exam is in the last week of march"), 0.02, language="en"),
        Hit("talk-2", Segment(2, 75.0, 118.0, "പരീക്ഷ മാർച്ചിൽ"), 0.01, language="ml"),
    ]


TITLES = {"lecture-1": "Soil mechanics", "talk-2": "Malayalam talk"}


def test_prompt_numbers_each_moment_with_title_time_and_language():
    prompt = build_prompt("When is the exam?", hits(), TITLES)
    assert "Question: When is the exam?" in prompt
    assert "[2] Soil mechanics, 10:00-10:40 (English): the exam is in the last week of march" in prompt
    assert "[3] Malayalam talk, 1:15-1:58 (Malayalam): പരീക്ഷ മാർച്ചിൽ" in prompt


def test_citations_point_to_the_moments_that_were_sent():
    llm = FakeLLM({"found": True, "answer": "In the last week of March.", "moments": [2, 3]})
    result = answer("When is the exam?", hits(), TITLES, llm=llm)
    assert result.found and result.text == "In the last week of March."
    assert [(c.video_id, c.start) for c in result.citations] == [("lecture-1", 600.0), ("talk-2", 75.0)]
    assert result.model == "fake-model"


def test_invented_or_repeated_citations_are_dropped():
    llm = FakeLLM({"found": True, "answer": "March.", "moments": [9, 2, 2, 0, "1"]})
    result = answer("When is the exam?", hits(), TITLES, llm=llm)
    assert [c.start for c in result.citations] == [600.0]


def test_an_answer_with_no_valid_citation_is_not_presented_as_found():
    llm = FakeLLM({"found": True, "answer": "March, probably.", "moments": [7]})
    result = answer("When is the exam?", hits(), TITLES, llm=llm)
    assert not result.found and result.citations == []


def test_not_covered_cites_nothing():
    llm = FakeLLM({"found": False, "answer": "The video does not cover parking.", "moments": [1]})
    result = answer("Where do I park?", hits(), TITLES, llm=llm)
    assert not result.found and result.citations == []
    assert "parking" in result.text


def test_no_hits_needs_no_model_call():
    result = answer("anything", [], TITLES, llm=None)
    assert not result.found


def response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    def __init__(self, reply):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.reply = reply

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def test_provider_request_asks_for_json_in_the_schema():
    client = FakeClient(response(json.dumps({"found": True, "answer": "Yes.", "moments": [1]})))
    llm = AnthropicLLM(model="test-model", effort="low", client=client)
    assert llm.complete("system", "prompt", {"type": "object"}) == {"found": True, "answer": "Yes.", "moments": [1]}
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}


@pytest.mark.parametrize("stop_reason, message", [("refusal", "declined"), ("max_tokens", "cut off")])
def test_provider_stop_reasons_become_clear_errors(stop_reason, message):
    llm = AnthropicLLM(client=FakeClient(response("{}", stop_reason)))
    with pytest.raises(AnswerError, match=message):
        llm.complete("s", "p", {})


def test_provider_reply_that_is_not_json_is_an_error():
    llm = AnthropicLLM(client=FakeClient(response("not json")))
    with pytest.raises(AnswerError, match="not valid JSON"):
        llm.complete("s", "p", {})
