"""Answers from the retrieved moments, each claim pointing to a moment the viewer can jump to.

The model sees only the numbered moments that search returned, never the whole video, and must reply in
a fixed JSON shape: the answer, whether the moments contain it at all, and which moment numbers support
it. The code then checks every cited number against the moments it actually sent, so a citation can
never point to a time that was not retrieved. If nothing relevant was found, the answer says so instead
of guessing.

The model provider sits behind a small interface (`complete(system, prompt, schema) -> dict`), so the
rest of Moments does not depend on any one provider. The provider and model are settings in `.env`.
"""
import json
import os
from dataclasses import dataclass, field

from polysub import languages

from .segments import format_time

DEFAULT_MODEL = os.environ.get("MOMENTS_LLM_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("MOMENTS_LLM_EFFORT", "low")  # short grounded answers need little thinking

SYSTEM_PROMPT = """You answer questions about videos using only the numbered moments given to you. Each \
moment is a short stretch of the video's transcript with its start time; transcripts can be in any \
language and may contain recognition errors.

- Answer in the language of the question, in at most 120 words, as plain text.
- Use only what the moments say. Do not add facts from general knowledge.
- Cite the moments that support your answer by their numbers. Cite only moments that really contain the \
information; usually one to three.
- If the moments do not contain the answer, set "found" to false, say briefly that the video does not \
seem to cover it, and cite nothing.
Return only the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "moments": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "moments"],
    "additionalProperties": False,
}


class AnswerError(Exception):
    pass


@dataclass
class Citation:
    video_id: str
    start: float
    end: float
    text: str


@dataclass
class Answer:
    text: str
    found: bool
    citations: list = field(default_factory=list)
    model: str = ""


def build_prompt(question, hits, titles):
    lines = [f"Question: {question}", "", "Moments:"]
    for number, hit in enumerate(hits, start=1):
        s = hit.segment
        title = titles.get(hit.video_id, hit.video_id)
        language = languages.name(hit.language) if hit.language else ""
        lines.append(f"[{number}] {title}, {format_time(s.start)}-{format_time(s.end)}"
                     f"{f' ({language})' if language else ''}: {s.text}")
    return "\n".join(lines)


def answer(question, hits, titles, llm=None):
    """Asks the model about the retrieved moments and returns an Answer with checked citations."""
    if not hits:
        return Answer("I couldn't find anything about that in the video.", found=False)
    llm = llm or AnthropicLLM()
    reply = llm.complete(SYSTEM_PROMPT, build_prompt(question, hits, titles), SCHEMA)
    text = str(reply.get("answer", "")).strip()
    found = bool(reply.get("found")) and bool(text)
    # Keep only citations that name a moment we actually sent, in order, without repeats.
    cited = []
    for number in reply.get("moments", []):
        if isinstance(number, int) and 1 <= number <= len(hits) and number not in cited:
            cited.append(number)
    citations = [Citation(hits[n - 1].video_id, hits[n - 1].segment.start, hits[n - 1].segment.end,
                          hits[n - 1].segment.text) for n in cited] if found else []
    return Answer(text or "The video does not seem to cover that.", found=found and bool(citations),
                  citations=citations, model=llm.label)


class AnthropicLLM:
    """The model provider used for testing (the same one as the author's other projects)."""

    def __init__(self, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, client=None):
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise AnswerError("No API key: set ANTHROPIC_API_KEY in .env to get written answers.")
            # The endpoint is Moments' own setting: a base URL inherited from another tool's environment
            # (developer tools set ANTHROPIC_BASE_URL for their own use) must not redirect the app's calls.
            base_url = os.environ.get("MOMENTS_LLM_BASE_URL") or "https://api.anthropic.com"
            client = anthropic.Anthropic(base_url=base_url, timeout=60.0, max_retries=2)
        self.client = client
        self.model = model
        self.effort = effort
        self.label = model

    def complete(self, system, prompt, schema):
        import anthropic

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                # If the model declines on safety grounds, the API retries on a suitable fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as error:
            raise AnswerError("The API key was not accepted. Check ANTHROPIC_API_KEY.") from error
        except anthropic.RateLimitError as error:
            raise AnswerError("Too many requests right now. Try again in a minute.") from error
        except anthropic.APIConnectionError as error:
            raise AnswerError("Could not reach the model provider. Check the connection.") from error
        except anthropic.APIStatusError as error:
            raise AnswerError(f"The model provider returned an error ({error.status_code}).") from error
        if response.stop_reason == "refusal":
            raise AnswerError("The model declined to answer this question.")
        if response.stop_reason == "max_tokens":
            raise AnswerError("The answer was cut off. Try a narrower question.")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise AnswerError("The model returned a reply that was not valid JSON.") from error
