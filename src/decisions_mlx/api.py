"""The Jev / SystemOne ``POST /v1/systemone`` contract, independent of any model.

A request is parsed into a :class:`Request`, a :class:`Decider` (one per model family, see
``adapters``) turns it into a probability per option per question, and :func:`systemone` formats
those probabilities as the response body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

QUESTION_TYPES = ("noul", "choice", "score")


class RequestError(ValueError):
    """The request body does not follow the /v1/systemone contract."""


@dataclass(frozen=True)
class Request:
    model: str
    state: Any
    # question id -> question, with ``choice`` criteria normalized to an {option: description} dict.
    questions: dict[str, dict[str, Any]]
    # Images in whatever form the decider takes (PIL images for the Clef adapter).
    images: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    # question id -> option id -> probability; option ids as in :func:`option_ids`.
    probabilities: dict[str, dict[str, float]]
    input_tokens: int


class Decider(Protocol):
    def decide(self, request: Request) -> Decision: ...


def option_ids(question: dict[str, Any]) -> list[str]:
    """The answer's option ids, in the order the response lists them."""
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "choice":
        return [str(option) for option in question["criteria"]]
    return [str(level) for level in range(len(question["criteria"]))]


def parse_request(body: Any) -> Request:
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestError("model is required")
    if "state" not in body:
        raise RequestError("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("at least one question is required")
    images = body.get("images") or []
    if not isinstance(images, list):
        raise RequestError("images must be a list")
    return Request(
        model=body["model"],
        state=body["state"],
        questions={str(qid): _question(str(qid), question) for qid, question in questions.items()},
        images=images,
    )


def _question(question_id: str, question: Any) -> dict[str, Any]:
    if not isinstance(question, dict):
        raise RequestError(f"{question_id}: a question must be an object")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise RequestError(f"{question_id}: type must be noul, choice, or score")
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise RequestError(f"{question_id}: instructions must be a string")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
            raise RequestError(f"{question_id}: noul criteria may only describe true and false")
    elif kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(str(option) for option in criteria)
        if not isinstance(criteria, dict) or not criteria:
            raise RequestError(f"{question_id}: choice criteria must name at least one option")
        criteria = {str(option): description for option, description in criteria.items()}
    elif not isinstance(criteria, list) or not criteria:
        raise RequestError(f"{question_id}: score criteria must list at least one level")
    return {**question, "criteria": criteria}


def answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    """One question's answer: the probability of true, the most likely choice, or the expected score."""
    options = option_ids(question)
    rounded = {option: round(probabilities[option], 4) for option in options}
    if question["type"] == "noul":
        return {"type": "noul", "noul": rounded["true"]}
    best = max(options, key=probabilities.__getitem__)
    if question["type"] == "choice":
        return {"type": "choice", "choice": best, "confidence": rounded[best], "probabilities": rounded}
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(options)), 4),
        "confidence": rounded[best],
        "legend": dict(zip(options, question["criteria"])),
        "probabilities": rounded,
    }


def respond(request: Request, decision: Decision) -> dict[str, Any]:
    return {
        "model": request.model,
        "answers": {
            qid: answer(question, decision.probabilities[qid]) for qid, question in request.questions.items()
        },
        "usage": {"input_tokens": decision.input_tokens, "output_tokens": 0},
    }


def systemone(decider: Decider, body: Any) -> dict[str, Any]:
    """Answer a /v1/systemone request body with ``decider``."""
    request = parse_request(body)
    return respond(request, decider.decide(request))
