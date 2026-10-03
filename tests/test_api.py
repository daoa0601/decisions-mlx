import pytest

from decisions_mlx import Decision, Request, RequestError, parse_request, systemone

BODY = {
    "model": "jev-latest",
    "state": "Checkout is down.",
    "questions": {
        "outage": {"type": "noul", "instructions": "Is a service down?"},
        "team": {"type": "choice", "criteria": {"billing": "Payments", "technical": "Bugs"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    },
}


class FixedDecider:
    """Answers every question with fixed probabilities and remembers the request it was given."""

    def __init__(self, probabilities: dict[str, dict[str, float]]) -> None:
        self.probabilities = probabilities
        self.requests: list[Request] = []

    def decide(self, request: Request) -> Decision:
        self.requests.append(request)
        return Decision(self.probabilities, 42)


def test_systemone_formats_every_question_type():
    decider = FixedDecider(
        {
            "outage": {"true": 0.91234, "false": 0.08766},
            "team": {"billing": 0.2, "technical": 0.8},
            "urgency": {"0": 0.1, "1": 0.3, "2": 0.6},
        }
    )
    response = systemone(decider, BODY)
    assert response == {
        "model": "jev-latest",
        "answers": {
            "outage": {"type": "noul", "noul": 0.9123},
            "team": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.8,
                "probabilities": {"billing": 0.2, "technical": 0.8},
            },
            "urgency": {
                "type": "score",
                "score": 1.5,
                "confidence": 0.6,
                "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            },
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


def test_choice_criteria_as_a_list_become_options_without_descriptions():
    request = parse_request({**BODY, "questions": {"team": {"type": "choice", "criteria": ["billing", "technical"]}}})
    assert request.questions["team"]["criteria"] == {"billing": None, "technical": None}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model": None}, "model is required"),
        ({"state": ...}, "state is required"),
        ({"questions": {}}, "at least one question"),
        ({"questions": {"q": {"type": "maybe"}}}, "type must be"),
        ({"questions": {"q": {"type": "choice", "criteria": {}}}}, "at least one option"),
        ({"questions": {"q": {"type": "score"}}}, "at least one level"),
        ({"questions": {"q": {"type": "noul", "criteria": {"yes": "y"}}}}, "only describe true and false"),
        ({"questions": {"q": {"type": "noul", "instructions": 3}}}, "instructions must be a string"),
        ({"images": "a.png"}, "images must be a list"),
    ],
)
def test_invalid_requests_are_rejected(change, message):
    body = {**BODY, **change}
    if change.get("state") is ...:
        del body["state"]
    with pytest.raises(RequestError, match=message):
        parse_request(body)
