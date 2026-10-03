import base64
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from decisions_mlx import Decision, Request, RequestError
from decisions_mlx.server import make_server

BODY = {"model": "jev-latest", "state": "Is it down?", "questions": {"down": {"type": "noul"}}}


class Recorder:
    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[Request] = []
        self.error = error

    def decide(self, request: Request) -> Decision:
        self.requests.append(request)
        if self.error:
            raise self.error
        return Decision({qid: {"true": 0.75, "false": 0.25} for qid in request.questions}, 5)


@pytest.fixture
def serve():
    servers = []

    def start(deciders):
        server = make_server(deciders, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()


def call(url, body=None, raw=None):
    data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_single_model_answers_any_model_name(serve):
    url = serve({"JevK5": Recorder()})
    status, response = call(f"{url}/v1/systemone", BODY)
    assert status == 200
    assert response == {
        "model": "jev-latest",
        "answers": {"down": {"type": "noul", "noul": 0.75}},
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }


def test_several_models_route_by_name(serve):
    clef, jevk5 = Recorder(), Recorder()
    url = serve({"clef-flash": clef, "JevK5": jevk5})
    assert call(f"{url}/v1/systemone", {**BODY, "model": "JevK5"})[0] == 200
    assert (len(clef.requests), len(jevk5.requests)) == (0, 1)
    status, response = call(f"{url}/v1/systemone", BODY)
    assert status == 400 and "unknown model" in response["error"]
    assert call(f"{url}/v1/models")[1]["data"] == [
        {"id": "clef-flash", "object": "model"},
        {"id": "JevK5", "object": "model"},
    ]


def test_images_arrive_as_pil_images(serve):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    decider = Recorder()
    url = serve({"clef": decider})
    assert call(f"{url}/v1/systemone", {**BODY, "images": [data_url]})[0] == 200
    (image,) = decider.requests[0].images
    assert image.size == (4, 3) and image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    "body, raw, message",
    [
        (None, b"{not json", "invalid JSON"),
        ({**BODY, "questions": {}}, None, "at least one question"),
        ({**BODY, "images": ["https://example.com/a.png"]}, None, "data:image"),
    ],
)
def test_bad_requests_get_400(serve, body, raw, message):
    status, response = call(f"{serve({'m': Recorder()})}/v1/systemone", body, raw)
    assert status == 400 and message in response["error"]


def test_decider_errors_are_reported(serve):
    url = serve({"text": Recorder(RequestError("text only")), "broken": Recorder(RuntimeError("boom"))})
    assert call(f"{url}/v1/systemone", {**BODY, "model": "text"}) == (400, {"error": "text only"})
    assert call(f"{url}/v1/systemone", {**BODY, "model": "broken"}) == (500, {"error": "RuntimeError: boom"})
