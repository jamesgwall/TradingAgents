"""The transcript analyst must query with the same embedding setup as ingestion."""

import json

from tradingagents.dataflows import transcript_store


def test_qwen_embedding_request(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps({"embeddings": [[0.0] * 1024]}).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(transcript_store.urllib.request, "urlopen", fake_urlopen)
    assert len(transcript_store._embed_text("AAPL macro outlook", "http://localhost:11434")) == 1024
    assert requests == [{
        "model": "qwen3-embedding:latest",
        "input": ["AAPL macro outlook"],
        "dimensions": 1024,
    }]
