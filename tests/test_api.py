from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_run_endpoint():
    response = client.post(
        "/run",
        json={"task": "What is 25 * 17?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]
    assert data["latency"] > 0
    assert data["total_tokens"] > 0
    assert data["cost"] is not None