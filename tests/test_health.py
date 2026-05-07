from fastapi.testclient import TestClient


def test_health_returns_service_metadata(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "SupportSmith",
        "version": "0.1.0",
        "environment": "test",
        "status": "ok",
        "database": {
            "status": "ok",
            "detail": None,
        },
    }
