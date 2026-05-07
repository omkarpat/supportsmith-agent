from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.postgres import DatabaseHealth, PostgresDatabase
from app.main import create_app


class FakeDatabase:
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health(self) -> DatabaseHealth:
        return DatabaseHealth(status="ok")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="postgresql://supportsmith:supportsmith@localhost:55432/supportsmith_test",
        )
    )

    with TestClient(app) as test_client:
        yield test_client
