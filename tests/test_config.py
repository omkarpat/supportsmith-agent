import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_require_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SUPPORTSMITH_DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_accept_database_url_field_name() -> None:
    settings = Settings(
        environment="test",
        database_url="postgresql://supportsmith:supportsmith@localhost:55432/supportsmith_test",
    )

    assert settings.database_url.endswith("/supportsmith_test")
