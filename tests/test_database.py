from pathlib import Path

from app.db.migrate import ALEMBIC_INI, build_alembic_config
from app.db.session import to_async_database_url, to_sync_database_url


def test_pgvector_alembic_migration_is_bundled() -> None:
    migration = Path("alembic/versions/20260507_0001_init_pgvector.py")

    assert migration.exists()
    assert "revision: str = \"20260507_0001\"" in migration.read_text()


def test_alembic_config_uses_project_config() -> None:
    config = build_alembic_config("postgresql://user:pass@localhost:55432/supportsmith")

    assert ALEMBIC_INI.name == "alembic.ini"
    assert config.get_main_option("sqlalchemy.url") == (
        "postgresql://user:pass@localhost:55432/supportsmith"
    )


def test_database_url_helpers() -> None:
    assert to_async_database_url("postgresql://u:p@localhost/db") == (
        "postgresql+psycopg://u:p@localhost/db"
    )
    assert to_sync_database_url("postgresql+psycopg://u:p@localhost/db") == (
        "postgresql://u:p@localhost/db"
    )
