"""Run Alembic migrations for SupportSmith."""

import argparse
from pathlib import Path

from alembic.config import Config

from alembic import command
from app.core.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def build_alembic_config(database_url: str) -> Config:
    """Build an Alembic config with the runtime database URL injected."""
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade(database_url: str, revision: str = "head") -> None:
    """Apply Alembic migrations up to the requested revision."""
    command.upgrade(build_alembic_config(database_url), revision)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Apply SupportSmith database migrations.")
    parser.add_argument(
        "revision",
        nargs="?",
        default="head",
        help="Alembic revision target. Defaults to head.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    settings = get_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL or SUPPORTSMITH_DATABASE_URL is required.")
    upgrade(settings.database_url, revision=args.revision)
    print(f"Applied Alembic migrations through {args.revision}.")


if __name__ == "__main__":
    main()
