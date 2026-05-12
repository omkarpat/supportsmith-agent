"""Typed website source configuration loaded from ``data/websites/*.yaml``.

A site config carries everything the ingestion pipeline needs to crawl a site
deterministically: name, base URL, crawl tunables, include/exclude rules, and
priority paths. The config is not Knotch-specific — Knotch is just one file in
``data/websites/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_DIR = Path("data/websites")
DEFAULT_LIMIT = 100
DEFAULT_INCLUDE: tuple[str, ...] = (r"^/.*",)


class CrawlConfig(BaseModel):
    """Per-site crawl knobs surfaced to the operator + the Firecrawl wrapper."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=DEFAULT_LIMIT, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    allow_subdomains: bool = False
    allow_external_links: bool = False
    ignore_query_parameters: bool = True
    only_main_content: bool = True


class WebsiteSourceConfig(BaseModel):
    """One site's ingestion configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    base_url: str
    source: Literal["website"] = "website"
    description: str | None = None
    include_paths: tuple[str, ...] = DEFAULT_INCLUDE
    priority_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = ()
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must be an http or https URL")
        if not parsed.netloc:
            raise ValueError("base_url must include a hostname")
        return value

    @model_validator(mode="after")
    def _validate_regexes(self) -> WebsiteSourceConfig:
        for label, patterns in (
            ("include_paths", self.include_paths),
            ("priority_paths", self.priority_paths),
            ("exclude_paths", self.exclude_paths),
        ):
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"{label} contains invalid regex {pattern!r}: {exc}") from exc
        return self

    @property
    def hostname(self) -> str:
        """Lowercase hostname of ``base_url`` for allowlist checks."""
        host = urlparse(self.base_url).netloc.lower()
        return host.split("@")[-1]

    def path_priority(self, path: str) -> bool:
        """True when ``path`` matches one of the configured priority regexes."""
        return any(re.match(pattern, path) for pattern in self.priority_paths)


def load_website_source(name: str, *, base_dir: Path = DEFAULT_CONFIG_DIR) -> WebsiteSourceConfig:
    """Load and validate ``data/websites/<name>.yaml``."""
    path = base_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"website source config not found: {path}")
    return load_website_source_file(path)


def load_website_source_file(path: Path) -> WebsiteSourceConfig:
    """Load and validate a website source from an explicit path."""
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"website source {path} must be a YAML mapping")
    return WebsiteSourceConfig.model_validate(raw)


def list_website_sources(*, base_dir: Path = DEFAULT_CONFIG_DIR) -> list[str]:
    """Return the names of every ``*.yaml`` site config in ``base_dir``."""
    if not base_dir.exists():
        return []
    return sorted(path.stem for path in base_dir.glob("*.yaml"))
