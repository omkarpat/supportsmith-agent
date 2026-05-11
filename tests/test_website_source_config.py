"""Validate the website source config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval.sources.websites import (
    DEFAULT_CONFIG_DIR,
    WebsiteSourceConfig,
    list_website_sources,
    load_website_source,
    load_website_source_file,
)


def test_loads_knotch_default_config() -> None:
    config = load_website_source("knotch")
    assert config.name == "knotch"
    assert config.base_url.startswith("https://knotch.com")
    assert config.source == "website"
    assert config.crawl.limit == 500
    assert "^/customers" in config.priority_paths
    assert "^/wp-admin" in config.exclude_paths


def test_lists_configured_sites() -> None:
    sites = list_website_sources()
    assert "knotch" in sites


def test_rejects_invalid_scheme(tmp_path: Path) -> None:
    raw = tmp_path / "bad.yaml"
    raw.write_text("name: bad\nbase_url: ftp://example.com/\n")
    with pytest.raises(ValueError, match="http or https"):
        load_website_source_file(raw)


def test_rejects_missing_host(tmp_path: Path) -> None:
    raw = tmp_path / "bad.yaml"
    raw.write_text("name: bad\nbase_url: https:///path\n")
    with pytest.raises(ValueError, match="hostname"):
        load_website_source_file(raw)


def test_rejects_invalid_regex(tmp_path: Path) -> None:
    raw = tmp_path / "bad.yaml"
    raw.write_text(
        "name: bad\nbase_url: https://example.com/\n"
        "include_paths:\n  - \"[unbalanced\"\n"
    )
    with pytest.raises(ValueError, match="invalid regex"):
        load_website_source_file(raw)


def test_path_priority_matches_configured_regex() -> None:
    config = WebsiteSourceConfig(
        name="t",
        base_url="https://example.com/",
        priority_paths=(r"^/case-studies",),
    )
    assert config.path_priority("/case-studies/acme")
    assert not config.path_priority("/blog/post-1")


def test_hostname_lowercases() -> None:
    config = WebsiteSourceConfig(name="t", base_url="https://EXAMPLE.com/")
    assert config.hostname == "example.com"


def test_default_dir_constant() -> None:
    assert Path("data/websites") == DEFAULT_CONFIG_DIR
