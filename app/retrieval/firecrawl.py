"""Firecrawl wrapper for website ingestion.

The :class:`FirecrawlClient` Protocol decouples our pipeline from the
``firecrawl-py`` SDK so unit tests can drive ingestion off a captured JSON
fixture without an API key. The real :class:`FirecrawlSDKClient` wraps the
SDK's synchronous calls in ``asyncio.to_thread`` so the API + CLI stay async.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class FirecrawlError(RuntimeError):
    """Raised when a Firecrawl call fails. Always recoverable, never fatal."""


class FirecrawlPage(BaseModel):
    """One page fetched from Firecrawl, normalized to fields we cite from."""

    model_config = ConfigDict(extra="forbid")

    url: str
    canonical_url: str | None = None
    title: str | None = None
    description: str | None = None
    markdown: str
    html: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FirecrawlMap(BaseModel):
    """Result of a Firecrawl `map` discovery call."""

    model_config = ConfigDict(extra="forbid")

    site_url: str
    urls: list[str]


class FirecrawlCrawl(BaseModel):
    """Result of a (synchronous) Firecrawl crawl."""

    model_config = ConfigDict(extra="forbid")

    job_id: str | None
    site_url: str
    pages: list[FirecrawlPage]
    completed: bool = True


class CrawlOptions(BaseModel):
    """Tunable crawl parameters resolved from site config + request overrides."""

    model_config = ConfigDict(extra="forbid")

    limit: int = 100
    max_depth: int | None = None
    include_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = ()
    allow_subdomains: bool = False
    allow_external_links: bool = False
    ignore_query_parameters: bool = True
    only_main_content: bool = True


class FirecrawlClient(Protocol):
    """Minimal async surface tests and the ingestion pipeline depend on."""

    async def map_site(self, site_url: str, *, limit: int | None = None) -> FirecrawlMap:
        """Return a flat list of URLs discovered under ``site_url``."""

    async def crawl_site(
        self,
        site_url: str,
        *,
        options: CrawlOptions,
    ) -> FirecrawlCrawl:
        """Run a synchronous crawl and return the materialized pages."""


class FirecrawlSDKClient:
    """Default :class:`FirecrawlClient` backed by ``firecrawl-py``.

    The SDK is synchronous; we hop to a thread so callers (FastAPI background
    tasks, CLI ``asyncio.run``) stay non-blocking. Import the SDK lazily so the
    rest of the app doesn't pay an import cost when Firecrawl is unused.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise FirecrawlError("Firecrawl API key is required")
        self._api_key = api_key

    def _new_sdk(self) -> Any:
        try:
            from firecrawl import FirecrawlApp  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise FirecrawlError(
                "firecrawl-py is not installed; add it to dependencies"
            ) from exc
        return FirecrawlApp(api_key=self._api_key)

    async def map_site(self, site_url: str, *, limit: int | None = None) -> FirecrawlMap:
        def _call() -> FirecrawlMap:
            sdk = self._new_sdk()
            try:
                params: dict[str, Any] = {}
                if limit is not None:
                    params["limit"] = limit
                # firecrawl-py exposes `map_url`; the response shape across recent
                # SDK versions is either {"links": [...]} or a list-of-strings.
                raw = sdk.map_url(site_url, params=params) if params else sdk.map_url(site_url)
            except Exception as exc:  # pragma: no cover - network surface
                raise FirecrawlError(f"firecrawl map failed: {exc}") from exc
            urls = _coerce_url_list(raw)
            return FirecrawlMap(site_url=site_url, urls=urls)

        return await asyncio.to_thread(_call)

    async def crawl_site(
        self,
        site_url: str,
        *,
        options: CrawlOptions,
    ) -> FirecrawlCrawl:
        def _call() -> FirecrawlCrawl:
            sdk = self._new_sdk()
            params = _crawl_params(options)
            try:
                raw = sdk.crawl_url(site_url, params=params, poll_interval=5)
            except Exception as exc:  # pragma: no cover - network surface
                raise FirecrawlError(f"firecrawl crawl failed: {exc}") from exc
            pages = _coerce_pages(raw)
            return FirecrawlCrawl(
                job_id=_extract_job_id(raw),
                site_url=site_url,
                pages=pages,
                completed=True,
            )

        return await asyncio.to_thread(_call)


def _crawl_params(options: CrawlOptions) -> dict[str, Any]:
    """Translate our :class:`CrawlOptions` to the SDK's parameter dict."""
    scrape_options: dict[str, Any] = {
        "formats": ["markdown"],
        "onlyMainContent": options.only_main_content,
    }
    params: dict[str, Any] = {
        "limit": options.limit,
        "scrapeOptions": scrape_options,
        "allowExternalLinks": options.allow_external_links,
        "allowBackwardLinks": False,
        "ignoreQueryParameters": options.ignore_query_parameters,
    }
    if options.max_depth is not None:
        params["maxDepth"] = options.max_depth
    if options.include_paths:
        params["includePaths"] = list(options.include_paths)
    if options.exclude_paths:
        params["excludePaths"] = list(options.exclude_paths)
    if options.allow_subdomains:
        params["allowSubdomains"] = True
    return params


def _coerce_url_list(raw: Any) -> list[str]:
    """Accept either ``{"links": [...]}`` or a list-of-strings from the SDK."""
    if isinstance(raw, dict):
        for key in ("links", "urls", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return [str(item) for item in value if isinstance(item, str)]
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if isinstance(item, str)]
    return []


def _coerce_pages(raw: Any) -> list[FirecrawlPage]:
    """Normalize crawl response variants into typed :class:`FirecrawlPage`."""
    items: Sequence[Any]
    if isinstance(raw, dict):
        data = raw.get("data")
        if not isinstance(data, list):
            return []
        items = data
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    pages: list[FirecrawlPage] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        raw_metadata = entry.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        markdown = entry.get("markdown") or entry.get("content") or ""
        if not markdown.strip():
            continue
        url = (
            metadata.get("sourceURL")
            or metadata.get("url")
            or entry.get("url")
            or entry.get("sourceURL")
        )
        if not isinstance(url, str) or not url:
            continue
        pages.append(
            FirecrawlPage(
                url=url,
                canonical_url=metadata.get("canonicalURL") or metadata.get("canonical"),
                title=metadata.get("title") or metadata.get("ogTitle"),
                description=metadata.get("description") or metadata.get("ogDescription"),
                markdown=markdown,
                html=entry.get("html") if isinstance(entry.get("html"), str) else None,
                metadata=metadata,
            )
        )
    return pages


def _extract_job_id(raw: Any) -> str | None:
    if isinstance(raw, dict):
        for key in ("id", "jobId", "job_id"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                return value
    return None


class FixtureFirecrawlClient:
    """In-memory :class:`FirecrawlClient` for tests and dry runs.

    Pass a captured ``FirecrawlMap`` and ``FirecrawlCrawl`` (or raw dicts loaded
    from a JSON fixture). Useful for deterministic chunker/classifier tests.
    """

    def __init__(self, *, map_result: FirecrawlMap, crawl_result: FirecrawlCrawl) -> None:
        self._map = map_result
        self._crawl = crawl_result

    async def map_site(self, site_url: str, *, limit: int | None = None) -> FirecrawlMap:
        urls = self._map.urls
        if limit is not None:
            urls = urls[:limit]
        return FirecrawlMap(site_url=site_url, urls=urls)

    async def crawl_site(
        self,
        site_url: str,
        *,
        options: CrawlOptions,
    ) -> FirecrawlCrawl:
        pages = self._crawl.pages[: options.limit]
        return FirecrawlCrawl(
            job_id=self._crawl.job_id,
            site_url=site_url,
            pages=pages,
            completed=True,
        )
