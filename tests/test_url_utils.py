"""URL normalization + SSRF guard tests."""

from __future__ import annotations

import socket

import pytest

from app.retrieval import url_utils


def test_normalize_strips_fragment_and_lowercases_host() -> None:
    normalized = url_utils.normalize_url("HTTPS://Knotch.COM/Case-Studies/Acme/#hero")
    assert normalized == "https://knotch.com/Case-Studies/Acme/"


def test_normalize_strips_tracking_params() -> None:
    normalized = url_utils.normalize_url(
        "https://knotch.com/blog/post?utm_source=x&id=42&utm_campaign=y"
    )
    # tracking params dropped; non-tracking params preserved
    assert "utm_source" not in normalized
    assert "utm_campaign" not in normalized
    assert "id=42" in normalized


def test_normalize_rejects_unsupported_scheme() -> None:
    with pytest.raises(url_utils.UnsafeUrlError):
        url_utils.normalize_url("ftp://example.com/")


def test_url_hash_is_stable() -> None:
    a = url_utils.url_hash("https://knotch.com/blog/post-1")
    b = url_utils.url_hash("HTTPS://knotch.com/blog/post-1#")
    assert a == b


def test_build_website_external_id_format() -> None:
    eid = url_utils.build_website_external_id(
        site_name="knotch", url="https://knotch.com/", chunk_index=3
    )
    assert eid.startswith("website:knotch:")
    assert eid.endswith(":003")


def test_validate_public_url_rejects_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make sure DNS for the bogus host resolves to a loopback address.
    def fake_getaddrinfo(host: str, *_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    with pytest.raises(url_utils.UnsafeUrlError):
        url_utils.validate_public_url(
            "https://internal.example/",
            allowed_hosts=("internal.example",),
        )


def test_validate_public_url_rejects_private_ip_literal() -> None:
    with pytest.raises(url_utils.UnsafeUrlError):
        url_utils.validate_public_url(
            "https://10.0.0.1/",
            allowed_hosts=(),
            allow_any=True,
        )


def test_validate_public_url_requires_allowlist_when_not_allow_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0))],
    )
    with pytest.raises(url_utils.UnsafeUrlError, match="allowlist"):
        url_utils.validate_public_url("https://example.com/", allowed_hosts=())


def test_validate_public_url_allows_subdomain_of_listed_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0))],
    )
    normalized = url_utils.validate_public_url(
        "https://blog.knotch.com/post-1",
        allowed_hosts=("knotch.com",),
    )
    assert normalized.startswith("https://blog.knotch.com")
