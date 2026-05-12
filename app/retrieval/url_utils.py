"""URL normalization + ingestion-safety helpers used by website ingestion.

Two responsibilities:

1. **Normalization** — strip fragments and tracking params, lowercase host,
   keep trailing-slash policy stable so the same page produces the same
   ``external_id`` across recrawls.
2. **SSRF guard** — reject loopback, link-local, and RFC1918 hosts when an
   operator submits a URL through the admin ingestion API.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PREFIXES: tuple[str, ...] = ("utm_", "mc_", "fbclid", "gclid", "yclid", "ref", "ref_")
_BLOCKED_HOSTS: frozenset[str] = frozenset({"localhost", "metadata.google.internal"})


class UnsafeUrlError(ValueError):
    """Raised when a URL fails SSRF / scheme / host validation."""


def normalize_url(raw: str, *, strip_query_params: bool = True) -> str:
    """Return a canonical form of ``raw`` for stable external ids and dedup.

    - Drops the URL fragment.
    - Lowercases the scheme and host.
    - Optionally strips tracking-style query parameters.
    - Preserves the trailing slash policy of the input path.
    """
    parsed = urlparse(raw.strip())
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError(f"Unsupported scheme: {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    query = parsed.query
    if strip_query_params:
        kept_params = [
            (key, value)
            for key, value in parse_qsl(query, keep_blank_values=False)
            if not _is_tracking_param(key)
        ]
        query = urlencode(kept_params)
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


def url_hash(url: str) -> str:
    """Stable 16-hex digest of the normalized URL for external-id construction."""
    digest = hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()
    return digest[:16]


def build_website_external_id(*, site_name: str, url: str, chunk_index: int) -> str:
    """Construct the external_id for a website chunk per the Phase 7 spec."""
    return f"website:{site_name}:{url_hash(url)}:{chunk_index:03d}"


def validate_public_url(
    raw: str,
    *,
    allowed_hosts: tuple[str, ...] | None = None,
    allow_any: bool = False,
) -> str:
    """Return the normalized URL after rejecting unsafe or non-allowlisted hosts.

    ``allowed_hosts`` are lowercase hostnames; an exact match or a subdomain of
    a listed host is accepted. When ``allow_any`` is True the host check is
    skipped, but SSRF rejection still applies.
    """
    normalized = normalize_url(raw, strip_query_params=False)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host or host in _BLOCKED_HOSTS:
        raise UnsafeUrlError(f"Host {host!r} is not permitted for ingestion")
    # Allowlist runs before DNS so a typo on a disallowed host surfaces the
    # right error (not "could not resolve") and we don't spend a DNS lookup
    # on something we'd have rejected anyway.
    if not allow_any:
        if not allowed_hosts:
            raise UnsafeUrlError(
                "No ingestion allowlist configured and arbitrary ingestion is disabled"
            )
        if not _matches_allowlist(host, allowed_hosts):
            raise UnsafeUrlError(f"Host {host!r} is not in the ingestion allowlist")
    if _is_loopback_or_private(host):
        raise UnsafeUrlError(f"Host {host!r} resolves to a loopback or private address")
    return normalized


def _matches_allowlist(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    for allowed in allowed_hosts:
        allowed = allowed.lower()
        if host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


def _is_tracking_param(key: str) -> bool:
    return any(key.lower().startswith(prefix) for prefix in TRACKING_PREFIXES)


def _is_loopback_or_private(host: str) -> bool:
    # Direct IP literals — easy reject.
    try:
        ip = ipaddress.ip_address(host)
        return _ip_is_private(ip)
    except ValueError:
        pass
    # Hostname: resolve and check every returned address. ``getaddrinfo`` may
    # raise ``socket.gaierror`` for an unknown host; in that case we fail
    # closed since we can't prove the host is safe.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _ip_is_private(ip):
            return True
    return False


def _ip_is_private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )
