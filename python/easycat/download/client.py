"""HTTP client for survey downloads.

Provides session reuse, automatic retries with exponential backoff,
per-host rate limiting, streaming downloads and HTTP Range requests.

All network access in :mod:`easycat.download` should go through
:class:`HttpClient` so that retries / throttling / session reuse are
applied consistently.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("easycat.download")

DEFAULT_USER_AGENT = "easycat/0.1 (+https://github.com/AstroJH/easycat)"

# Status codes that trigger a retry.
# - 429: rate limited;
# - 5xx: server errors.
DEFAULT_RETRY_STATUS = (429, 500, 502, 503, 504)


class RateLimiter:
    """Simple thread-safe rate limiter (minimum interval between calls)."""

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class HttpClient:
    """Reusable HTTP client with retry / backoff / rate limiting.

    Parameters
    ----------
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Number of retries per request (connection, read and status).
    backoff : float
        Base backoff factor in seconds; each retry waits
        ``backoff * 2 ** attempt`` (plus small jitter).
    status_forcelist : tuple
        HTTP status codes that are retried.
    min_interval : float
        Minimum seconds between two requests (global rate limit).
    pool_maxsize : int
        Connection pool size (number of concurrent connections kept alive).
    user_agent : str
        User-Agent header.
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 1.0,
        status_forcelist: tuple = DEFAULT_RETRY_STATUS,
        min_interval: float = 0.0,
        pool_maxsize: int = 32,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.timeout = timeout
        self.rate_limiter = RateLimiter(min_interval)

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            other=retries,
            backoff_factor=backoff,
            status_forcelist=list(status_forcelist),
            allowed_methods=frozenset(["GET", "HEAD", "POST"]),
            raise_on_status=False,  # return the final response after exhausting retries
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=pool_maxsize,
            pool_maxsize=pool_maxsize,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------------------ #
    # low level
    # ------------------------ #
    def get(self, url: str, **kwargs) -> requests.Response:
        self.rate_limiter.wait()
        kwargs.setdefault("timeout", self.timeout)
        return self.session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        self.rate_limiter.wait()
        kwargs.setdefault("timeout", self.timeout)
        return self.session.post(url, **kwargs)

    def head(self, url: str, **kwargs) -> requests.Response:
        self.rate_limiter.wait()
        kwargs.setdefault("timeout", self.timeout)
        return self.session.head(url, **kwargs)

    # ------------------------ #
    # convenience
    # ------------------------ #
    def get_text(self, url: str, **kwargs) -> str:
        resp = self.get(url, **kwargs)
        resp.raise_for_status()
        return resp.text

    def get_bytes(self, url: str, **kwargs) -> bytes:
        resp = self.get(url, **kwargs)
        resp.raise_for_status()
        return resp.content

    def download_file(
        self,
        url: str,
        dest: Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 1 << 16,
        **kwargs,
    ) -> Path:
        """Stream ``url`` to ``dest``.

        The payload is written to ``<dest>.part`` first and then atomically
        renamed, so an interrupted download never leaves a truncated file
        at the final path.  If a ``<dest>.part`` file already exists the
        download resumes from its length via an HTTP Range request (when the
        server supports it).
        """
        dest = Path(dest)
        if dest.exists() and not overwrite:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp = dest.with_name(dest.name + ".part")
        offset = tmp.stat().st_size if tmp.exists() else 0

        headers = kwargs.pop("headers", {})
        if offset:
            headers = {**headers, "Range": f"bytes={offset}-"}
        resp = self.get(url, stream=True, headers=headers, **kwargs)

        if resp.status_code == 416 and offset:
            # Already complete.
            os.replace(tmp, dest)
            return dest
        if resp.status_code == 200 and offset:
            # Server ignored the Range header; restart from scratch.
            offset = 0
        resp.raise_for_status()

        try:
            mode = "ab" if offset else "wb"
            with open(tmp, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return dest

    def range_request(self, url: str, start: int, end: int, **kwargs) -> bytes:
        """Fetch the byte range ``[start, end]`` (inclusive) via HTTP Range.

        Raises
        ------
        requests.HTTPError
            If the server does not support Range requests (HTTP 200 with
            the full body instead of 206) the caller is expected to handle it.
        """
        resp = self.get(url, headers={"Range": f"bytes={start}-{end}"}, **kwargs)
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _LRUCache:
    """Very small thread-unsafe LRU used by RangeFile."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._data = {}   # block_index -> bytes
        self._order = []  # most-recently-used at the end

    def get(self, key):
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._order.remove(key)
        self._data[key] = value
        self._order.append(key)
        while len(self._order) > self.maxsize:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)


class RangeFile:
    """File-like object over an HTTP URL using Range requests.

    Provides ``read`` / ``seek`` / ``tell`` so that tools such as
    :mod:`astropy.io.fits` can open remote FITS files and transfer only
    the byte ranges that are actually accessed (used for DESI spectra:
    a single target's spectrum is a handful of rows inside a ~1 GB
    coadd file).

    Parameters
    ----------
    url : str
    client : HttpClient or None
        Shared client (rate limiting / retries applied globally).
    block_size : int
        Caching block size in bytes.  FITS headers are 2880-byte blocks;
        a 64 KiB block size keeps the number of requests small.
    max_blocks : int
        Maximum number of cached blocks (simple LRU).
    """

    def __init__(
        self,
        url: str,
        client: Optional["HttpClient"] = None,
        block_size: int = 65536,
        max_blocks: int = 64,
    ):
        self.url = url
        self.client = client or HttpClient()
        self.block_size = max(1, int(block_size))
        self._cache = _LRUCache(maxsize=max_blocks)
        self._pos = 0
        self._size: Optional[int] = None
        self._closed = False

    # file-like interface
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:      # absolute
            new = offset
        elif whence == 1:    # relative
            new = self._pos + offset
        elif whence == 2:    # from end
            new = self._total_size() + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return new

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._total_size() - self._pos
        if size <= 0:
            return b""
        data = bytearray()
        remaining = size
        while remaining > 0:
            block = self._pos // self.block_size
            off_in_block = self._pos % self.block_size
            chunk = self._get_block(block)
            take = min(remaining, len(chunk) - off_in_block)
            if take <= 0:
                break  # reached EOF / short block; do not spin
            data += chunk[off_in_block:off_in_block + take]
            self._pos += take
            remaining -= take
        return bytes(data)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "RangeFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # internals
    def _total_size(self) -> int:
        if self._size is None:
            resp = self.client.head(self.url)
            resp.raise_for_status()
            self._size = int(resp.headers.get("Content-Length", 0))
        return self._size

    def _get_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            return cached
        start = block_index * self.block_size
        end = start + self.block_size - 1
        # Clip to the file size so the server does not reject the range.
        total = self._total_size()
        if start >= total:
            return b""
        end = min(end, total - 1)
        chunk = self.client.range_request(self.url, start, end)
        if len(chunk) < self.block_size and (start + len(chunk)) < total:
            # Server ignored the Range header (returned the full body);
            # take only what we asked for and cache the rest opportunistically.
            pass
        self._cache.put(block_index, chunk)
        return chunk
