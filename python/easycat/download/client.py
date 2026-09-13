"""HTTP client for survey downloads.

Provides session reuse, automatic retries with exponential backoff,
rate limiting, streaming downloads (atomic rename + resumable ``.part``
files), integrity checks, progress reporting and HTTP Range requests.

All network access in :mod:`easycat.download` should go through
:class:`HttpClient` so that retries / throttling / session reuse are
applied consistently.

Error model
-----------
``HttpClient`` methods return :class:`requests.Response` objects (the
low-level ``requests`` contract) unless ``raise_for_status=True`` is passed.
The higher-level helpers -- :meth:`~HttpClient.get_text`,
:meth:`~HttpClient.get_bytes`, :meth:`~HttpClient.download_file`,
:meth:`~HttpClient.range_request` -- always raise the structured errors
defined below, which carry ``status`` / ``url`` / ``body_snippet``.
"""
from __future__ import annotations

import gzip
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("easycat.download")

DEFAULT_USER_AGENT = "easycat/0.1 (+https://github.com/AstroJH/easycat)"

# Status codes that trigger a retry.
# - 429: rate limited;
# - 5xx: server errors.
DEFAULT_RETRY_STATUS = (429, 500, 502, 503, 504)

# Pre-compressed payloads (.fits.gz, .tar.gz, ...) must not be transparently
# decompressed by the HTTP layer, otherwise the bytes on disk differ from the
# bytes the server advertises.
DEFAULT_ACCEPT_ENCODING = "identity"

BODY_SNIPPET_CHARS = 500


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class DownloadError(Exception):
    """Base class for all easycat download errors."""


class HttpError(DownloadError):
    """An HTTP response with an error status (4xx / 5xx).

    Attributes
    ----------
    status : int or None
    url : str or None
    body_snippet : str
        First ~500 characters of the response body (useful for services that
        report details -- e.g. VOTable errors -- inside an error response).
    """

    def __init__(self, message: str, *, url: Optional[str] = None,
                 status: Optional[int] = None, body_snippet: str = ""):
        super().__init__(message)
        self.url = url
        self.status = status
        self.body_snippet = body_snippet


class RangeNotSupported(DownloadError):
    """The server ignored a Range request (HTTP 200 instead of 206)."""

    def __init__(self, message: str, *, url: Optional[str] = None,
                 status: Optional[int] = None):
        super().__init__(message)
        self.url = url
        self.status = status


class IncompleteDownload(DownloadError):
    """The transferred byte count does not match the expected size."""

    def __init__(self, message: str, *, url: Optional[str] = None,
                 expected: Optional[int] = None, actual: Optional[int] = None,
                 part_path: Optional[Path] = None):
        super().__init__(message)
        self.url = url
        self.expected = expected
        self.actual = actual
        self.part_path = part_path


class ChecksumMismatch(DownloadError):
    """A downloaded file does not match the expected checksum."""

    def __init__(self, message: str, *, url: Optional[str] = None,
                 algorithm: Optional[str] = None,
                 expected: Optional[str] = None, actual: Optional[str] = None):
        super().__init__(message)
        self.url = url
        self.algorithm = algorithm
        self.expected = expected
        self.actual = actual


class VerificationFailed(DownloadError):
    """A user-supplied ``verify=`` callback rejected the downloaded file."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Simple thread-safe rate limiter (minimum interval between calls)."""

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last: Optional[float] = None

    def wait(self) -> None:
        """Block until the next call is allowed."""
        if self.min_interval <= 0:
            return

        with self._lock:
            now = time.monotonic()

            # First call: allow immediately.
            if self._last is None:
                self._last = now
                return

            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            # Re-read the clock because sleep() is not exact.
            self._last = time.monotonic()


class _ProgressReporter:
    """Normalise ``progress=True | callable | tqdm-like`` into callbacks."""

    def __init__(self, progress, *, total: Optional[int], desc: str = ""):
        self._callable: Optional[Callable[[int, int], None]] = None
        self._bar = None
        self.total = total
        self.done = 0

        if progress is None or progress is False:
            return
        if progress is True:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=total, desc=desc, unit="B", unit_scale=True)
            self._bar.update(0)
        elif callable(progress):
            self._callable = progress
        elif hasattr(progress, "update"):
            self._bar = progress
            if getattr(self._bar, "total", None) is None and total is not None:
                try:
                    self._bar.total = total
                except Exception:
                    pass
        else:
            raise TypeError(
                "progress must be True, a callable(done, total), or a "
                "tqdm-like object with .update()"
            )

    def start(self, initial: int = 0) -> None:
        self.done = initial
        if self._bar is not None and initial:
            self._bar.update(initial)
        if self._callable is not None:
            self._callable(self.done, self.total)

    def advance(self, n: int) -> None:
        self.done += n
        if self._bar is not None:
            self._bar.update(n)
        if self._callable is not None:
            self._callable(self.done, self.total)

    def close(self) -> None:
        if self._bar is not None and hasattr(self._bar, "close"):
            self._bar.close()


def _sha_or_md5(path: Path, algorithm: str) -> str:
    import hashlib

    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
_SHARED_LOCK = threading.Lock()
_SHARED_CLIENT: Optional["HttpClient"] = None


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
    accept_encoding : str or None
        Value of the ``Accept-Encoding`` header.  Defaults to ``"identity"``
        so pre-compressed payloads (``.fits.gz``, ``.tar.gz``, ...) are
        transferred verbatim instead of being transparently decompressed by
        ``requests``.  Pass ``None`` to leave the header untouched.
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
        accept_encoding: Optional[str] = DEFAULT_ACCEPT_ENCODING,
    ):
        self.timeout = timeout
        self.rate_limiter = RateLimiter(min_interval)

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        if accept_encoding is not None:
            self.session.headers["Accept-Encoding"] = accept_encoding

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

    # ------------------------------------------------------------------ #
    # shared / default client
    # ------------------------------------------------------------------ #
    @classmethod
    def shared(cls, **kwargs) -> "HttpClient":
        """Process-wide shared client (created on first use).

        Useful for long-lived consumer code that performs many small
        requests: the underlying :class:`requests.Session` (and its
        connection pool) is reused instead of being recreated per call.
        Extra keyword arguments are only used when the instance is created.
        """
        global _SHARED_CLIENT
        with _SHARED_LOCK:
            if _SHARED_CLIENT is None:
                _SHARED_CLIENT = cls(**kwargs)
            elif kwargs:
                logger.debug("HttpClient.shared(): ignoring kwargs, "
                             "the shared client already exists")
        return _SHARED_CLIENT

    # ------------------------------------------------------------------ #
    # low-level requests
    # ------------------------------------------------------------------ #
    def get(self, url: str, *, raise_for_status: bool = False, **kwargs) -> requests.Response:
        """HTTP GET.

        Note
        ----
        Unlike the convenience helpers, this returns the response *without*
        raising on 4xx/5xx (matching ``requests``).  Pass
        ``raise_for_status=True`` to get a structured :class:`HttpError`.
        """
        self.rate_limiter.wait()
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.get(url, **kwargs)
        if raise_for_status:
            self.raise_for_status(resp)
        return resp

    def post(self, url: str, *, raise_for_status: bool = False, **kwargs) -> requests.Response:
        """HTTP POST (see :meth:`get` for the error contract)."""
        self.rate_limiter.wait()
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.post(url, **kwargs)
        if raise_for_status:
            self.raise_for_status(resp)
        return resp

    def head(self, url: str, *, raise_for_status: bool = False, **kwargs) -> requests.Response:
        """HTTP HEAD (see :meth:`get` for the error contract)."""
        self.rate_limiter.wait()
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.head(url, **kwargs)
        if raise_for_status:
            self.raise_for_status(resp)
        return resp

    def raise_for_status(self, resp: requests.Response) -> requests.Response:
        """Raise :class:`HttpError` for 4xx/5xx, with a body snippet."""
        if 400 <= resp.status_code:
            snippet = ""
            try:
                snippet = resp.text[:BODY_SNIPPET_CHARS]
            except Exception:  # pragma: no cover - binary/odd encodings
                pass
            raise HttpError(
                f"HTTP {resp.status_code} for {resp.url}",
                url=resp.url, status=resp.status_code, body_snippet=snippet,
            )
        return resp

    # ------------------------ #
    # convenience
    # ------------------------ #
    def get_text(self, url: str, **kwargs) -> str:
        resp = self.get(url, **kwargs)
        self.raise_for_status(resp)
        return resp.text

    def get_bytes(self, url: str, **kwargs) -> bytes:
        resp = self.get(url, **kwargs)
        self.raise_for_status(resp)
        return resp.content

    # ------------------------ #
    # downloads
    # ------------------------ #
    def download_file(
        self,
        url: str,
        dest: Union[str, Path],
        *,
        overwrite: bool = False,
        chunk_size: int = 1 << 16,
        resume: str = "auto",
        verify_size: bool = True,
        trust_existing: bool = False,
        expected_size: Optional[int] = None,
        checksum: Optional[Tuple[str, str]] = None,
        validate: Optional[str] = None,
        verify: Optional[Callable[[Path, Optional[requests.Response]], bool]] = None,
        cleanup_partial: bool = False,
        progress=None,
        **kwargs,
    ) -> Path:
        """Stream ``url`` to ``dest`` (atomically).

        Parameters
        ----------
        overwrite : bool
            Re-download even if ``dest`` exists.
        resume : {"auto", "never", "required"}
            How to treat an existing ``<dest>.part`` file:

            * ``auto``   - resume with a Range request; if the server ignores
              Range, log a warning and restart from scratch;
            * ``never``  - discard any partial file and start over;
            * ``required`` - resume, and fail with
              :class:`RangeNotSupported` if the server cannot resume.
        verify_size : bool
            Compare the number of bytes written with the size advertised by
            the server (``Content-Length`` / ``Content-Range``); a mismatch
            raises :class:`IncompleteDownload` and keeps the partial file.
        trust_existing : bool
            When ``dest`` already exists and ``overwrite`` is false, return
            it immediately without checking.  By default the existing file is
            checked against ``expected_size`` / ``checksum`` / a ``HEAD``
            request, and re-downloaded if it does not match.
        expected_size : int or None
            Known size of the remote file (bytes).
        checksum : (algorithm, hexdigest) or None
            e.g. ``("sha256", "ab12...")``; verified before the atomic rename.
        validate : {"gzip", "fits"} or None
            Cheap structural check of the downloaded file (before rename).
        verify : callable(path, response) -> bool or None
            Custom verification of the downloaded file; returning false
            raises :class:`VerificationFailed`.
        cleanup_partial : bool
            Delete ``<dest>.part`` when the download fails.  Defaults to
            ``False`` so partial data survives transient failures / Ctrl+C
            and can be resumed (SIGKILL, ConnectionError, ...).
        progress : True, callable(done, total), or tqdm-like
            Progress reporting.  ``done``/``total`` are byte counts
            (``total`` is ``None`` when the server does not advertise a size).

        Returns
        -------
        Path
            ``dest`` (the file exists and has been verified on success).

        Notes
        -----
        The payload is streamed to ``<dest>.part`` and only renamed onto
        ``dest`` after all checks pass, so ``dest`` is never truncated or
        partial.
        """
        if resume not in ("auto", "never", "required"):
            raise ValueError(
                f"resume must be 'auto', 'never' or 'required', got {resume!r}"
            )

        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".part")

        # ---- already there? -------------------------------------------- #
        if dest.exists() and not overwrite:
            if trust_existing or self._existing_ok(
                url, dest, expected_size=expected_size, checksum=checksum,
                validate=validate, verify=verify,
            ):
                return dest
            logger.warning("%s exists but failed verification; re-downloading",
                           dest)

        dest.parent.mkdir(parents=True, exist_ok=True)

        if resume == "never" and tmp.exists():
            tmp.unlink()

        offset = tmp.stat().st_size if tmp.exists() else 0

        headers = dict(kwargs.pop("headers", None) or {})
        if offset:
            headers["Range"] = f"bytes={offset}-"

        resp = self.get(url, stream=True, headers=headers, **kwargs)

        if resp.status_code == 416 and offset:
            # The partial file is already as long as the remote file.
            try:
                total = expected_size
                content_range = resp.headers.get("Content-Range", "")
                if "/" in content_range:
                    tail = content_range.rsplit("/", 1)[-1].strip()
                    if tail.isdigit():
                        total = int(tail)

                # Without a remote size (or another verifier) a 416 response
                # cannot prove that the local part is complete.
                if (verify_size and total is None and checksum is None
                        and validate is None and verify is None):
                    raise IncompleteDownload(
                        f"{url}: server returned HTTP 416 but did not report "
                        f"the remote size; cannot verify {tmp}",
                        url=url, actual=offset, part_path=tmp,
                    )

                self._finalize(tmp, dest, url=url, offset=offset,
                               response=resp, verify_size=verify_size,
                               expected_size=total, checksum=checksum,
                               validate=validate, verify=verify)
            finally:
                resp.close()
            return dest

        if resp.status_code == 200 and offset:
            if resume == "required":
                resp.close()
                raise RangeNotSupported(
                    f"server ignored the Range request for {url}; "
                    f"cannot resume from {offset} bytes (resume='required')",
                    url=url, status=resp.status_code,
                )
            logger.warning(
                "server ignored the Range request for %s (HTTP 200); "
                "restarting the download from scratch", url,
            )
            offset = 0

        try:
            self.raise_for_status(resp)
        except BaseException:
            resp.close()
            raise

        total = self._expected_total(resp, offset, expected_size)

        reporter = _ProgressReporter(progress, total=total,
                                     desc=dest.name)
        reporter.start(offset)
        written = offset
        try:
            mode = "ab" if offset else "wb"
            with open(tmp, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    reporter.advance(len(chunk))
        except BaseException:
            reporter.close()
            if cleanup_partial and tmp.exists():
                tmp.unlink()
                written = 0
            logger.info(
                "%s: download interrupted after %d bytes%s", url, written,
                "" if cleanup_partial else f" (kept {tmp.name} for resume)",
            )
            raise
        finally:
            try:
                resp.close()
            except Exception:  # pragma: no cover
                pass

        reporter.close()

        self._finalize(tmp, dest, url=url, offset=0, response=resp,
                       verify_size=verify_size, expected_size=total,
                       checksum=checksum, validate=validate, verify=verify)
        return dest

    # -- internals ------------------------------------------------------- #
    @staticmethod
    def _expected_total(resp: requests.Response, offset: int,
                        expected_size: Optional[int]) -> Optional[int]:
        """Total size of the remote file (None when unknown)."""
        if resp.status_code == 206:
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[-1].strip()
                if tail.isdigit():
                    return int(tail)
        length = resp.headers.get("Content-Length")
        if length is not None and length.isdigit():
            return offset + int(length)
        return expected_size

    def _existing_ok(
        self,
        url: str,
        dest: Path,
        *,
        expected_size: Optional[int],
        checksum: Optional[Tuple[str, str]],
        validate: Optional[str],
        verify: Optional[Callable[[Path, Optional[requests.Response]], bool]],
    ) -> bool:
        """Best-effort verification of an already present file."""
        if checksum is not None:
            try:
                if _sha_or_md5(dest, checksum[0]) != checksum[1]:
                    return False
            except Exception as exc:
                logger.debug("checksum of %s failed: %s", dest, exc)
                return False

        if verify is not None:
            try:
                if not verify(dest, None):
                    return False
            except Exception as exc:
                logger.debug("custom verification of %s failed: %s", dest, exc)
                return False

        size = expected_size
        if size is None:
            try:
                resp = self.head(url)
                self.raise_for_status(resp)
                length = resp.headers.get("Content-Length")
                if length is not None and length.isdigit():
                    size = int(length)
            except Exception as exc:
                logger.debug("cannot verify %s via HEAD: %s", dest, exc)
                if validate is None:
                    logger.warning(
                        "cannot verify the size of existing file %s; "
                        "re-downloading (use trust_existing=True to keep it)",
                        dest,
                    )
                    return False
        if size is not None and dest.stat().st_size != size:
            return False
        if validate is not None:
            return self._validate_file(dest, validate)
        return True

    @staticmethod
    def _validate_file(path: Path, kind: str) -> bool:
        try:
            if kind == "gzip":
                with gzip.open(path, "rb") as f:
                    # Consume the entire stream so gzip validates the final
                    # CRC/length trailer instead of only the first block.
                    for _ in iter(lambda: f.read(1 << 20), b""):
                        pass
                return True
            if kind == "fits":
                with open(path, "rb") as f:
                    return f.read(6) == b"SIMPLE"
            raise ValueError(f"unknown validate= value: {kind!r}")
        except Exception as exc:
            logger.debug("validation %s of %s failed: %s", kind, path, exc)
            return False

    def _finalize(
        self,
        tmp: Path,
        dest: Path,
        *,
        url: str,
        offset: int,
        response: Optional[requests.Response],
        verify_size: bool,
        expected_size: Optional[int],
        checksum: Optional[Tuple[str, str]],
        validate: Optional[str],
        verify: Optional[Callable[[Path, Optional[requests.Response]], bool]],
    ) -> None:
        """Verify the partial file, then atomically move it onto ``dest``."""
        actual = tmp.stat().st_size if tmp.exists() else 0

        if verify_size and expected_size is not None and actual != expected_size:
            raise IncompleteDownload(
                f"{url}: wrote {actual} bytes, expected {expected_size} "
                f"(partial file kept at {tmp})",
                url=url, expected=expected_size, actual=actual, part_path=tmp,
            )

        if checksum is not None:
            algorithm, expected = checksum
            actual_sum = _sha_or_md5(tmp, algorithm)
            if actual_sum != expected:
                tmp.unlink(missing_ok=True)
                raise ChecksumMismatch(
                    f"{url}: {algorithm} mismatch (expected {expected}, "
                    f"got {actual_sum}); invalid partial file was removed",
                    url=url, algorithm=algorithm, expected=expected,
                    actual=actual_sum,
                )

        if validate is not None and not self._validate_file(tmp, validate):
            tmp.unlink(missing_ok=True)
            raise VerificationFailed(
                f"{url}: validation {validate!r} failed; invalid partial "
                f"file {tmp} was removed"
            )

        if verify is not None:
            try:
                accepted = verify(tmp, response)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                raise VerificationFailed(
                    f"{url}: custom verify() raised {exc!r}; invalid partial "
                    f"file was removed"
                ) from exc
            if not accepted:
                tmp.unlink(missing_ok=True)
                raise VerificationFailed(
                    f"{url}: custom verify() rejected the downloaded file; "
                    "invalid partial file was removed"
                )

        os.replace(tmp, dest)

    def range_request(
        self,
        url: str,
        start: int,
        end: int,
        *,
        allow_full: bool = False,
        **kwargs,
    ) -> bytes:
        """Fetch the byte range ``[start, end]`` (inclusive) via HTTP Range.

        Parameters
        ----------
        allow_full : bool
            Some servers ignore the ``Range`` header and answer with HTTP 200
            plus the *whole* file.  By default this raises
            :class:`RangeNotSupported` (protecting the caller from pulling a
            multi-GB body into memory).  With ``allow_full=True`` the
            requested slice of that body is returned instead.

        Raises
        ------
        RangeNotSupported
            The server answered 200 (or anything other than 206).
        """
        resp = self.get(
            url, headers={"Range": f"bytes={start}-{end}"}, stream=True,
            **kwargs,
        )
        try:
            self.raise_for_status(resp)

            if resp.status_code == 206:
                return b"".join(resp.iter_content(chunk_size=1 << 16))

            if resp.status_code == 200:
                if not allow_full:
                    # Do not touch the body: a multi-GB response would
                    # otherwise be transferred into memory before the error.
                    raise RangeNotSupported(
                        f"server ignored the Range request for {url} "
                        f"(HTTP 200 with the full body)",
                        url=url, status=200,
                    )

                # Explicit opt-in: stream the whole response but retain only
                # the requested slice, keeping memory use bounded by the
                # requested range instead of the full file.
                data = bytearray()
                cursor = 0
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    chunk_end = cursor + len(chunk)
                    lo = max(start, cursor)
                    hi = min(end + 1, chunk_end)
                    if lo < hi:
                        data.extend(chunk[lo - cursor:hi - cursor])
                    cursor = chunk_end
                    if cursor > end:
                        break
                return bytes(data)

            raise RangeNotSupported(
                f"unexpected status {resp.status_code} for a Range request "
                f"to {url}",
                url=url, status=resp.status_code,
            )
        finally:
            resp.close()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def get_default_client(**kwargs) -> HttpClient:
    """Return (creating if needed) the process-wide shared client."""
    return HttpClient.shared(**kwargs)


def reset_default_client() -> None:
    """Drop the shared client (mainly for tests)."""
    global _SHARED_CLIENT
    with _SHARED_LOCK:
        if _SHARED_CLIENT is not None:
            _SHARED_CLIENT.close()
        _SHARED_CLIENT = None


# --------------------------------------------------------------------------- #
# RangeFile
# --------------------------------------------------------------------------- #
class _LRUCache:
    """Small thread-safe LRU cache used by :class:`RangeFile`."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._data = {}   # block_index -> bytes
        self._order = []  # most-recently-used at the end
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._order.remove(key)
            self._order.append(key)
            return self._data[key]

    def put(self, key, value):
        with self._lock:
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
    :mod:`astropy.io.fits` can open remote FITS files and transfer only the
    byte ranges that are actually accessed (used for DESI spectra:
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
        Maximum number of cached blocks (LRU, thread-safe).

    Notes
    -----
    One ``RangeFile`` keeps a read position (``seek``/``tell``) and is meant
    to be used by a **single thread**; the block cache itself is locked, so
    separate ``RangeFile`` instances can share a client safely.  Servers that
    do not support Range requests raise :class:`RangeNotSupported` instead of
    quietly transferring the whole file.
    """

    def __init__(
        self,
        url: str,
        client: Optional["HttpClient"] = None,
        block_size: int = 65536,
        max_blocks: int = 64,
    ):
        self.url = url
        self.client = client or HttpClient.shared()
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
            self.client.raise_for_status(resp)
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
        try:
            chunk = self.client.range_request(self.url, start, end)
        except RangeNotSupported as exc:
            raise RangeNotSupported(
                f"{self.url}: cannot use RangeFile, the server does not "
                f"support HTTP Range requests",
                url=self.url, status=exc.status,
            ) from exc
        expected = end - start + 1
        if len(chunk) != expected:
            raise IncompleteDownload(
                f"{self.url}: short range response (asked {expected} bytes "
                f"from {start}, got {len(chunk)})",
                url=self.url, expected=expected, actual=len(chunk),
            )
        self._cache.put(block_index, chunk)
        return chunk
