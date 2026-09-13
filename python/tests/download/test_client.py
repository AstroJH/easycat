"""Offline tests for HttpClient / RangeFile using a local HTTP server."""
from __future__ import annotations

import functools
import gzip
import hashlib
import http.server
import re
import socketserver
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

from easycat.download import (
    ChecksumMismatch,
    HttpClient,
    HttpError,
    IncompleteDownload,
    RangeFile,
    RangeNotSupported,
    VerificationFailed,
    download_urls,
    reset_default_client,
)


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Small HTTP server with explicit Range, error and truncation endpoints."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/error":
            body = b"VOTable error: invalid query"
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
            return

        file_path = Path(self.translate_path(parsed.path))
        if parsed.path == "/truncated":
            file_path = Path(self.directory) / "data.bin"

        if not file_path.is_file():
            self.send_error(404)
            return

        payload = file_path.read_bytes()
        range_header = self.headers.get("Range")
        self.server.requests.append({  # type: ignore[attr-defined]
            "path": parsed.path,
            "query": parsed.query,
            "range": range_header,
        })

        if parsed.path == "/truncated":
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            try:
                self.wfile.write(payload[:100])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return

        # `?norange=1` deliberately acts like a server that ignores Range.
        if range_header and parsed.query != "norange=1":
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if match is None:
                self.send_error(400, "invalid Range header")
                return

            start = int(match.group(1))
            requested_end = int(match.group(2)) if match.group(2) else None
            total = len(payload)
            if start >= total:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            end = total - 1 if requested_end is None else min(requested_end, total - 1)
            chunk = payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture(autouse=True)
def _shared_client_isolation():
    reset_default_client()
    yield
    reset_default_client()


@pytest.fixture()
def server(tmp_path):
    payload = (b"0123456789abcdef" * (300 * 1024 // 16))[:300 * 1024]
    (tmp_path / "data.bin").write_bytes(payload)
    with gzip.open(tmp_path / "data.bin.gz", "wb") as f:
        f.write(payload)

    handler = functools.partial(_Handler, directory=str(tmp_path))
    httpd = _ThreadingServer(("127.0.0.1", 0), handler)
    httpd.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, payload, httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_download_file_progress_and_identity_encoding(server, tmp_path):
    base, payload, _ = server
    client = HttpClient(timeout=10, retries=1)
    updates = []

    dest = client.download_file(
        f"{base}/data.bin", tmp_path / "out.bin", chunk_size=4096,
        progress=lambda done, total: updates.append((done, total)),
    )

    assert dest.read_bytes() == payload
    assert updates[0] == (0, len(payload))
    assert updates[-1] == (len(payload), len(payload))
    assert all(a[0] <= b[0] for a, b in zip(updates, updates[1:]))
    assert client.session.headers["Accept-Encoding"] == "identity"


def test_range_file_read_and_cache(server):
    base, payload, httpd = server
    client = HttpClient(timeout=10, retries=1)

    with RangeFile(f"{base}/data.bin", client=client,
                   block_size=4096, max_blocks=32) as rf:
        assert rf.readable() and rf.seekable()
        assert rf._total_size() == len(payload)

        rf.seek(0)
        assert rf.read(100) == payload[:100]
        rf.seek(1000)
        assert rf.read(200) == payload[1000:1200]
        rf.seek(len(payload) - 50)
        assert rf.read() == payload[-50:]
        rf.seek(-10, 2)
        assert rf.read() == payload[-10:]

        # Re-reading a cached block must not trigger another HTTP request.
        n_before = len(httpd.requests)  # type: ignore[attr-defined]
        rf.seek(0)
        assert rf.read(100) == payload[:100]
        assert len(httpd.requests) == n_before  # type: ignore[attr-defined]

    ranges = [r["range"] for r in httpd.requests]  # type: ignore[attr-defined]
    assert "bytes=0-4095" in ranges


def test_range_not_supported_does_not_return_full_body(server):
    base, _, _ = server
    client = HttpClient(timeout=10, retries=1)
    url = f"{base}/data.bin?norange=1"

    with pytest.raises(RangeNotSupported):
        client.range_request(url, 0, 99)

    with RangeFile(url, client=client, block_size=4096) as rf:
        with pytest.raises(RangeNotSupported):
            rf.read(100)


def test_download_file_resume_and_resume_modes(server, tmp_path):
    base, payload, httpd = server
    url = f"{base}/data.bin"
    client = HttpClient(timeout=10, retries=1)

    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    part.write_bytes(payload[:1000])
    assert client.download_file(url, dest) == dest
    assert dest.read_bytes() == payload
    assert any(r["range"] == "bytes=1000-" for r in httpd.requests)  # type: ignore[attr-defined]

    required_dest = tmp_path / "required.bin"
    required_part = tmp_path / "required.bin.part"
    required_part.write_bytes(payload[:1000])
    with pytest.raises(RangeNotSupported):
        client.download_file(
            f"{base}/data.bin?norange=1", required_dest,
            resume="required",
        )
    assert required_part.read_bytes() == payload[:1000]

    never_dest = tmp_path / "never.bin"
    never_part = tmp_path / "never.bin.part"
    never_part.write_bytes(payload[:1000])
    client.download_file(
        f"{base}/data.bin?norange=1", never_dest, resume="never",
    )
    assert never_dest.read_bytes() == payload


def test_download_file_keeps_partial_on_connection_error(server, tmp_path):
    base, payload, _ = server
    client = HttpClient(timeout=10, retries=0)
    dest = tmp_path / "truncated.bin"

    with pytest.raises(requests.RequestException):
        client.download_file(f"{base}/truncated", dest)

    part = tmp_path / "truncated.bin.part"
    assert not dest.exists()
    assert part.exists()
    # urllib3 may detect the truncated Content-Length before exposing the
    # buffered bytes; the important contract is that the resumable file stays.
    assert part.stat().st_size < len(payload)


def test_download_file_cleanup_partial_is_explicit(server, tmp_path):
    base, _, _ = server
    client = HttpClient(timeout=10, retries=0)
    dest = tmp_path / "truncated.bin"

    with pytest.raises(requests.RequestException):
        client.download_file(f"{base}/truncated", dest, cleanup_partial=True)

    assert not dest.exists()
    assert not (tmp_path / "truncated.bin.part").exists()


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers=None):
        self.body = body
        self.status_code = status
        self.headers = headers or {}
        self.url = "https://example.test/file"
        self.closed = False

    def iter_content(self, chunk_size=1):
        yield self.body

    def close(self):
        self.closed = True


def test_size_mismatch_keeps_resumable_part(tmp_path):
    client = HttpClient(timeout=10, retries=0)
    response = _FakeResponse(b"abc", headers={"Content-Length": "10"})
    client.get = lambda *args, **kwargs: response  # type: ignore[assignment]

    dest = tmp_path / "out.bin"
    with pytest.raises(IncompleteDownload) as exc:
        client.download_file("https://example.test/file", dest)

    assert exc.value.actual == 3 and exc.value.expected == 10
    assert not dest.exists()
    assert (tmp_path / "out.bin.part").read_bytes() == b"abc"


def test_416_without_remote_size_is_not_accepted(tmp_path):
    client = HttpClient(timeout=10, retries=0)
    response = _FakeResponse(b"", status=416)
    client.get = lambda *args, **kwargs: response  # type: ignore[assignment]

    dest = tmp_path / "out.bin"
    (tmp_path / "out.bin.part").write_bytes(b"abc")
    with pytest.raises(IncompleteDownload):
        client.download_file("https://example.test/file", dest)

    assert not dest.exists()
    assert (tmp_path / "out.bin.part").read_bytes() == b"abc"


def test_existing_file_is_size_checked_then_replaced(server, tmp_path):
    base, payload, httpd = server
    client = HttpClient(timeout=10, retries=1)
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"bad")

    client.download_file(f"{base}/data.bin", dest)

    assert dest.read_bytes() == payload
    assert any(r["path"] == "/data.bin" for r in httpd.requests)  # type: ignore[attr-defined]


def test_checksum_and_verify_failures_remove_bad_partial(server, tmp_path):
    base, payload, _ = server
    client = HttpClient(timeout=10, retries=1)

    dest = tmp_path / "checksum.bin"
    with pytest.raises(ChecksumMismatch):
        client.download_file(
            f"{base}/data.bin", dest,
            checksum=("sha256", hashlib.sha256(b"wrong").hexdigest()),
        )
    assert not dest.exists()
    assert not (tmp_path / "checksum.bin.part").exists()

    verify_dest = tmp_path / "verify.bin"
    with pytest.raises(VerificationFailed):
        client.download_file(
            f"{base}/data.bin", verify_dest,
            verify=lambda path, response: False,
        )
    assert not verify_dest.exists()
    assert not (tmp_path / "verify.bin.part").exists()

    ok_dest = tmp_path / "verified.bin"
    client.download_file(
        f"{base}/data.bin", ok_dest,
        verify=lambda path, response: path.read_bytes() == payload,
    )
    assert ok_dest.read_bytes() == payload


def test_gzip_validation_detects_and_replaces_corrupt_existing(server, tmp_path):
    base, payload, _ = server
    client = HttpClient(timeout=10, retries=1)
    dest = tmp_path / "out.gz"
    dest.write_bytes(b"not gzip")

    client.download_file(
        f"{base}/data.bin.gz", dest, validate="gzip",
    )

    assert gzip.open(dest, "rb").read() == payload


def test_http_error_contains_status_url_and_body(server):
    base, _, _ = server
    client = HttpClient(timeout=10, retries=0)

    with pytest.raises(HttpError) as exc:
        client.get_text(f"{base}/error")

    assert exc.value.status == 404
    assert exc.value.url.endswith("/error")
    assert "invalid query" in exc.value.body_snippet


def test_shared_client_lifecycle():
    first = HttpClient.shared(timeout=12)
    assert HttpClient.shared(timeout=99) is first
    assert first.timeout == 12

    reset_default_client()
    second = HttpClient.shared(timeout=20)
    assert second is not first
    assert second.timeout == 20


def test_download_urls_end_to_end_with_checkpoint(server, tmp_path):
    base, payload, _ = server
    checkpoint = tmp_path / "checkpoint.json"
    urls = {"a.bin": f"{base}/data.bin", "nested/b.bin": f"{base}/data.bin"}

    summary = download_urls(
        urls, tmp_path, checkpoint=checkpoint, n_workers=2,
        progress=False, client_kwargs={"timeout": 10, "retries": 1},
    )
    assert summary.completed == 2 and summary.failed == 0
    assert (tmp_path / "a.bin").read_bytes() == payload
    assert (tmp_path / "nested" / "b.bin").read_bytes() == payload

    again = download_urls(
        urls, tmp_path, checkpoint=checkpoint, n_workers=2,
        progress=False, client_kwargs={"timeout": 10, "retries": 1},
    )
    assert again.skipped == 2 and again.completed == 2


def test_download_urls_custom_destination(server, tmp_path):
    base, payload, _ = server

    summary = download_urls(
        {"a.bin": f"{base}/data.bin"}, tmp_path,
        checkpoint=tmp_path / "checkpoint.json", progress=False,
        dest_fn=lambda obj_id, default: default.parent / "custom" / default.name,
    )

    assert summary.completed == 1
    assert (tmp_path / "custom" / "a.bin").read_bytes() == payload
