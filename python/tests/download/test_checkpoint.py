import json
from pathlib import Path

import pytest

from easycat.download.checkpoint import (
    CheckpointStore,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
)


def test_default_status_is_pending(tmp_path):
    cp = CheckpointStore(tmp_path / "cp.json")
    assert cp.status("a") == STATUS_PENDING
    assert cp.is_done("a") is False


def test_mark_done_and_failed(tmp_path):
    cp = CheckpointStore(tmp_path / "cp.json")
    cp.mark_done("a")
    cp.mark_failed("b", "boom")
    assert cp.status("a") == STATUS_DONE
    assert cp.status("b") == STATUS_FAILED
    assert cp.pending(["a", "b", "c"]) == ["b", "c"]
    assert cp.failed(["a", "b", "c"]) == ["b"]


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "cp.json"
    cp = CheckpointStore(path)
    cp.mark_done("a")
    cp.mark_failed("b", "err")
    cp.save()

    cp2 = CheckpointStore(path)
    assert cp2.status("a") == STATUS_DONE
    assert cp2.status("b") == STATUS_FAILED
    assert cp2.items["b"]["error"] == "err"


def test_atomic_save_creates_no_temp(tmp_path):
    path = tmp_path / "cp.json"
    cp = CheckpointStore(path)
    cp.mark_done("x")
    cp.save()
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text(json.dumps({"version": 999, "items": {}}))
    with pytest.raises(ValueError):
        CheckpointStore(path)


def test_mark_pending_resets(tmp_path):
    cp = CheckpointStore(tmp_path / "cp.json")
    cp.mark_done("a")
    cp.mark_pending("a")
    assert cp.status("a") == STATUS_PENDING
