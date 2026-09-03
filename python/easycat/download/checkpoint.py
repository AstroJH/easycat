"""JSON checkpoint store with atomic writes.

Tracks per-source download status so a long catalog download can be
interrupted (Ctrl+C, crash, network outage) and resumed later without
re-downloading the sources that already finished.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCHEMA_VERSION = 1

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class CheckpointStore:
    """File-backed status store.

    Parameters
    ----------
    path : str or Path or None
        Checkpoint file location. ``None`` disables persistence
        (in-memory only).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self._data: Dict = {"version": SCHEMA_VERSION, "items": {}}
        if self.path is not None and self.path.exists():
            self.load()

    # ---------------------- #
    # state
    # ---------------------- #
    @property
    def items(self) -> Dict:
        return self._data["items"]

    def status(self, obj_id: str) -> str:
        entry = self.items.get(obj_id)
        if entry is None:
            return STATUS_PENDING
        return entry.get("status", STATUS_PENDING)

    def is_done(self, obj_id: str) -> bool:
        return self.status(obj_id) == STATUS_DONE

    def mark_done(self, obj_id: str) -> None:
        self.items[str(obj_id)] = {
            "status": STATUS_DONE,
            "updated": time.time(),
        }

    def mark_failed(self, obj_id: str, error: str = "") -> None:
        self.items[str(obj_id)] = {
            "status": STATUS_FAILED,
            "error": str(error)[:500],
            "updated": time.time(),
        }

    def mark_pending(self, obj_id: str) -> None:
        self.items.pop(str(obj_id), None)

    def pending(self, obj_ids: Iterable[str]) -> List[str]:
        return [i for i in obj_ids if not self.is_done(i)]

    def failed(self, obj_ids: Iterable[str]) -> List[str]:
        return [i for i in obj_ids if self.status(i) == STATUS_FAILED]

    # ---------------------- #
    # persistence
    # ---------------------- #
    def load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        if data.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {data.get('version')!r} "
                f"(expected {SCHEMA_VERSION}); remove {self.path} and retry."
            )
        if "items" not in data:
            raise ValueError(f"Invalid checkpoint file: {self.path}")
        self._data = data

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated"] = time.time()

        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=self.path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
