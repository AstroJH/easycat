"""Batch execution for processing pipelines."""
from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd
from tqdm.auto import tqdm

from .core import DataPacket, Pipeline


def _result_scalar(value: Any) -> Any:
    """Convert nested result values to FITS/CSV-friendly scalars."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Recursively convert a collector result to JSON-safe data."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _packet_from_value(value: Any, row: pd.Series, id_column: str) -> DataPacket:
    if isinstance(value, DataPacket):
        packet = value
    elif isinstance(value, Mapping):
        packet = DataPacket(**dict(value))
    else:
        raise TypeError(
            "resolver must return DataPacket or a mapping of DataPacket fields"
        )
    if packet.obj_id is None:
        packet.obj_id = str(row[id_column])
    return packet


def _execute_task(
    pipeline_factory: Callable[[], Pipeline],
    resolver: Callable[[pd.Series], DataPacket | Mapping[str, Any]],
    collector: Optional[Callable[[DataPacket], Mapping[str, Any]]],
    id_column: str,
    row: pd.Series,
):
    """Execute one complete source pipeline.

    This function is deliberately top-level: joblib/loky serializes it with
    cloudpickle for process workers.  The worker owns all per-source state and
    returns only a small result tuple to the parent process.
    """
    obj_id = str(row[id_column])
    try:
        packet = _packet_from_value(resolver(row), row, id_column)
        # A fresh Pipeline is mandatory: nodes retain statuses and detector
        # instances that are not safe to share between concurrent sources.
        pipeline = pipeline_factory()
        packet = pipeline.run(packet, break_on_error=True)
        failed_nodes = pipeline.get_failed_nodes()
        if failed_nodes:
            messages = [
                outcome.errors[0]["message"]
                for outcome in packet.node_outcomes.values()
                if outcome.errors
            ]
            return obj_id, False, None, f"{failed_nodes}: {'; '.join(messages)}"
        payload = packet if collector is None else collector(packet)
        return obj_id, True, payload, ""
    except Exception as exc:  # noqa: BLE001 - batch boundary
        return obj_id, False, None, repr(exc)


@dataclass
class PipelineRunSummary:
    """Summary of one batch pipeline run."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    recovered: int = 0
    interrupted: bool = False
    failed_ids: List[str] = field(default_factory=list)
    recovered_ids: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        status = " (interrupted)" if self.interrupted else ""
        return (
            f"total={self.total} completed={self.completed} "
            f"failed={self.failed} skipped={self.skipped} "
            f"recovered={self.recovered}{status}"
        )

    def results_frame(self) -> pd.DataFrame:
        """Return collected per-source results as a DataFrame."""
        rows = []
        for obj_id, value in self.results.items():
            if isinstance(value, Mapping):
                row = {key: _result_scalar(item) for key, item in value.items()}
            else:
                row = {"value": _result_scalar(value)}
            row.setdefault("obj_id", obj_id)
            rows.append(row)
        return pd.DataFrame(rows)


class _JsonCheckpoint:
    """Small atomic JSON checkpoint for processed source ids."""

    def __init__(self, path: Optional[Path]):
        self.path = None if path is None else Path(path)
        self.data = {"version": 1, "items": {}}
        if self.path is not None and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != 1:
                raise ValueError(f"unsupported checkpoint version in {self.path}")
            self.data = payload

    def status(self, obj_id: str) -> str:
        return self.data["items"].get(str(obj_id), {}).get("status", "pending")

    def result(self, obj_id: str) -> Any:
        return self.data["items"].get(str(obj_id), {}).get("result")

    def mark(
        self,
        obj_id: str,
        status: str,
        error: str = "",
        result: Any = None,
    ) -> None:
        # Preserve older checkpoint entries and only replace mutable fields.
        entry = self.data["items"].get(str(obj_id), {})
        self.data["items"][str(obj_id)] = {
            **entry,
            "status": status,
            "error": str(error)[:1000],
            "updated": time.time(),
        }
        if status == "done" and result is not None:
            # Collector output is persisted so a later run can reconstruct
            # the summary without rerunning completed sources.
            self.data["items"][str(obj_id)]["result"] = _jsonable(result)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace prevents a Ctrl+C/interruption from truncating an
        # existing checkpoint file.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


class PipelineRunner:
    """Run a pipeline independently over every row of a catalog.

    Parameters
    ----------
    pipeline_factory : callable
        Zero-argument callable returning a fresh :class:`Pipeline`.  A fresh
        instance is required because nodes retain per-execution state.
    catalog : DataFrame
        One row per source.
    resolver : callable
        ``row -> DataPacket`` (or mapping of DataPacket fields).
    collector : callable or None
        ``packet -> mapping`` summary row used for the batch result table.
    output_path : callable or None
        Optional ``row -> Path`` used to recover completed files on disk.
    checkpoint : Path or None
        Atomic JSON checkpoint path.
    mode : {"thread", "process"}
        ``"thread"`` is convenient for I/O and shares memory.  ``"process"``
        uses joblib/loky (cloudpickle) and is preferred for CPU-heavy nodes;
        notebook-defined functions and closures are supported.
    """

    def __init__(
        self,
        pipeline_factory: Callable[[], Pipeline],
        catalog: pd.DataFrame,
        resolver: Callable[[pd.Series], DataPacket | Mapping[str, Any]],
        *,
        collector: Optional[Callable[[DataPacket], Mapping[str, Any]]] = None,
        output_path: Optional[Callable[[pd.Series], Path]] = None,
        id_column: str = "obj_id",
        checkpoint: Optional[Path] = None,
        n_workers: int = 1,
        mode: str = "thread",
        progress: bool = True,
        retry_failed: bool = True,
        max_retries: int = 1,
        save_interval: float = 5.0,
    ):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if mode not in ("thread", "process"):
            raise ValueError("mode must be 'thread' or 'process'")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.pipeline_factory = pipeline_factory
        self.catalog = catalog
        self.resolver = resolver
        self.collector = collector
        self.output_path = output_path
        self.id_column = id_column
        self.checkpoint = _JsonCheckpoint(checkpoint)
        self.n_workers = n_workers
        self.mode = mode
        self.progress = progress
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.save_interval = max(0.5, save_interval)

    def _packet_from_row(self, row: pd.Series) -> DataPacket:
        return _packet_from_value(self.resolver(row), row, self.id_column)

    def _run_one(self, row: pd.Series):
        return _execute_task(
            self.pipeline_factory,
            self.resolver,
            self.collector,
            self.id_column,
            row,
        )

    def _iter_thread_results(self, remaining, rows, pbar):
        # Thread mode completes in submission order-independent fashion and
        # is suitable for notebook closures or I/O-heavy nodes.
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(self._run_one, rows[obj_id]): obj_id
                for obj_id in remaining
            }
            for future in as_completed(futures):
                obj_id = futures[future]
                result_obj_id, success, payload, error = future.result()
                if pbar is not None:
                    pbar.update(1)
                yield result_obj_id or obj_id, success, payload, error

    def _iter_process_results(self, remaining, rows, pbar):
        # loky + cloudpickle is the key requirement for scientific
        # notebooks: local functions, closures and lambdas remain usable.
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "mode='process' requires joblib; install easycat with its "
                "scientific dependencies or add joblib>=1.3"
            ) from exc

        # Materialize tasks before the first Parallel call so the compatibility
        # fallback below can safely reuse them if an older joblib is present.
        tasks = tuple(
            delayed(_execute_task)(
                self.pipeline_factory,
                self.resolver,
                self.collector,
                self.id_column,
                rows[obj_id],
            )
            for obj_id in remaining
        )
        try:
            results = Parallel(
                n_jobs=self.n_workers,
                backend="loky",
                return_as="generator_unordered",
            )(tasks)
        except TypeError:  # joblib < 1.3 fallback
            results = Parallel(n_jobs=self.n_workers, backend="loky")(tasks)
        for result in results:
            obj_id, success, payload, error = result
            if pbar is not None:
                pbar.update(1)
            yield obj_id, success, payload, error

    def _iter_results(self, remaining, rows, pbar):
        if self.mode == "process":
            yield from self._iter_process_results(remaining, rows, pbar)
        else:
            yield from self._iter_thread_results(remaining, rows, pbar)

    def recover_from_disk(self) -> List[str]:
        """Mark sources whose declared output file already exists as done."""
        if self.output_path is None:
            return []
        recovered = []
        for _, row in self.catalog.iterrows():
            obj_id = str(row[self.id_column])
            if self.checkpoint.status(obj_id) == "done":
                continue
            try:
                path = Path(self.output_path(row))
            except Exception:
                continue
            if path.exists():
                # This is the same safety net used by DownloadRunner: the
                # output is authoritative when the checkpoint was lost.
                self.checkpoint.mark(obj_id, "done")
                recovered.append(obj_id)
        if recovered:
            self.checkpoint.save()
        return recovered

    def run(self) -> PipelineRunSummary:
        ids = [str(value) for value in self.catalog[self.id_column]]
        rows = {
            str(row[self.id_column]): row for _, row in self.catalog.iterrows()
        }
        summary = PipelineRunSummary(total=len(ids))
        summary.recovered_ids = self.recover_from_disk()
        summary.recovered = len(summary.recovered_ids)

        skipped = [
            obj_id for obj_id in ids
            if self.checkpoint.status(obj_id) == "done"
            and (self.collector is None or self.checkpoint.result(obj_id) is not None)
        ]
        summary.skipped = len(skipped)
        for obj_id in skipped:
            result = self.checkpoint.result(obj_id)
            if result is not None:
                summary.results[obj_id] = result
        # A source with status=done but no collector result is re-executed
        # when collector is configured.  This supports upgrading old
        # checkpoints that only stored success/failure.
        remaining = [obj_id for obj_id in ids if obj_id not in skipped]
        passes = 0

        try:
            while remaining:
                passes += 1
                failed_again: List[str] = []
                last_save = time.monotonic()
                pbar = tqdm(total=len(remaining), desc="pipeline sources",
                            unit="src") if self.progress else None
                try:
                    # Results are consumed as they finish so the checkpoint
                    # can be updated during long CPU-bound runs.
                    for obj_id, success, payload, error in self._iter_results(
                        remaining, rows, pbar,
                    ):
                        if success:
                            self.checkpoint.mark(
                                obj_id, "done", result=payload,
                            )
                            summary.results[obj_id] = payload
                        else:
                            self.checkpoint.mark(obj_id, "failed", error)
                            summary.errors[obj_id] = error
                            failed_again.append(obj_id)
                        if time.monotonic() - last_save >= self.save_interval:
                            self.checkpoint.save()
                            last_save = time.monotonic()
                except KeyboardInterrupt:
                    summary.interrupted = True
                    self.checkpoint.save()
                    raise
                if pbar is not None:
                    pbar.close()
                self.checkpoint.save()
                remaining = failed_again
                # Retry only failed sources; successful sources are now in
                # the checkpoint and will not be recomputed.
                if not self.retry_failed or passes > self.max_retries:
                    break
        except KeyboardInterrupt:
            summary.interrupted = True

        summary.completed = sum(
            1 for obj_id in ids if self.checkpoint.status(obj_id) == "done"
        )
        summary.failed_ids = [
            obj_id for obj_id in ids
            if self.checkpoint.status(obj_id) == "failed"
        ]
        summary.failed = len(summary.failed_ids)
        self.checkpoint.save()
        return summary


__all__ = ["PipelineRunSummary", "PipelineRunner"]
