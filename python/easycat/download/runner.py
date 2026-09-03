"""Concurrent, checkpointed batch download runner.

:class:`DownloadRunner` drives a :class:`SurveyArchive` over a catalog:

* groups pending sources into batches (one thread per batch),
* updates a JSON checkpoint as batches finish,
* shows a tqdm progress bar,
* retries failed batches,
* handles Ctrl+C gracefully and can resume from the checkpoint.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .base import FetchContext, ItemResult, SurveyArchive
from .checkpoint import CheckpointStore
from .client import HttpClient

logger = logging.getLogger("easycat.download")

@dataclass
class RunSummary:
    """Summary of a :meth:`DownloadRunner.run` call."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    interrupted: bool = False
    failed_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = " (interrupted, run again to resume)" if self.interrupted else ""
        return (
            f"total={self.total} completed={self.completed} "
            f"failed={self.failed} skipped={self.skipped}{status}"
        )


class DownloadRunner:
    """Run an archive over a catalog with concurrency and checkpointing.

    Parameters
    ----------
    archive : SurveyArchive
        Archive implementation (e.g. WISEArchive, ZTFArchive).
    catalog : pandas.DataFrame
        Catalog with id / ra / dec columns.
    store_dir : str or Path
        Output directory for per-source files.
    checkpoint : str or Path or None
        JSON checkpoint path for resume / crash recovery.
    n_workers : int
        Thread pool size (each worker processes one batch at a time).
    batch_size : int or None
        Rows per batch; defaults to ``archive.default_batch_size``.
    retry_failed : bool
        Re-run failed batches after the first pass (up to ``max_retries``).
    max_retries : int
        Extra passes over failed items.
    progress : bool
        Show a tqdm progress bar.
    save_interval : float
        Seconds between checkpoint saves.
    client_kwargs : dict or None
        Keyword arguments for :class:`HttpClient`.
    """

    def __init__(
        self,
        archive: SurveyArchive,
        catalog: pd.DataFrame,
        *,
        store_dir: Path,
        checkpoint: Optional[Path] = None,
        n_workers: int = 8,
        batch_size: Optional[int] = None,
        retry_failed: bool = True,
        max_retries: int = 1,
        progress: bool = True,
        save_interval: float = 5.0,
        id_column: str = "obj_id",
        ra_column: str = "raj2000",
        dec_column: str = "dej2000",
        radius_arcsec: float = 3.0,
        client_kwargs: Optional[dict] = None,
    ):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.archive = archive
        self.catalog = catalog
        self.store_dir = Path(store_dir)
        self.checkpoint = CheckpointStore(checkpoint)
        self.n_workers = n_workers
        self.batch_size = batch_size or getattr(archive, "default_batch_size", 1)
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.progress = progress
        self.save_interval = max(0.5, save_interval)
        self.id_column = id_column
        self.ra_column = ra_column
        self.dec_column = dec_column
        self.radius_arcsec = radius_arcsec
        self.client = HttpClient(**(client_kwargs or {}))

    def run(self) -> RunSummary:
        ids = [str(i) for i in self.catalog[self.id_column]]
        summary = RunSummary(total=len(ids))

        todo = self.checkpoint.pending(ids)
        summary.skipped = len(ids) - len(todo)

        remaining = todo
        passes = 0
        try:
            while remaining:
                passes += 1
                remaining = self._run_pass(remaining, summary, pass_number=passes)
                if not self.retry_failed or passes > self.max_retries:
                    break
        except KeyboardInterrupt:
            summary.interrupted = True
            logger.warning("Interrupted: checkpoint saved, run again to resume.")

        summary.completed = len(
            [i for i in ids if self.checkpoint.is_done(i)]
        )
        summary.failed = len(self.checkpoint.failed(ids))
        summary.failed_ids = self.checkpoint.failed(ids)
        self.checkpoint.save()

        if self.progress:
            logger.info("Summary: %s", summary)
        return summary


    def _run_pass(
        self,
        ids: Sequence[str],
        summary: RunSummary,
        pass_number: int,
    ) -> List[str]:
        """Run one pass over ``ids``; returns the ids that failed again."""
        batches = self._build_batches(ids)
        ctx = FetchContext(
            store_dir=self.store_dir,
            client=self.client,
            id_column=self.id_column,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            radius_arcsec=self.radius_arcsec,
        )

        n_done = 0
        failed_again: List[str] = []
        last_save = time.monotonic()

        pbar = None
        if self.progress:
            label = f"{self.archive.name} pass {pass_number}"
            pbar = tqdm(total=len(ids), desc=label, unit="src")

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(self.archive.fetch_batch, rows, ctx): rows
                for rows in batches
            }
            try:
                for future in as_completed(futures):
                    rows = futures[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        # A batch raised unexpectedly: mark its sources failed
                        # and keep going (they can be retried in a later pass).
                        logger.warning(
                            "Batch failed (%d sources): %s", len(rows), exc
                        )
                        results = [
                            ItemResult(obj_id=ctx.row_id(row), success=False, error=repr(exc))
                            for _, row in rows.iterrows()
                        ]

                    if len(results) != len(rows):
                        raise RuntimeError(
                            f"{self.archive.name}.fetch_batch returned "
                            f"{len(results)} results for {len(rows)} rows"
                        )

                    for (_, row), result in zip(rows.iterrows(), results):
                        obj_id = ctx.row_id(row)
                        if result.success:
                            self.checkpoint.mark_done(obj_id)
                            n_done += 1
                        else:
                            self.checkpoint.mark_failed(obj_id, result.error)
                            failed_again.append(obj_id)
                    if pbar is not None:
                        pbar.update(len(rows))

                    now = time.monotonic()
                    if now - last_save >= self.save_interval:
                        self.checkpoint.save()
                        last_save = now
            except KeyboardInterrupt:
                logger.warning("Stopping... waiting for in-flight batches.")
                for f in futures:
                    f.cancel()

                # drain running ones
                for f in as_completed(futures):
                    pass
                
                raise

        if pbar is not None:
            pbar.close()

        summary.completed += n_done
        self.checkpoint.save()
        return failed_again

    def _build_batches(self, ids: Sequence[str]) -> List[pd.DataFrame]:
        id_set = set(ids)
        rows = self.catalog[self.catalog[self.id_column].astype(str).isin(id_set)]
        batches: List[pd.DataFrame] = []
        for start in range(0, len(rows), self.batch_size):
            batches.append(rows.iloc[start:start + self.batch_size])
        return batches
