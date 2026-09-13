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
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
    recovered: int = 0
    recovered_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = " (interrupted, run again to resume)" if self.interrupted else ""
        recovered = f" recovered={self.recovered}" if self.recovered else ""
        return (
            f"total={self.total} completed={self.completed} "
            f"failed={self.failed} skipped={self.skipped}{recovered}{status}"
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
    dest_fn : callable or None
        ``(obj_id, default_path[, row]) -> Path`` override applied to every
        per-source output file (custom on-disk layout).  Two-argument hooks
        remain supported.

    Notes
    -----
    **Threading model**: all worker threads share a single
    :class:`HttpClient` (and therefore one :class:`requests.Session` with a
    connection pool).  The session is configured once and not mutated during
    a run; this keeps rate limiting / retries / connection reuse global.  For
    per-thread proxies, credentials, certificates or cookies, implement your
    own archive/worker boundary with one ``HttpClient`` per thread.
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
        ra_column: str = "ra",
        dec_column: str = "dec",
        radius_arcsec: float = 3.0,
        client_kwargs: Optional[dict] = None,
        dest_fn: Optional[Callable[..., Path]] = None,
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
        self.dest_fn = dest_fn
        self.client = HttpClient(**(client_kwargs or {}))

    def _make_context(self) -> FetchContext:
        return FetchContext(
            store_dir=self.store_dir,
            client=self.client,
            id_column=self.id_column,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            radius_arcsec=self.radius_arcsec,
            dest_fn=self.dest_fn,
        )

    def recover_from_disk(self, *, warn: bool = True) -> List[str]:
        """Mark sources whose output file already exists as done.

        Safety net for hard interrupts (SIGKILL, kernel restart, closed
        notebook): the archives write their file *before* the runner records
        the result, so a finished download can be missing from the
        checkpoint.  Such sources are marked done (with a warning) instead of
        being downloaded again.

        Returns
        -------
        list of str
            The source ids that were recovered from files on disk.
        """
        ids = [str(i) for i in self.catalog[self.id_column]]
        rows_by_id = {
            str(row[self.id_column]): row for _, row in self.catalog.iterrows()
        }
        ctx = self._make_context()
        recovered: List[str] = []

        for obj_id in self.checkpoint.pending(ids):
            try:
                path = self.archive.resolve_output_path(
                    ctx, obj_id, row=rows_by_id.get(obj_id),
                )
            except Exception as exc:      # never break a run because of this
                logger.debug("output_path() failed for %s: %s", obj_id, exc)
                continue
            if path is not None and Path(path).exists():
                self.checkpoint.mark_done(obj_id)
                recovered.append(obj_id)
                logger.debug("recovered %s from %s", obj_id, path)

        if recovered:
            self.checkpoint.save()
            if warn:
                preview = ", ".join(recovered[:5])
                if len(recovered) > 5:
                    preview += f", ... (+{len(recovered) - 5} more)"
                logger.warning(
                    "%d source(s) already have output files on disk but were "
                    "missing from the checkpoint (e.g. after a hard "
                    "interrupt); marking them as done: %s",
                    len(recovered), preview,
                )
        return recovered

    def run(self) -> RunSummary:
        ids = [str(i) for i in self.catalog[self.id_column]]
        summary = RunSummary(total=len(ids))

        # safety net: count files that are already on disk as done
        summary.recovered_ids = self.recover_from_disk()
        summary.recovered = len(summary.recovered_ids)

        todo = self.checkpoint.pending(ids)
        summary.skipped = len(ids) - len(todo) - summary.recovered

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
        ctx = self._make_context()

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

                    n_done += self._record_batch(rows, results, ctx,
                                                 failed_again)
                    if pbar is not None:
                        pbar.update(len(rows))

                    now = time.monotonic()
                    if now - last_save >= self.save_interval:
                        self.checkpoint.save()
                        last_save = now
            except KeyboardInterrupt:
                logger.warning(
                    "Stopping... waiting for in-flight batches "
                    "(results already written to disk are recorded first)."
                )
                for f in futures:
                    f.cancel()

                # Drain running batches AND record their results: the files
                # are already on disk, so leaving them unmarked would make
                # the checkpoint disagree with reality (and would re-download
                # them on the next run).
                for f in as_completed(futures):
                    rows = futures[f]
                    if f.cancelled():
                        continue
                    try:
                        results = f.result()
                        n_done += self._record_batch(rows, results, ctx,
                                                     failed_again)
                    except Exception as exc:
                        logger.warning(
                            "Batch finished during interrupt but could not "
                            "be recorded (%d sources): %s", len(rows), exc
                        )
                    if pbar is not None:
                        pbar.update(len(rows))

                # persist immediately so an interrupted run resumes correctly
                self.checkpoint.save()
                if pbar is not None:
                    pbar.close()
                raise

        if pbar is not None:
            pbar.close()

        summary.completed += n_done
        self.checkpoint.save()
        return failed_again

    def _record_batch(
        self,
        rows: pd.DataFrame,
        results: Sequence[ItemResult],
        ctx: FetchContext,
        failed_again: List[str],
    ) -> int:
        """Mark one batch's items in the checkpoint; return #done."""
        if len(results) != len(rows):
            raise RuntimeError(
                f"{self.archive.name}.fetch_batch returned "
                f"{len(results)} results for {len(rows)} rows"
            )
        n_done = 0
        for (_, row), result in zip(rows.iterrows(), results):
            obj_id = ctx.row_id(row)
            path = self.archive.resolve_output_path(ctx, obj_id, row=row)
            self.archive.enrich_result(result, row=row, dest=path)
            if result.success:
                self.checkpoint.mark_done(obj_id)
                n_done += 1
            else:
                self.checkpoint.mark_failed(obj_id, result.error)
                failed_again.append(obj_id)
        return n_done

    def _build_batches(self, ids: Sequence[str]) -> List[pd.DataFrame]:
        id_set = set(ids)
        rows = self.catalog[self.catalog[self.id_column].astype(str).isin(id_set)]
        batches: List[pd.DataFrame] = []
        for start in range(0, len(rows), self.batch_size):
            batches.append(rows.iloc[start:start + self.batch_size])
        return batches


# --------------------------------------------------------------------------- #
# convenience: download a list of URLs (no archive required)
# --------------------------------------------------------------------------- #
class _UrlListArchive(SurveyArchive):
    """Internal archive used by :func:`download_urls`."""

    name = "urls"
    default_batch_size = 1

    def __init__(self, mapping: Dict[str, str], *, overwrite: bool = False,
                 download_kwargs: Optional[dict] = None):
        super().__init__(download_kwargs=download_kwargs)
        self.mapping = dict(mapping)
        self.overwrite = overwrite

    def _dest(self, ctx: FetchContext, obj_id: str,
              row: Optional[pd.Series] = None) -> Path:
        return ctx.dest(obj_id, ctx.store_dir / obj_id, row=row)

    def output_path(self, ctx: FetchContext, obj_id: str, *,
                    row: Optional[pd.Series] = None) -> Path:
        return self._dest(ctx, obj_id, row=row)

    def fetch_batch(self, rows, ctx):
        results = []
        for _, row in rows.iterrows():
            obj_id = ctx.row_id(row)
            dest = self._dest(ctx, obj_id, row=row)
            try:
                self._download_file(
                    ctx, self.mapping[obj_id], dest,
                    overwrite=self.overwrite,
                )
                result = ItemResult(obj_id=obj_id, success=True, data=dest)
                results.append(self.enrich_result(
                    result, row=row, url=self.mapping[obj_id], dest=dest,
                ))
            except Exception as exc:
                result = ItemResult(
                    obj_id=obj_id, success=False, error=repr(exc),
                )
                results.append(self.enrich_result(
                    result, row=row, url=self.mapping[obj_id], dest=dest,
                ))
        return results


def download_urls(
    urls: Union[Mapping[str, str], Sequence[Tuple[str, str]]],
    dest_dir: Path,
    *,
    n_workers: int = 8,
    checkpoint: Optional[Path] = None,
    overwrite: bool = False,
    progress: bool = True,
    client_kwargs: Optional[dict] = None,
    dest_fn: Optional[Callable[..., Path]] = None,
    **download_kwargs,
) -> RunSummary:
    """Download a list of URLs -- ``{filename: url}`` or ``[(filename, url)]``.

    The destination file name is the mapping key (relative to ``dest_dir``),
    so no :class:`SurveyArchive` subclass and no catalog are needed.

    Parameters
    ----------
    urls : mapping or sequence of (filename, url)
        ``filename`` becomes ``dest_dir / filename`` (sub-directories are
        created as needed).
    dest_dir : Path
        Directory that receives the files.
    overwrite, progress, client_kwargs, dest_fn
        Forwarded to :class:`DownloadRunner`.
    **download_kwargs
        Forwarded to :meth:`HttpClient.download_file` (e.g. ``resume="never"``,
        ``checksum=("sha256", "...")``, ``progress=...``).

    Examples
    --------
    >>> download_urls({"a.fits.gz": "https://example.org/a.fits.gz"}, "./data")
    """
    mapping = dict(urls)
    if not mapping:
        summary = RunSummary()
        return summary

    dest_dir = Path(dest_dir)
    archive = _UrlListArchive(mapping, overwrite=overwrite,
                              download_kwargs=download_kwargs)
    catalog = pd.DataFrame({
        "obj_id": list(mapping),
        # placeholder coordinates: this archive never uses them
        "ra": [0.0] * len(mapping),
        "dec": [0.0] * len(mapping),
    })
    runner = DownloadRunner(
        archive=archive,
        catalog=catalog,
        store_dir=dest_dir,
        checkpoint=checkpoint,
        n_workers=n_workers,
        progress=progress,
        client_kwargs=client_kwargs,
        dest_fn=dest_fn,
    )
    return runner.run()
