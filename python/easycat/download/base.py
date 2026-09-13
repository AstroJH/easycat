"""Common interfaces for the survey download framework."""
from __future__ import annotations

import inspect
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd


DestFn = Callable[..., Path]


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    """Return whether ``func`` accepts the keyword ``name``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


@dataclass
class FetchContext:
    """Runtime context passed to every archive call.

    Attributes
    ----------
    store_dir : Path
        Directory where per-source files are written.
    client : Any
        :class:`~easycat.download.client.HttpClient` instance (shared by all
        workers, so retries / rate limiting are applied globally).
    id_column, ra_column, dec_column : str
        Catalog column names.
    radius_arcsec : float
        Default matching radius in arcseconds.
    extra : dict
        Survey-specific free-form context.
    download_kwargs : dict
        Per-call keyword arguments forwarded to
        :meth:`HttpClient.download_file`.  Archive-level defaults are stored on
        :class:`SurveyArchive`; values here take precedence.
    dest_fn : callable or None
        Optional output-path hook.  Two-argument callables
        ``(obj_id, default_path)`` remain supported; callables may additionally
        accept a keyword-only ``row`` containing the catalog row.
    """

    store_dir: Path
    client: Any
    id_column: str = "obj_id"
    ra_column: str = "raj2000"
    dec_column: str = "dej2000"
    radius_arcsec: float = 3.0
    extra: Dict[str, Any] = field(default_factory=dict)
    dest_fn: Optional[DestFn] = None
    download_kwargs: Dict[str, Any] = field(default_factory=dict)
    _dest_fn_accepts_row: Optional[bool] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def row_id(self, row: pd.Series) -> str:
        return str(row[self.id_column])

    def row_coord(self, row: pd.Series):
        """Return (ra, dec) in degrees for a catalog row."""
        return float(row[self.ra_column]), float(row[self.dec_column])

    def dest(
        self,
        obj_id: str,
        default: Path,
        *,
        row: Optional[pd.Series] = None,
    ) -> Path:
        """Apply the optional ``dest_fn`` override to ``default``.

        Archives call this when writing their per-source file *and* when
        reporting :meth:`SurveyArchive.output_path`, so custom layouts stay
        consistent with the resume safety net.  New hooks may accept
        ``row=...``; existing two-argument hooks continue to work unchanged.
        """
        if self.dest_fn is None:
            return Path(default)
        if self._dest_fn_accepts_row is None:
            self._dest_fn_accepts_row = _accepts_keyword(self.dest_fn, "row")
        if self._dest_fn_accepts_row:
            return Path(self.dest_fn(str(obj_id), Path(default), row=row))
        return Path(self.dest_fn(str(obj_id), Path(default)))


@dataclass
class ItemResult:
    """Outcome of fetching a single source.

    ``meta`` contains layout-independent facts about this fetch, e.g.
    ``url``, ``dest``, ``size``, ``checksum``, ``product`` and ``version``.
    It is intentionally separate from ``data`` so consumers can record
    provenance without interpreting product-specific return values.
    """

    obj_id: str
    success: bool
    data: Any = None
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ItemResult(obj_id={self.obj_id!r}, success={self.success}, "
            f"error={self.error!r}, meta={self.meta!r})"
        )


class _RecordingClient:
    """Delegate to an HttpClient while recording product downloads.

    ``download=False`` intercepts ``download_file`` so file-oriented archives
    can resolve metadata without transferring the payload.  Query-oriented
    archives still perform their query and may write small result tables to a
    temporary directory managed by :meth:`SurveyArchive.fetch_one`.
    """

    def __init__(self, client: Any, *, download: bool):
        self.client = client
        self.download = download
        self.calls: List[Dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def download_file(self, url: str, dest: Any, **kwargs):
        path = Path(dest)
        call = {
            "url": str(url),
            "dest": str(path),
            "download_kwargs": dict(kwargs),
        }
        self.calls.append(call)
        if self.download:
            path = Path(self.client.download_file(url, path, **kwargs))
            call["dest"] = str(path)
        return path


class SurveyArchive(ABC):
    """Base class for survey downloaders.

    Implementations provide :meth:`fetch_batch` (a batch of catalog rows);
    :meth:`fetch_one` adds a single-product, checkpoint-free API on top.
    :meth:`fetch_item` remains the low-level one-row convenience method.

    Subclasses usually define a ``default_batch_size`` class attribute used
    by :class:`~easycat.download.runner.DownloadRunner` when the caller does
    not specify one.
    """

    name: str = "survey"
    default_batch_size: int = 1

    def __init__(
        self,
        *,
        download_kwargs: Optional[Dict[str, Any]] = None,
        **config: Any,
    ):
        self.config = dict(config)
        self.download_kwargs = dict(download_kwargs or {})

    @abstractmethod
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        """Fetch data for a batch of rows.

        Must return exactly one :class:`ItemResult` per input row, in the
        same order as ``rows``.
        """

    def fetch_item(self, row: Any, ctx: FetchContext) -> ItemResult:
        """Low-level convenience wrapper around ``fetch_batch``.

        This method still requires an existing :class:`FetchContext`.
        New consumer integrations should normally use :meth:`fetch_one`.
        """
        if isinstance(row, pd.DataFrame):
            rows = row
        else:
            rows = pd.DataFrame([row])
        return self.fetch_batch(rows, ctx)[0]

    def fetch_one(
        self,
        target: Any,
        *,
        client: Any = None,
        dest: Optional[Path] = None,
        store_dir: Optional[Path] = None,
        download: bool = True,
        progress: Any = None,
        **download_kwargs: Any,
    ) -> ItemResult:
        """Fetch or resolve one item without a checkpoint or catalog batch.

        Parameters
        ----------
        target : str, mapping, pandas.Series or one-row DataFrame
            A source identifier or a row carrying the columns required by the
            archive.  Coordinate archives need coordinate columns; identifiers
            that directly encode the product (e.g. MaNGA ``plateifu``) can be
            passed as a string.
        client : HttpClient or None
            HTTP client.  Defaults to the process-wide shared client.
        dest : Path or None
            Exact output file chosen by the consumer.  When supplied,
            ``store_dir`` is not needed.
        store_dir : Path or None
            Directory where the archive's default output naming is used.
            Exactly one of ``dest`` / ``store_dir`` is required when
            ``download=True``.
        download : bool
            If false, resolve metadata without persisting binary payloads.
            Query-oriented archives may still query a service and use a
            temporary directory internally.
        progress : bool, callable or tqdm-like or None
            Forwarded to ``HttpClient.download_file``.
        **download_kwargs
            Additional ``download_file`` options (``validate``, ``checksum``,
            ``resume``, ``overwrite``, ...).  They override archive defaults.

        Returns
        -------
        ItemResult
            ``success=True, data=None`` means the query succeeded but the
            requested product does not exist.  ``success=False`` is reserved
            for operational or parsing failures.
        """
        if dest is not None and store_dir is not None:
            raise ValueError("pass only one of dest= or store_dir=")
        if download and dest is None and store_dir is None:
            raise ValueError("fetch_one(download=True) requires dest= or store_dir=")

        row = self._coerce_target_row(target)
        row = self.prepare_item_row(target, row)
        obj_id = str(row.get("obj_id", target))

        if client is None:
            from .client import HttpClient

            client = HttpClient.shared()

        tmpdir: Optional[tempfile.TemporaryDirectory] = None
        dest_path = Path(dest) if dest is not None else None
        if not download:
            # Resolve-only mode must never write into a consumer-selected
            # product path, so all direct writes go to a temporary directory.
            tmpdir = tempfile.TemporaryDirectory(prefix="easycat-fetch-one-")
            actual_store_dir = Path(tmpdir.name)
            dest_fn = None
        elif dest_path is not None:
            actual_store_dir = dest_path.parent

            def dest_fn(obj_id, default, *, row=None):
                return dest_path

        elif store_dir is not None:
            actual_store_dir = Path(store_dir)
            dest_fn = None
        request_kwargs = dict(download_kwargs)
        if progress is not None:
            request_kwargs["progress"] = progress

        recorder = _RecordingClient(client, download=download)
        ctx = FetchContext(
            store_dir=actual_store_dir,
            client=recorder,
            id_column="obj_id",
            ra_column="raj2000",
            dec_column="dej2000",
            download_kwargs=request_kwargs,
            dest_fn=dest_fn,
        )

        try:
            results = self.fetch_batch(pd.DataFrame([row]), ctx)
            if len(results) != 1:
                raise RuntimeError(
                    f"{self.name}.fetch_batch returned {len(results)} results "
                    "for one item"
                )
            result = results[0]
        except Exception as exc:
            result = ItemResult(obj_id=obj_id, success=False, error=repr(exc))

        inferred_dest = dest_path or self._infer_output_path(ctx, obj_id, row, result)
        call = recorder.calls[-1] if recorder.calls else None
        if call is not None:
            inferred_dest = Path(call["dest"])
        if not download and dest_path is not None:
            inferred_dest = dest_path

        result = self.enrich_result(
            result,
            row=row,
            url=call["url"] if call is not None else None,
            dest=inferred_dest,
            download_kwargs=request_kwargs,
        )
        if not download and dest_path is not None:
            result.meta["dest"] = str(dest_path)
        result.meta["download"] = bool(download)

        if not download and isinstance(result.data, Path):
            result.data = None

        if tmpdir is not None:
            tmpdir.cleanup()
        return result

    def _coerce_target_row(self, target: Any) -> pd.Series:
        """Normalise the public ``fetch_one`` target into a catalog row."""
        if isinstance(target, pd.DataFrame):
            if len(target) != 1:
                raise ValueError("a one-row DataFrame is required")
            row = target.iloc[0].copy()
        elif isinstance(target, pd.Series):
            row = target.copy()
        elif isinstance(target, Mapping):
            row = pd.Series(dict(target))
        else:
            row = pd.Series({"obj_id": str(target)})

        if "obj_id" not in row.index:
            for name in ("plateifu", "PLATEIFU", "targetid", "TARGETID", "id", "name"):
                if name in row.index:
                    row["obj_id"] = str(row[name])
                    break
        if "obj_id" not in row.index:
            row["obj_id"] = str(target)
        if "raj2000" not in row.index and "ra" in row.index:
            row["raj2000"] = row["ra"]
        if "dej2000" not in row.index and "dec" in row.index:
            row["dej2000"] = row["dec"]
        if "raj2000" not in row.index and "RA" in row.index:
            row["raj2000"] = row["RA"]
        if "dej2000" not in row.index and "DEC" in row.index:
            row["dej2000"] = row["DEC"]
        return row

    def prepare_item_row(self, target: Any, row: pd.Series) -> pd.Series:
        """Archive-specific hook for turning an identifier into columns."""
        return row

    def item_metadata(self, row: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Return layout-independent, product-level metadata."""
        meta: Dict[str, Any] = {"archive": self.name}
        product = getattr(self, "manga_product", None) or getattr(self, "mode", None)
        if product is None:
            product = self.name
        meta["product"] = str(product)
        for attr in (
            "manga_version", "manga_dap_version", "ls_release", "release",
            "manga_dap",
        ):
            value = getattr(self, attr, None)
            if value:
                meta["version"] = str(value)
                break
        return meta

    def enrich_result(
        self,
        result: ItemResult,
        *,
        row: Optional[pd.Series] = None,
        url: Optional[str] = None,
        dest: Optional[Path] = None,
        download_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ItemResult:
        """Attach archive and download provenance to an ``ItemResult``."""
        meta = self.item_metadata(row)
        meta.update(result.meta)
        if url is not None:
            meta.setdefault("url", str(url))
        if dest is not None:
            path = Path(dest)
            meta.setdefault("dest", str(path))
            if path.exists() and path.is_file():
                meta.setdefault("size", path.stat().st_size)
        kwargs = download_kwargs or self.download_kwargs
        checksum = kwargs.get("checksum") if kwargs else None
        if checksum is not None:
            meta.setdefault("checksum", checksum)
        result.meta = meta
        return result

    def _infer_output_path(
        self,
        ctx: FetchContext,
        obj_id: str,
        row: pd.Series,
        result: ItemResult,
    ) -> Optional[Path]:
        if isinstance(result.data, Path):
            return result.data
        return self.resolve_output_path(ctx, obj_id, row=row)

    def resolve_output_path(
        self,
        ctx: FetchContext,
        obj_id: str,
        *,
        row: Optional[pd.Series] = None,
    ) -> Optional[Path]:
        """Call ``output_path`` while supporting both old and new signatures."""
        try:
            if _accepts_keyword(self.output_path, "row"):
                return self.output_path(ctx, obj_id, row=row)
            return self.output_path(ctx, obj_id)
        except Exception:
            return None

    def _download_file(
        self,
        ctx: FetchContext,
        url: str,
        dest: Path,
        **overrides: Any,
    ) -> Path:
        """Download a product using archive-level and per-call options."""
        kwargs = dict(self.download_kwargs)
        kwargs.update(ctx.download_kwargs)
        kwargs.update(overrides)
        return Path(ctx.client.download_file(url, dest, **kwargs))

    # storage helpers
    def store_path(
        self,
        ctx: FetchContext,
        obj_id: str,
        suffix: str,
        *,
        row: Optional[pd.Series] = None,
    ) -> Path:
        return ctx.dest(obj_id, ctx.store_dir / f"{obj_id}{suffix}", row=row)

    def output_path(
        self,
        ctx: FetchContext,
        obj_id: str,
        *,
        row: Optional[pd.Series] = None,
    ) -> Optional[Path]:
        """Expected output file for one source (or ``None`` if unknown).

        Used as a *safety net* by :class:`~easycat.download.runner.DownloadRunner`:
        archives write their files before the runner records them, so after a
        hard interrupt (SIGKILL, kernel restart, closed notebook) a finished
        download can be missing from the checkpoint.  If this method returns a
        path that already exists, the runner marks the source as done instead
        of downloading it again.

        Implementations should return ``None`` when a source may legitimately
        produce *no* file (e.g. "no data" results), since a missing file must
        never be interpreted as "not finished".
        """
        return None
