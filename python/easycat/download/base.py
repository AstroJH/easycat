"""Common interfaces for the survey download framework."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


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
    """

    store_dir: Path
    client: Any
    id_column: str = "obj_id"
    ra_column: str = "raj2000"
    dec_column: str = "dej2000"
    radius_arcsec: float = 3.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def row_id(self, row: pd.Series) -> str:
        return str(row[self.id_column])

    def row_coord(self, row: pd.Series):
        """Return (ra, dec) in degrees for a catalog row."""
        return float(row[self.ra_column]), float(row[self.dec_column])


@dataclass
class ItemResult:
    """Outcome of fetching a single source."""

    obj_id: str
    success: bool
    data: Any = None
    error: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ItemResult(obj_id={self.obj_id!r}, success={self.success}, "
            f"error={self.error!r})"
        )


class SurveyArchive(ABC):
    """Base class for survey downloaders.

    Implementations provide :meth:`fetch_batch` (a batch of catalog rows);
    :meth:`fetch_item` is provided for convenience and delegates to a
    single-row batch.

    Subclasses usually define a ``default_batch_size`` class attribute used
    by :class:`~easycat.download.runner.DownloadRunner` when the caller does
    not specify one.
    """

    name: str = "survey"
    default_batch_size: int = 1

    def __init__(self, **config: Any):
        self.config = dict(config)

    @abstractmethod
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        """Fetch data for a batch of rows.

        Must return exactly one :class:`ItemResult` per input row, in the
        same order as ``rows``.
        """

    def fetch_item(self, row: Any, ctx: FetchContext) -> ItemResult:
        if isinstance(row, pd.DataFrame):
            rows = row
        else:
            rows = pd.DataFrame([row])
        return self.fetch_batch(rows, ctx)[0]

    # storage helpers
    def store_path(self, ctx: FetchContext, obj_id: str, suffix: str) -> Path:
        return ctx.store_dir / f"{obj_id}{suffix}"
