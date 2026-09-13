"""Survey data download framework.

New (batch-oriented) API::

    from easycat.download import WISEArchive, ZTFArchive, DownloadRunner

    runner = DownloadRunner(
        archive=WISEArchive(radius_arcsec=3),
        catalog=catalog,              # DataFrame with obj_id/raj2000/dej2000
        store_dir="./wise_data",
        checkpoint="./wise_data/checkpoint.json",
        n_workers=4,
    )
    summary = runner.run()
"""
from .base import FetchContext, ItemResult, SurveyArchive
from .checkpoint import CheckpointStore
from .client import (
    ChecksumMismatch,
    DownloadError,
    HttpClient,
    HttpError,
    IncompleteDownload,
    RangeFile,
    RangeNotSupported,
    VerificationFailed,
    get_default_client,
    reset_default_client,
)
from .runner import DownloadRunner, RunSummary, download_urls

from .survey.wise import WISEArchive
from .survey.ztf import ZTFArchive
from .survey.desi import DESIArchive
from .survey.sdss import SDSSArchive
from .survey.panstarrs import PanSTARRSArchive
from .survey.muse import MUSEArchive

__all__ = [
    # Core framework
    "DownloadError",
    "HttpError",
    "RangeNotSupported",
    "IncompleteDownload",
    "ChecksumMismatch",
    "VerificationFailed",
    "HttpClient",
    "RangeFile",
    "get_default_client",
    "reset_default_client",
    "CheckpointStore",
    "DownloadRunner",
    "download_urls",
    "RunSummary",
    "FetchContext",
    "ItemResult",
    "SurveyArchive",

    # Survey archives
    "WISEArchive",
    "ZTFArchive",
    "DESIArchive",
    "SDSSArchive",
    "PanSTARRSArchive",
    "MUSEArchive",
]
