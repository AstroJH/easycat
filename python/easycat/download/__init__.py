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
from .client import HttpClient
from .runner import DownloadRunner, RunSummary

from .survey.wise import WISEArchive
from .survey.ztf import ZTFArchive
from .survey.desi import DESIArchive
from .survey.sdss import SDSSArchive
from .survey.panstarrs import PanSTARRSArchive

__all__ = [
    # Core framework
    "HttpClient",
    "CheckpointStore",
    "DownloadRunner",
    "RunSummary",
    "FetchContext",
    "ItemResult",
    "SurveyArchive",

    # Survey archives
    "WISEArchive",
    "ZTFArchive",
    "DESIArchive",
    "SDSSArchive",
    "PanSTARRSArchive"
]