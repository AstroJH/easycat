from . import download
from . import subcat2d
from . import lightcurve
from . import parallel
from . import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('easycat')

__all__ = [
    "download",
    "subcat2d",
    "lightcurve",
    "parallel",
    "stats",
]