from .wise import WiseReprocessor
from .ztf import ZtfReprocessor
from .core import ReprocessFactory, LightcurveReprocessor
from . import util

__all__ = [
    "ReprocessFactory",
    "LightcurveReprocessor",
    "WiseReprocessor",
    "ZtfReprocessor",
    "util"
]

for processor in [WiseReprocessor, ZtfReprocessor]:
    ReprocessFactory.register_processor(processor)
