from .wise import WISEReprocessor
from .ztf import ZTFReprocessor
from .core import ReprocessFactory, LightcurveReprocessor
from .util import bin_lightcurve

__all__ = [
    "ReprocessFactory",
    "LightcurveReprocessor",
    "WISEReprocessor",
    "ZTFReprocessor",
    "bin_lightcurve"
]

for processor in [WISEReprocessor, ZTFReprocessor]:
    ReprocessFactory.register(processor)
