from .wise import WISEReprocessor
from .ztf import ZTFReprocessor
from .core import ReprocessFactory, LightcurveReprocessor

__all__ = [
    "ReprocessFactory",
    "LightcurveReprocessor",
    "WISEReprocessor",
    "ZTFReprocessor"
]

for processor in [WISEReprocessor, ZTFReprocessor]:
    ReprocessFactory.register(processor)
