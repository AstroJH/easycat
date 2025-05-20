from .wise import WiseReprocessor
from .ztf import ZtfReprocessor
from .core import ReprocessFactory, LightcurveReprocessor

__all__ = [
    "ReprocessFactory",
    "LightcurveReprocessor",
    "WiseReprocessor",
    "ZtfReprocessor"
]

for processor in [WiseReprocessor, ZtfReprocessor]:
    ReprocessFactory.register(processor)
