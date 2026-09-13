"""Public API for easycat's single-source and batch pipeline framework."""

from .core import (
    DataPacket,
    NodeOutcome,
    NodeStatus,
    Pipeline,
    PipelineRun,
    ProcessingNode,
)
from .runner import PipelineRunner, PipelineRunSummary
from .builder import PipelineBuilder
from .nodes import (
    OutlierFilterNode,
    PositionFilterNode,
    BinningNode,
    EpochCleanNode
)

__all__ = [
    "Pipeline",
    "DataPacket",
    "NodeOutcome",
    "NodeStatus",
    "ProcessingNode",
    "PipelineRun",
    "PipelineRunner",
    "PipelineRunSummary",
    "PipelineBuilder",
    "OutlierFilterNode",
    "PositionFilterNode",
    "BinningNode",
    "EpochCleanNode",
]
