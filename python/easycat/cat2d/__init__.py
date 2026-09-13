"""Two-dimensional catalogue selection, binning and statistics."""
from .binning import GridResult, adaptive_grid, quantile_grid, regular_grid
from .collection import Cat2DCollection
from .dataset import Cat2D, HexBinPlot, MarginalPlot, ScatterPlot
from .inspection import PointInfo, PointInspector
from .selection import (
    MultiSelectionResult,
    PolygonSelector,
    SelectionResult,
    polygon_area,
    polygon_mask,
)
from .statistics import DEFAULT_STATS, summarize

__all__ = [
    "Cat2D",
    "Cat2DCollection",
    "DEFAULT_STATS",
    "GridResult",
    "HexBinPlot",
    "MarginalPlot",
    "MultiSelectionResult",
    "PointInfo",
    "PointInspector",
    "PolygonSelector",
    "ScatterPlot",
    "SelectionResult",
    "adaptive_grid",
    "polygon_area",
    "polygon_mask",
    "quantile_grid",
    "regular_grid",
    "summarize",
]
