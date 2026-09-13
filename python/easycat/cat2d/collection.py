"""Multiple samples sharing one geometric selection."""
from __future__ import annotations

from typing import Iterator, Mapping, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .dataset import Cat2D, ScatterPlot


class Cat2DCollection:
    """Named collection of :class:`Cat2D` samples.

    All samples share the same X/Y column names but may have independent
    ranges and sizes.  Polygon selections return one subset per sample.
    """

    def __init__(
        self,
        samples: Mapping[str, Cat2D | pd.DataFrame],
        *,
        x: Optional[str] = None,
        y: Optional[str] = None,
        value: Optional[str] = None,
        errors: Optional[str] = None,
        weights: Optional[str] = None,
    ):
        if not samples:
            raise ValueError("at least one sample is required")
        if x is None or y is None:
            first = next(iter(samples.values()))
            if not isinstance(first, Cat2D):
                raise ValueError("x and y are required for DataFrame samples")
            x = x or first.x
            y = y or first.y

        self.samples: dict[str, Cat2D] = {}
        for name, sample in samples.items():
            if isinstance(sample, Cat2D):
                if sample.x != x or sample.y != y:
                    raise ValueError(
                        f"sample {name!r} uses ({sample.x}, {sample.y}), "
                        f"expected ({x}, {y})"
                    )
                self.samples[str(name)] = sample
            else:
                self.samples[str(name)] = Cat2D(
                    sample, x, y, value, errors=errors, weights=weights,
                )

    @property
    def x(self) -> str:
        return next(iter(self.samples.values())).x

    @property
    def y(self) -> str:
        return next(iter(self.samples.values())).y

    def __getitem__(self, name: str) -> Cat2D:
        return self.samples[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.samples)

    def items(self):
        return self.samples.items()

    def scatter(
        self,
        *,
        ax=None,
        colors=None,
        labels: bool = True,
        **kwargs,
    ) -> ScatterPlot:
        """Plot each sample with a distinct colour."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        palette = list(colors or plt.get_cmap("tab10").colors)
        artists = []
        for i, (name, cat) in enumerate(self.samples.items()):
            plot = cat.scatter(
                ax=ax, color_by=None, color=palette[i % len(palette)],
                colorbar=False, label=name, **kwargs,
            )
            artists.append(plot.artist)
        if labels:
            ax.legend()
        return ScatterPlot(fig=fig, ax=ax, artist=artists)

    def plot(self, *, ax=None, **kwargs):
        """Alias for :meth:`scatter`, returning the axes."""
        return self.scatter(ax=ax, **kwargs).ax

    def select_polygon(self, *, ax=None, marker_size: float = 4.0):
        """Create an interactive polygon selection shared by all samples."""
        from .selection import PolygonSelector

        selector = PolygonSelector(self, ax=ax, marker_size=marker_size)
        selector.ax.set_title(
            "Left click: add | right/backspace: undo | Enter: finish"
        )
        return selector

    def inspect_points(
        self,
        *,
        ax=None,
        on_click=None,
        default: str = "print",
        columns=None,
        pick_radius: float = 8.0,
        highlight: bool = True,
    ):
        """Create a click-to-inspect inspector for all samples."""
        from .inspection import PointInspector

        if ax is None:
            ax = self.scatter().ax
        return PointInspector(
            self, ax=ax, on_click=on_click, default=default,
            columns=columns, pick_radius=pick_radius, highlight=highlight,
        )


__all__ = ["Cat2DCollection"]
