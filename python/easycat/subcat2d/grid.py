import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import matplotlib as mpl
from pandas import DataFrame
from typing import Callable, List

class GridGuiSelector():
    def __init__(self, data: DataFrame, attr1: str, attr2: str):
        self.state = 0
        self.data = data
        self.attr1 = attr1
        self.attr2 = attr2

        self.xps = []
        self.yps = []

        self.fig, self.ax = plt.subplots()
        fig = self.fig
        ax = self.ax

        col1 = data[attr1]
        col2 = data[attr2]
        ax.plot(col1, col2, marker="o", markersize=5, color="tab:blue",
                    markerfacecolor="none", linestyle="none", alpha=0.1)
        ax.set_xlabel(attr1)
        ax.set_ylabel(attr2)

        self.axbtn1 = fig.add_axes([0.5, 0.05, 0.1, 0.075])
        self.axbtn2 = fig.add_axes([0.61, 0.05, 0.1, 0.075])
        self.axbtn3 = fig.add_axes([0.72, 0.05, 0.1, 0.075])
        self.btn1 = Button(self.axbtn1, "Add (H)") # 1
        self.btn2 = Button(self.axbtn2, "Add (V)") # 2
        self.btn3 = Button(self.axbtn3, "Select")  # 3

        fig.canvas.mpl_connect("pick_event", self._on_pick)
        fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event: MouseEvent):
        ax = self.ax
        if   event.inaxes is self.axbtn1: self.state = 1
        elif event.inaxes is self.axbtn2: self.state = 2
        elif event.inaxes is self.axbtn3: self.state = 3

        elif event.inaxes is self.ax:
            if (event.button is MouseButton.LEFT) and (self.state in [1, 2]):
                x = event.xdata
                y = event.ydata

                if self.state == 1:
                    ax.axhline(y=y, linestyle="--", color="k", picker=True, pickradius=15)
                    self.yps.append(y)
                else:
                    ax.axvline(x=x, linestyle="--", color="k", picker=True, pickradius=15)
                    self.xps.append(x)
                event.canvas.draw()

    def _on_pick(self, event:PickEvent):
        if self.state != 3: return

        artist = event.artist
        if isinstance(artist, Line2D):
            artist.set_color("red")
            event.canvas.draw()
    
    def calc(self, mapper: Callable[[DataFrame], float]):
        self.xps.sort()
        self.yps.sort(reverse=True)

        X, Y = np.meshgrid(self.xps, self.yps)
        n_row, n_col = X.shape

        fig, ax = plt.subplots()
        col1 = self.data[self.attr1]
        col2 = self.data[self.attr2]
        ax.plot(col1, col2, marker="o", markersize=5, color="tab:blue",
                    markerfacecolor="none", linestyle="none", alpha=0.1)
        ax.set_xlabel(self.attr1)
        ax.set_ylabel(self.attr2)

        colors = ['darkorange', 'gold', 'lawngreen', 'lightseagreen']
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)

        vals = []
        rects = []
        for i in range(0, n_row-1):
            for j in range(0, n_col-1):
                p0 = (X[i,j], Y[i,j])
                p1 = (X[i,j+1], Y[i,j+1])
                p2 = (X[i+1,j+1], Y[i+1,j+1])
                p3 = (X[i+1,j], Y[i+1,j])

                height = p0[1]-p3[1]
                width = p1[0]-p0[0]

                mask1 = (col1>=p0[0]) & (col1<=p1[0])
                mask2 = (col2>=p3[1]) & (col2<=p0[1])
                mask = mask1 & mask2

                val = mapper(self.data[mask])
                vals.append(val)

                rects.append((p3, width, height))
        
        cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for val, rect in zip(vals,rects):
            p3, width, height = rect
            ax.add_patch(Rectangle(p3, width, height, edgecolor="k", facecolor=mappable.to_rgba(val)))


def regular_grid_map(
    data: DataFrame,
    attr1: str,
    attr2: str,
    edge1: List,
    edge2: List,
    mapper: Callable[[DataFrame], float],
    *,
    need_rects: bool = False
):
    col1 = data[attr1]
    col2 = data[attr2]

    lo_edge1 = edge1[:-1]
    hi_edge1 = edge1[1:]
    lo_edge2 = edge2[:-1]
    hi_edge2 = edge2[1:]

    X1 = []
    X2 = []
    Y = []
    num = []

    for lo1, hi1 in zip(lo_edge1, hi_edge1):
        for lo2, hi2 in zip(lo_edge2, hi_edge2):
            mask = (col1 > lo1) & (col1 <= hi1) & (col2 > lo2) & (col2 <= hi2)
            subdata = data[mask]
            subcol1 = col1[mask]
            subcol2 = col2[mask]

            if len(subdata) <= 0:
                X1.append(np.nan)
                X2.append(np.nan)
                Y.append(np.nan)
                num.append(0)
            else:
                Y.append(mapper(subdata))
                X1.append(np.median(subcol1))
                X2.append(np.median(subcol2))
                num.append(len(subdata))
    
    result = {
        "X1": np.array(X1),
        "X2": np.array(X2),
        "Y": np.array(Y),
        "num": np.array(num)
    }

    if not need_rects:
        return result
    
    rects_attr = []
    for lo1, hi1 in zip(lo_edge1, hi_edge1):
        for lo2, hi2 in zip(lo_edge2, hi_edge2):
            width1 = hi1 - lo1
            width2 = hi2 - lo2

            rects_attr.append((lo1, lo2, width1, width2))
    
    result.update({
        "rects": np.array(rects_attr)
    })
    
    return result
