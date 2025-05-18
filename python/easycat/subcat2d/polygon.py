import pandas as pd
from pandas import DataFrame
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

def pnpoly(polygon:list[list]|list[tuple], point:tuple|list) -> bool:
    """
    `polygon` and `point` do not accept NaN!!
    """
    px = point[0]
    py = point[1]

    num = len(polygon)
    j = num - 1
    odd_nodes = False
    for i in range(num):
        if (polygon[i][1] < py and polygon[j][1] >= py) or (polygon[j][1] < py and polygon[i][1] >= py):
            if polygon[i][0] + (py - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) * (polygon[j][0] - polygon[i][0]) < px:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes

def get_subsample(datas:list[pd.DataFrame], attr1:str, attr2:str,
                  polygon:list[list]|list[tuple]):
    result:list[pd.DataFrame] = []
    for data in datas:
        mask = np.empty(len(data), dtype=np.bool_)
        for i in range(0, len(data)):
            row = data.iloc[i]
            val1 = row[attr1]
            val2 = row[attr2]
            mask[i] = pnpoly(polygon, (val1, val2))
        result.append(data[mask])

    return result


COLOR_LST = ["tab:red", "tab:blue", "green", "orange", "purple"]
class PolygonGuiSelector():
    def __init__(self, data:list[DataFrame]|DataFrame, attr1:str, attr2:str):
        self._data:list[DataFrame] = [data] if isinstance(data, pd.DataFrame) else data
        self._polygon:list[tuple] = []
        self._artist_lines:list[Line2D] = []

        self.attr1 = attr1
        self.attr2 = attr2
        
        self.fig, self.ax = plt.subplots()
        ax = self.ax
        fig = self.fig

        for i, d in enumerate(self._data):
            col1 = d[attr1]
            col2 = d[attr2]
            ax.plot(col1, col2, marker="o", markersize=5, color=COLOR_LST[i],
                    markerfacecolor="none", linestyle="none", alpha=0.1)
            ax.set_xlabel(attr1)
            ax.set_ylabel(attr2)

        fig.canvas.mpl_connect("button_press_event", self._on_click)
    
    def _on_click(self, event:MouseEvent):
        polygon = self._polygon
        artist_lines = self._artist_lines
        ax = self.ax
        fig = self.fig

        if event.button is MouseButton.LEFT:
            x = event.xdata
            y = event.ydata
            ax.plot(x, y, marker="*", markersize=5, color="k",
                     markerfacecolor="none", linestyle="none")
            
            if len(polygon) > 0:
                if len(artist_lines) > 0:
                    l = artist_lines.pop()
                    l.remove()

                lp = polygon[-1] # last point
                fp = polygon[0] # first point
                l1, = ax.plot([x, lp[0]], [y, lp[1]], color="k", linestyle="--")
                l2, = ax.plot([x, fp[0]], [y, fp[1]], color="k", linestyle="--")
                artist_lines.append(l1)
                artist_lines.append(l2)

            polygon.append((x, y))
            plt.pause(0.25)
    
    def get_subsample(self):
        return get_subsample(self._data, self.attr1, self.attr2, self._polygon)