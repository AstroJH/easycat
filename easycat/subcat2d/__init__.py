import pandas as pd
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np

def pnpoly(polygon:list[list]|list[tuple], point:tuple|list):
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

def generate_polygon(data:list[pd.DataFrame]|pd.DataFrame, attr1:str, attr2:str):
    if isinstance(data, pd.DataFrame):
        data_lst = [data]
    else:
        data_lst = data

    polygon = []

    lines = []
    def on_click(event):
        if event.button is MouseButton.LEFT:
            x = event.xdata
            y = event.ydata
            plt.plot([x], [y], marker="*", markersize=5, color="k",  markerfacecolor="none", linestyle="none")
            
            if len(polygon) > 0:
                if len(lines) > 0:
                    l = lines.pop()
                    l.remove()

                lp = polygon[-1] # last point
                fp = polygon[0] # first point
                l1, = plt.plot([x, lp[0]], [y, lp[1]], color="k", linestyle="--")
                l2, = plt.plot([x, fp[0]], [y, fp[1]], color="k", linestyle="--")
                lines.append(l1)
                lines.append(l2)

            polygon.append((x, y))
            plt.pause(0.25)

    color_list = ["tab:red", "tab:blue", "green", "orange", "purple"]
    for i, d in enumerate(data_lst):
        col1 = d[attr1]
        col2 = d[attr2]
        plt.plot(col1, col2, marker="o", markersize=5, color=color_list[i],
                 markerfacecolor="none", linestyle="none", alpha=0.1)
        plt.xlabel(attr1)
        plt.ylabel(attr2)

    plt.connect('button_press_event', on_click)
    plt.show()
    plt.close()
    return polygon

def get_subsample(datas:list[pd.DataFrame], attr1:str, attr2:str):
    polygon = generate_polygon(datas, attr1, attr2)
    print(polygon)

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