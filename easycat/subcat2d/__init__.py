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

def generate_polygon(data:pd.DataFrame, attr1:str, attr2:str):
    col1 = data[attr1]
    col2 = data[attr2]

    polygon_attr1 = []
    polygon_attr2 = []
    polygon = []

    def on_click(event):
        if event.button is MouseButton.LEFT:
            x = event.xdata
            y = event.ydata
            polygon_attr1.append(x)
            polygon_attr2.append(y)
            polygon.append((x, y))
            plt.plot([x], [y], marker="*", markersize=5, color="tab:red",  markerfacecolor="none", linestyle="none")
            plt.pause(0.25)

    plt.plot(col1, col2, marker="o", markersize=5, color="tab:blue",  markerfacecolor="none", linestyle="none", alpha=0.1)
    plt.xlabel(attr1)
    plt.ylabel(attr2)

    plt.connect('button_press_event', on_click)
    plt.show()
    plt.close()
    return polygon_attr1, polygon_attr2, polygon

def get_subsample(data:pd.DataFrame, attr1:str, attr2:str):
    mask = np.empty(len(data), dtype=np.bool_)
    polygon_attr1, polygon_attr2, polygon = generate_polygon(data, attr1, attr2)
    print(polygon)
    for i in range(0, len(data)):
        row = data.iloc[i]
        val1 = row[attr1]
        val2 = row[attr2]
        mask[i] = pnpoly(polygon, (val1, val2))
    return data[mask]