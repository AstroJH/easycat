import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np
import matplotlib as mpl
from pandas import DataFrame
from typing import Callable, Literal


def calc_limition(rectangles:list[Rectangle], obj:Rectangle):
    lefts = []
    rights = []
    tops = []
    bottoms = []

    for rect in rectangles:
        if rect == obj: continue
        x_left = rect.get_x()
        x_right = x_left+rect.get_width()
        y_bottom = rect.get_y()
        y_top = y_bottom + rect.get_height()

        lefts.append(x_left)
        rights.append(x_right)
        tops.append(y_top)
        bottoms.append(y_bottom)
    
    x_left = obj.get_x()
    x_right = x_left+obj.get_width()
    y_bottom = obj.get_y()
    y_top = y_bottom + obj.get_height()

    bottoms = np.array(bottoms)
    tops = np.array(tops)
    lefts = np.array(lefts)
    rights = np.array(rights)

    d_tops = bottoms-y_top
    d_bottoms = y_bottom-tops
    d_lefts = rights-x_left
    d_rights = x_right-lefts

    min_top = np.min(d_tops[d_tops>=0])
    min_bottom = -np.min(d_bottoms[d_bottoms>=0])
    min_left = -np.min(d_lefts[d_lefts>=0])
    min_right = np.min(d_rights[d_rights>=0])

    if np.isnan(min_top): min_top = np.inf
    if np.isnan(min_bottom): min_bottom = -np.inf
    if np.isnan(min_left): min_left = -np.inf
    if np.isnan(min_right): min_right = np.inf

    return [(min_bottom, min_top), (min_left, min_right)]