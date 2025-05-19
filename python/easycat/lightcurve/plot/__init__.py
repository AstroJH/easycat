import numpy as np

from matplotlib.axes import Axes
from astropy.coordinates import SkyCoord
import pandas as pd
from ...util import grp_by_max_interval, databinner
from ..reprocess import WiseReprocessor

def plot_positions(ax:Axes, positions:SkyCoord, pos_ref:SkyCoord):
    ax.scatter(positions.ra.to_value("deg"),
               positions.dec.to_value("deg"))
    

    d = 2/3600
    alpha_ref = pos_ref.ra.to_value("deg")
    delta_ref = pos_ref.dec.to_value("deg")

    t = np.linspace(0, 2*np.pi, 100)
    alpha = alpha_ref + d*np.cos(t)/np.cos(delta_ref/180*np.pi)
    delta = delta_ref + d*np.sin(t)

    ax.plot(alpha, delta, color="k")


def plot_wiselc(ax:Axes, lcurve:pd.DataFrame):
    mjd = lcurve["mjd"]
    w1mag = lcurve["w1mag"]
    w2mag = lcurve["w2mag"]
    w1err = lcurve["w1sigmag"]
    w2err = lcurve["w2sigmag"]


    start_mjd = mjd[0]
    rmjd = mjd - start_mjd

    errorbar_param = {
        "color": "grey",
        "linestyle": "none",
        "capsize": 4,
        "alpha": 0.3
    }
    ax.errorbar(rmjd, w1mag, yerr=w1err, **errorbar_param)
    ax.errorbar(rmjd, w2mag, yerr=w2err, **errorbar_param)

    
    errorbar_param = {
        "linestyle": "none",
        "capsize": 8,
        "marker": "o",
        "markersize": 8
    }

    longterm_lcurve = WiseReprocessor().generate_longterm_lcurve(lcurve, max_interval=50)
    
    ax.errorbar(
        x=longterm_lcurve["mjd"]-start_mjd,
        y=longterm_lcurve["w1mag"],
        yerr=longterm_lcurve["w1sigmag"],
        color="tab:blue",
        **errorbar_param
    )

    ax.errorbar(
        x=longterm_lcurve["mjd"]-start_mjd,
        y=longterm_lcurve["w2mag"],
        yerr=longterm_lcurve["w2sigmag"],
        color="tab:red",
        **errorbar_param
    )

    ax.yaxis.set_inverted(True)
    

