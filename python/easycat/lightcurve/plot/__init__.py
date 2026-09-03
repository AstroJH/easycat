import numpy as np

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import ticker

from astropy.coordinates import SkyCoord
import pandas as pd
from astropy import units as u
from astropy.units import Quantity


def plot_positions(ax:Axes, positions:SkyCoord, pos_ref:SkyCoord, ref_radius:Quantity=2*u.arcsec):
    ax.scatter(positions.ra.to_value("deg"),
               positions.dec.to_value("deg"))
    
    d = ref_radius.to_value("deg")

    alpha_ref = pos_ref.ra.to_value("deg")
    delta_ref = pos_ref.dec.to_value("deg")

    t = np.linspace(0, 2*np.pi, 100)
    alpha = alpha_ref + d*np.cos(t)/np.cos(delta_ref/180*np.pi)
    delta = delta_ref + d*np.sin(t)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False, useMathText=False))
    ax.xaxis.get_major_formatter().set_scientific(False)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.xaxis.set_inverted(True)

    ax.plot(alpha, delta, color="k")


def plot_wiselc(lcurve:pd.DataFrame):
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(2, 1, figure=fig, wspace=0, hspace=0, height_ratios=(2, 1))

    ax_mag = fig.add_subplot(gs[0])
    ax_color = fig.add_subplot(gs[1])
    ax_color.sharex(ax_mag)
    ax_mag.xaxis.set_visible(False)


    mjd_start = lcurve.mjd[0]
    rmjd = (lcurve.mjd - mjd_start)/365
    ax_color.set_xlabel(f"$\\rm \\frac{{MJD-{mjd_start:.2f}}}{{365}}$ [yr]")

    param_errorbar = {
        "linestyle": "none",
        "capsize": 2,
        "marker": ".",
        "markersize": 5
    }

    w1data = (rmjd, lcurve.w1mag, lcurve.w1sigmag)
    w2data = (rmjd, lcurve.w2mag, lcurve.w2sigmag)
    ax_mag.invert_yaxis()

    ax_mag.errorbar(*w1data, color="tab:blue", **param_errorbar)
    ax_mag.errorbar(*w2data, color="tab:red",  **param_errorbar)

    color = lcurve.w1mag - lcurve.w2mag
    sigcolor = np.sqrt(lcurve.w1sigmag**2 + lcurve.w2sigmag**2)
    ax_color.errorbar(rmjd, color, sigcolor, color="green", **param_errorbar)
    return fig
