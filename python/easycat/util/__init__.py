import numpy as np
from . import dbscan
from typing import Literal
from astropy import units as u
from astropy.units import Quantity
from numba import jit

__all__ = ["dbscan"]


def calc_boundary(data, xsigma=1):
    i_data_min = np.argmin(data)
    i_data_max = np.argmax(data)

    data_ = np.delete(data, [i_data_min, i_data_max])

    mean = np.mean(data_)
    std  = np.std(data_, ddof=1)

    if std == 0: ...

    delta = xsigma * std
    return mean - delta, mean + delta


def fit_histogram1d(bin_lo, bin_hi, data, model):
    """ Fit 1-D histogram data.
    
    """
    M = []

    for lo, hi in zip(bin_lo, bin_hi):
        
        ...
