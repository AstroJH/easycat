import numpy as np
from . import dbscan
from typing import Literal
from astropy import units as u
from astropy.units import Quantity

__all__ = ["dbscan", "grp_by_max_interval"]

def grp_by_max_interval(data, max_interval=1.2):
    lo = []
    hi = []

    N = len(data)
    if N == 0:
        return (np.array(lo), np.array(hi))

    lo.append(0)
    i = 1
    while i < N:
        if data[i] - data[i-1] > max_interval:
            hi.append(i-1)
            lo.append(i)
        i += 1
    hi.append(N-1)
    
    return (np.array(lo), np.array(hi))


def calc_boundary(data, xsigma=1):
    i_data_min = np.argmin(data)
    i_data_max = np.argmax(data)

    data_ = np.delete(data, [i_data_min, i_data_max])

    mean = np.mean(data_)
    std  = np.std(data_, ddof=1)

    if std == 0: ...

    delta = xsigma * std
    return mean - delta, mean + delta


def find_outliers(data, threshold=3):

    if len(data) < 5:
        return np.empty(0, dtype=np.intp) # 至少 5 个数据点才施行异常点检测
    
    lower, upper = calc_boundary(data, xsigma=threshold)
    
    return np.where(
        (data < lower) | (data > upper)
    )



def databinner(data, sigmas, method="mean", skipnan=False):
    """ a simple tool to help calculate error transfer when binning data

    TODO [jhwu] Emmmm... Only the mean-algorithm under uncorrelated assumptions is implemented.
    """
    if skipnan:
        m1 = ~np.isnan(data)
        m2 = ~np.isnan(sigmas)
        m  = m1 & m2

        data  = data[m]
        sigmas = sigmas[m]
    
    if len(data) == 0:
        return np.nan, np.nan
        # raise ValueError("Empty data.", 0)
    
    if sigmas is None:
        return np.mean(data), 0.0
    else:
        if len(data) != len(sigmas):
            raise Exception("len(data) and len(sigmas) must be same.")

        # \sigma = \sqrt{\sum{{\sigma_i}^2}}/n
        mean_err = np.sqrt(np.sum(np.square(sigmas)))/len(sigmas)
        # var = np.var(data)
        return np.mean(data), mean_err


def fit_histogram1d(bin_lo, bin_hi, data, model):
    """ Fit 1-D histogram data.
    
    """
    M = []

    for lo, hi in zip(bin_lo, bin_hi):
        
        ...


def get_flux_zero(band: Literal["W1", "W2", "AB"]) -> Quantity:
    if band == "W1":
        # f0 = 306.681 * u.Jy
        f0 = 309.540 * u.Jy
    elif band == "W2":
        # f0 = 170.663 * u.Jy
        f0 = 171.787 * u.Jy
    elif band == "AB":
        f0 = 3631 * u.Jy
    else:
        raise Exception(f"Error band: {band}")
    
    return f0


def mag2flux(mag: float, band: Literal["W1", "W2"]) -> Quantity:
    f0 = get_flux_zero(band)
    flux = f0 / (10**(0.4*mag))
    return flux.to(u.Jy)


def flux2mag(flux: Quantity, band: Literal["W1", "W2"]) -> float:
    f0 = get_flux_zero(band)
    mag = 2.5 * np.log10(f0/flux)
    return mag.to_value()