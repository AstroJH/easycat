import numpy as np
import pandas as pd
from typing import List
import os
from astropy.io import fits
from astropy.table import Table

def sfmodel(t, A, beta, tau):
    return A*np.sqrt(1-np.exp(-(t/tau)**beta))

def brokenpl(t, gamma1, gamma2, A, t_b):
    conds = t < t_b
    res = [A*(_t/365)**gamma1 if c else A*(t_b/365)**(gamma1-gamma2) * (_t/365)**gamma2 for (_t, c) in zip(t, conds)]
    return np.array(res)

def powerlaw(t, gamma, A):
    return A*(t/365)**gamma

def sfdata(t, val, err, z:float=0.):
    len_ts = len(t)

    if len_ts < 2:
        # tmp = lambda a: [np.float64(x) for x in range(0)]
        # return tmp(None), tmp(None), tmp(None)
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    len_tau = int(len_ts*(len_ts-1)/2)

    tau = np.empty(len_tau, dtype=np.float64)
    delta = np.empty(len_tau, dtype=np.float64)
    sigma = np.empty(len_tau, dtype=np.float64)

    for i in range(len_ts-1):
        span = len_ts - 1 - i
        begin = i * len_ts - int(i*(i+1)/2)
        end = begin + span

        tau[begin:end]   = t[i+1:] - t[i]
        delta[begin:end] = val[i+1:] - val[i]
        sigma[begin:end] = np.sqrt(err[i]**2 + err[i+1:]**2)

    return tau/(1+z), delta, sigma


def esfdata(lcurves:List[pd.DataFrame], redshifts, time_name, val_name, err_name):
    t_list = []
    val_list = []
    err_list = []

    for lc in lcurves:
        t_list.append(lc[time_name])
        val_list.append(lc[val_name])
        err_list.append(lc[err_name])
    
    return _esfdata(t_list, val_list, err_list, redshifts)


def _esfdata(t_list, val_list, err_list, redshifts:List|float=0.):
    if isinstance(redshifts, float):
        redshifts = np.full(len(t_list), redshifts)

    size_ts = np.array([len(t) for t in t_list])
    size_tau = size_ts*(size_ts-1) >> 1
    len_result = np.sum(size_tau)

    tau = np.empty(len_result)
    delta = np.empty(len_result)
    sigma = np.empty(len_result)

    # traverse each lightcurve
    begin = 0
    for ele_size_tau, t, val, err, z in \
        zip(size_tau, t_list, val_list, err_list, redshifts):
        end = begin + ele_size_tau

        _tau, _delta, _sigma = sfdata(t, val, err, z)
        tau[begin:end] = _tau
        delta[begin:end] = _delta
        sigma[begin:end] = _sigma

        begin = end
    
    return tau, delta, sigma


def sfmtd_default(tau, delta, sigma):
    if len(tau) == 0:
        return np.nan, np.nan, np.nan
    
    sftau = np.median(tau)
    sf = np.sqrt(np.pi/2 * np.mean(np.abs(delta))**2 - np.mean(sigma*sigma))
    return sftau, sf, np.nan


def calc_sf(
    tau, delta, sigma, bin_lo, bin_hi, sfmtd=sfmtd_default
):
    len_sf = len(bin_lo)
    res_tau = np.empty(len_sf)
    res_sf = np.empty(len_sf)
    res_sigma = np.empty(len_sf)

    for (i, lo, hi) in zip(range(len_sf), bin_lo, bin_hi):
        mask = (tau >= lo) & (tau <= hi)

        _tau = tau[mask]
        _delta = delta[mask]
        _sigma = sigma[mask]

        result = sfmtd(_tau, _delta, _sigma)

        res_tau[i], res_sf[i], res_sigma[i] = result
    
    mask = res_tau >= 0
    return res_tau[mask], res_sf[mask], res_sigma[mask]
