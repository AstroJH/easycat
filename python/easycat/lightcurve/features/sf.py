import numpy as np
import pandas as pd

def sfdata(t, val, err, z: float=0.):
    len_ts = len(t)

    if len_ts < 2:
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

def sfmtd_default(tau, delta, sigma, n_least: int=5):
    if len(tau) <= n_least:
        return np.nan, np.nan, np.nan
    
    sftau = np.mean(tau)

    delta_f2 = delta**2 - sigma**2
    sf2 = np.mean(delta_f2)
    sf2_std = np.std(delta_f2, ddof=1)

    if sf2 < 0:
        return np.nan, np.nan, np.nan
    else:
        return sftau, np.sqrt(sf2), np.sqrt(sf2_std)

def calc_sf(
    tau, delta, sigma,
    bin_lo, bin_hi,
    sfmtd=sfmtd_default
):
    """
    Compute the structure function (SF) of an individual target.
    SF is defined as the root mean square of the magnitudes or flux at a given time difference.
    """
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

def calc_esf(
    files, redshifts,
    tau_lo, tau_hi,
    timename, valuename, errname
):
    dt_s = np.array([])
    sf_s = np.array([])
    sferr_s = np.array([])

    for file, z in zip(files, redshifts):
        lc = pd.read_csv(file)
        tau, delta, sigma = sfdata(lc[timename], lc[valuename], lc[errname], z)
        dt, sf, sferr = calc_sf(tau, delta, sigma, tau_lo, tau_hi)
        dt_s = np.concatenate([dt_s, dt])
        sf_s = np.concatenate([sf_s, sf])
        sferr_s = np.concatenate([sferr_s, sferr])
    
    tau = []
    sf = []
    err = []
    num = []
    for lo, hi in zip(tau_lo, tau_hi):
        mask = (dt_s >= lo) & (dt_s <= hi)
        if np.sum(mask) == 0:
            continue

        _dt = dt_s[mask]
        _sf = sf_s[mask]
        _sferr = sferr_s[mask]

        tau.append(np.nanmean(_dt))
        # sf.append(np.sqrt(np.nanmean(_sf*_sf)))
        sf.append(np.nanmean(_sf))
        err.append(
            np.sqrt(np.sum(_sferr*_sferr))/len(_sf)
        )
        num.append(len(_sf))
    return np.array(tau), np.array(sf), np.array(err), np.array(num)