import numpy as np


def bin_lightcurve(time, value, error, time_lo, time_hi, mtd):
    time = np.asarray(time)
    value = np.asarray(value)
    time_lo = np.asarray(time_lo)
    time_hi = np.asarray(time_hi)
    

    res_time = np.empty_like(time_lo, np.float64)
    res_value = np.empty_like(time_lo, np.float64)
    res_error = np.empty_like(time_lo, np.float64)

    ptr = 0
    for lo, hi in zip(time_lo, time_hi):
        mask = (time >= lo) & (time < hi)

        bin_time = time[mask]
        bin_value = value[mask]
        bin_err = error[mask]

        _t, _v, _e = mtd(bin_time, bin_value, bin_err)

        res_time[ptr] = _t
        res_value[ptr] = _v
        res_error[ptr] = _e

        ptr += 1
    
    mask = np.isnan(res_time) & np.isnan(res_value) & np.isnan(res_error)
    mask = ~mask
    return res_time[mask], res_value[mask], res_error[mask]