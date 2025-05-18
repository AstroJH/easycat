import numpy as np
from numpy import float64
from numpy.typing import NDArray

import pandas as pd
from pandas import DataFrame


def sfdata(ts:DataFrame, z:float):
    len_ts = len(ts)
    if len_ts < 2:
        return [],[],[]
    
    len_tau = int(len_ts*(len_ts-1)/2)

    t = ts["time"]
    val = ts["val"]
    err = ts["err"]

    tau   = np.empty(len_tau, dtype=np.float64)
    delta = np.empty(len_tau, dtype=np.float64)
    sigma = np.empty(len_tau, dtype=np.float64)

    for i in range(0, len_ts-1):
        span = len_ts - i
        begin_pos = int((2*(len_ts-1)+2-i)*(i-1)/2)
        end_pos = begin_pos + span

        tau[begin_pos:end_pos]   = t[i+1:] - t[i]
        delta[begin_pos:end_pos] = val[i+1:] - val[i]
        sigma[begin_pos:end_pos] = np.sqrt(err[i]**2 + err[i+1:]**2)
    
    tau/(1+z), delta, sigma

def esfdata(ts_list:list[DataFrame], redshifts):
    n_ts = len(ts_list)
    len_ts = np.array([len(ts) for ts in ts_list])
    len_tau = len_ts * (len_ts - 1) / 2
    len_result = int(np.sum(len_tau))

    tau   = np.empty(len_result, dtype=np.float64)
    delta = np.empty(len_result, dtype=np.float64)
    sigma = np.empty(len_result, dtype=np.float64)

    for i in range(0, n_ts):
        ts = ts_list[i]
        z = redshifts[i]
        begin_pos = 0
        if i > 1:
            begin_pos = np.sum(len_tau[1:i-1]) + 1
        
        end_pos = begin_pos + len_tau[i] - 1

        _tau, _delta, _sigma = sfdata(ts, z)

        tau[begin_pos:end_pos] = _tau
        delta[begin_pos:end_pos] = _delta
        sigma[begin_pos:end_pos] = _sigma
    
    return tau, delta, sigma