import numpy as np
import pandas as pd
from typing import Literal
from collections import defaultdict

def amplitude(lcurve, valname, errname, err_sys=0):
    vals = lcurve[valname]
    errs = lcurve[errname]

    maxval = np.max(vals)
    minval = np.min(vals)

    maxval - minval


def sigma_m(lcurve, valname, errname, err_sys=0., redshift=0.):
    vals = lcurve[valname]
    errs = lcurve[errname]

    epsilon_square = np.mean(errs**2) + err_sys**2
    epsilon = np.sqrt(epsilon_square)

    sigma_m_square = np.var(vals, ddof=1) - epsilon_square
    sigma_m = 0 if sigma_m_square <= 0 else np.sqrt(sigma_m_square)
    
    return sigma_m*np.sqrt(1+redshift), sigma_m/epsilon, epsilon


def wisesf(lcurve, timename, valname, bin_size=0.5):
    times = lcurve[timename]
    magnitudes = lcurve[valname]
    # error = lcurve[errname]

    n = len(times)
    
    # 存储每个时间差对应的平方差值
    delta_mag = defaultdict(list)
    
    # 计算所有点对的时间差和平方差值
    for i in range(n):
        for j in range(i+1, n):
            # 计算时间差
            tau = abs(times[j] - times[i])/365
            
            # 将时间差归算到最近的bin_size倍数
            tau_binned = round(tau / bin_size) * bin_size
            
            # 计算差值
            dm = np.abs(magnitudes[i] - magnitudes[j])
            
            # 存储结果
            delta_mag[tau_binned].append(dm)
    
    # 计算每个时间差对应的平均平方差值
    structure_function = {}
    for tau, values in delta_mag.items():
        structure_function[tau] = np.mean(values)
    
    return structure_function

