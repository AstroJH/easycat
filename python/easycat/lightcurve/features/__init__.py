import numpy as np
import pandas as pd
from typing import Literal

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
