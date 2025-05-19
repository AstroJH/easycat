import numpy as np
import pandas as pd

def amplitude(lcurve, valname, errname, err_sys=0):
    vals = lcurve[valname]
    errs = lcurve[errname]

    maxval = np.max(vals)
    minval = np.min(vals)

    maxval - minval


def sigma_m(lcurve, colname):
    vals = lcurve[colname]
    mu = np.mean(vals)

