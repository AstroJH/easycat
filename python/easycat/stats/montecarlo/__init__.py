import numpy as np
from scipy import stats
from typing import Literal

def perturb(x, xerr):
    N = len(x)
    G = np.random.normal(loc=0, scale=1, size=N)

    return x + G * xerr


def spearmanr_mc(x, y, xerr, yerr,
    N=1000,
    method:Literal["perturbation", "resampling", "composite"]="composite"):

    size = len(x)
    data = np.arange(size)

    rho = np.empty(N)
    zscore = np.empty(N)

    for i in range(N):
        if method in ["composite", "resampling"]:
            indice = np.random.choice(data, size, replace=True)
            newx = x[indice]
            newy = y[indice]
            newxerr = xerr[indice]
            newyerr = yerr[indice]
        else:
            newx = x
            newy = y
            newxerr = xerr
            newyerr = yerr
        
        if method in ["composite", "perturbation"]:
            newx = perturb(newx, newxerr)
            newy = perturb(newy, newyerr)

        r = stats.spearmanr(newx, newy).statistic
        z = np.sqrt((size-3)/1.06) * np.log((1+r)/(1-r))/2 # BUG if r = 1 or -1

        rho[i] = r
        zscore[i] = z
    
    return rho, zscore


__all__ = ["spearmanr_mc"]