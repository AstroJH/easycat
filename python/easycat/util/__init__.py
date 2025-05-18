import numpy as np
from scipy import stats

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


def ftest(chi1, chi2, dof1, dof2):
    F = ((chi1-chi2)/(dof1-dof2))/(chi2/dof2)

    p_value = stats.f.cdf(F, dof1-dof2, dof2)

    return F, 1-p_value


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
    
    if len(data) != len(sigmas):
        raise Exception("len(data) and len(sigmas) must be same.")

    # \sigma = \sqrt{\sum{{\sigma_i}^2}}/n
    return np.mean(data), np.sqrt(np.sum(np.square(sigmas)))/len(sigmas)