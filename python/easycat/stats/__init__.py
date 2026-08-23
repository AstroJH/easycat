from scipy import stats
from . import montecarlo
import numpy as np

def ftest(chi1, chi2, dof1, dof2):
    F = ((chi1-chi2)/(dof1-dof2))/(chi2/dof2)

    p_value = stats.f.cdf(F, dof1-dof2, dof2)

    return F, 1-p_value

def rchi2(model, data, error, k=0):
    z = (data - model)/error
    chi2 = np.sum(z*z)
    dof = len(data) - k
    return chi2/dof


__all__ = [
    "ftest", "rchi2", "montecarlo"
]