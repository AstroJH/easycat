from scipy import stats
from . import montecarlo

def ftest(chi1, chi2, dof1, dof2):
    F = ((chi1-chi2)/(dof1-dof2))/(chi2/dof2)

    p_value = stats.f.cdf(F, dof1-dof2, dof2)

    return F, 1-p_value


__all__ = [
    "ftest", "montecarlo"
]