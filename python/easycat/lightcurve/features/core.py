import numpy as np
# import pandas as pd
# from typing import Literal
# from collections import defaultdict

from scipy.optimize import minimize

# from scipy import stats
# from statsmodels.tsa.stattools import acf
# from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.special import gammainc

FRAC_PI_2 = np.pi/2
DOUBLE_PI = 2*np.pi

def weighted_average(values, errors):
    weights = 1/(errors * errors)
    return np.average(values, weights=weights)


def peak_to_peak_amplitude(values, errors):
    maxval = np.max(values)
    minval = np.min(values)
    delta = maxval - minval

    epsilon_square = np.mean(errors*errors)

    PP = delta * delta - 2 * epsilon_square

    if PP < 0:
        return 0.
    else:
        return np.sqrt(PP)


def gross_variation(values, errors=None):
    if errors is not None:
        avg = weighted_average(values, errors)
    else:
        avg = np.mean(values)
    diff = values - avg
    N = len(values)
    return np.sqrt(
        np.sum(diff*diff) / (N-1)
    )


def intrinsic_variability_amplitude(values, errors, is_weighted=True):
    epsilon_square = np.mean(errors*errors)

    if is_weighted:
        var = gross_variation(values, errors)
    else:
        var = gross_variation(values)
    
    var_square = var * var

    sigma_m_square = var_square - epsilon_square

    sigma_m = 0. if sigma_m_square <= 0 else np.sqrt(sigma_m_square)

    return sigma_m


def fractional_variability_amplitude(values, errors):
    S = intrinsic_variability_amplitude(values, errors, False)
    S_square = S * S
    epsilon_square = np.mean(errors*errors)

    F_square = (S_square - epsilon_square)/np.mean(values*values)

    if F_square < 0:
        return 0.
    else:
        return np.sqrt(F_square)


def laughlin1996(values, errors, chi2_empirical=None):
    if chi2_empirical is None:
        chi2_empirical = len(values) - 1
    
    chi2 = np.sum(
        (values - np.mean(values))**2 / errors / errors
    )

    Q = 1 - gammainc(chi2_empirical/2, chi2/2)
    return Q


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


def fit_damped_random_walk(times, values, errors, 
    log_a=np.log(0.1), log_c=np.log(1/100.0), log_sigma=np.log(0.01)):

    from celerite import GP
    from celerite import terms

    kernel = terms.RealTerm(log_a=log_a, log_c=log_c) + terms.JitterTerm(log_sigma=log_sigma)
    gp = GP(kernel, mean=np.mean(values))
    gp.compute(times, errors)

    # 定义似然函数
    def neg_log_like(params, y, gp: GP):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    # 定义参数的梯度函数
    def grad_neg_log_like(params, y, gp: GP):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y)[1]

    initial_params = gp.get_parameter_vector()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                method="L-BFGS-B", args=(values, gp))
    gp.set_parameter_vector(soln.x)

    # best_params = gp.get_parameter_vector()
    # log_a, log_c, log_sigma = best_params
    # a = np.sqrt(np.exp(log_a))
    # tau = 1/np.exp(log_c)
 
    return gp

# def assess_damped_random_walk(
#     t, y, yerr, gp: GP,
#     alpha_norm=0.02, alpha_lb=0.02, buffer=0.03, nlags=None
# ):
#     y_pred = gp.predict(y, t, return_cov=False)
#     residuals = y - y_pred
#     standardized_residuals = residuals / yerr

#     n = len(residuals)
#     if nlags is None:
#         nlags = min(20, n // 5)
    
#     results = {
#         'n': n,
#         'nlags': nlags,
#         'assessment': 'POOR',  # 默认较差
#         'details': {}
#     }

#     # 1. 正态性检验
#     norm_stat, norm_p = stats.shapiro(residuals)

#     results['details']['normality'] = {
#         'statistic': norm_stat,
#         'p_value': norm_p,
#         'pass': norm_p > alpha_norm
#     }

#     # 2. 自相关函数分析
#     acf_values = acf(residuals, nlags=nlags, fft=True)
#     ci = 1.96 / np.sqrt(n)  # 95% CI
    
#     acf_pass = np.all(np.abs(acf_values[1:]) < ci + buffer)  # ignore lag0
    
#     results['details']['acf'] = {
#         'values': acf_values,
#         'confidence_interval': ci,
#         'buffer': buffer,
#         'threshold': ci + buffer,
#         'pass': acf_pass,
#         'max_acf': np.max(np.abs(acf_values[1:])),
#         'lags_exceeding': np.where(np.abs(acf_values[1:]) >= ci + buffer)[0] + 1
#     }
    
#     # 3. Ljung-Box检验
#     lb_test = acorr_ljungbox(residuals, lags=nlags, return_df=True)
#     lb_p_values = lb_test['lb_pvalue'].values
    
#     lb_pass = np.all(lb_p_values > alpha_lb)
    
#     results['details']['ljung_box'] = {
#         'p_values': lb_p_values,
#         'threshold': alpha_lb,
#         'pass': lb_pass,
#         'min_p_value': np.min(lb_p_values),
#         'lags_below_threshold': np.where(lb_p_values <= alpha_lb)[0] + 1
#     }
    
#     # 4. 综合评估
#     norm_ok = results['details']['normality']['pass']
#     acf_ok = results['details']['acf']['pass']
#     lb_ok = results['details']['ljung_box']['pass']
    
#     # 根据文本标准判断
#     if norm_ok and (acf_ok or lb_ok):
#         results['assessment'] = 'GOOD'
#     else:
#         results['assessment'] = 'POOR'
    
#     return results