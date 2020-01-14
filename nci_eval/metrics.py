import math
import numpy as np


def precision_from_tpr_fpr(tpr, fpr, positive_prior):
    """
    Computes precision for a given positive prevalence from TPR and FPR
    """
    return (positive_prior * tpr) / ((positive_prior * tpr) + ((1 - positive_prior) * fpr))

def precision_with_confidence_interval(tpr, fpr, sigma_tpr, sigma_fpr, positive_prior):
    """
    Precision with confidence interval for given positive prevalence from provided tpr, fpr, sigma_tpr, sigma_fpr.
    This is the alpha^2-interval from the paper.
    """
    val = precision_from_tpr_fpr(tpr, fpr, positive_prior)
    low = precision_from_tpr_fpr(max(tpr - sigma_tpr, 0.0000001), min(fpr + sigma_fpr, 0.999999), positive_prior)
    high = precision_from_tpr_fpr(min(tpr + sigma_tpr, 0.99999999), max(fpr - sigma_fpr, 0.0000001), positive_prior)
    return val, low, high

def precision_with_confidence_interval_cv(tpr, fpr, cv_tpr, cv_fpr, positive_prior):
    """
    Convenience method for precision_with_confidence_interval. Instead of sigmas this takes coefficients of variation.
    """
    return precision_with_confidence_interval(tpr, fpr, cv_tpr*tpr, cv_fpr*fpr, positive_prior)

def max_width_of_precision_confidence_interval(tpr, fpr, sigma_tpr, sigma_fpr):
    """
    Computes the max-width of precision alpha^2 confidence interval (UB(\eta) - LB(\eta)) across range of imbalance
    rates [10^-7, 10^-1].
    Returns tuple (argmax_positive_prevalence, interval_width)
    """
    max_delta = (-1, -1)
    for pp in np.logspace(-7, 1, 10000):
        _, l, h = precision_with_confidence_interval(tpr, fpr, sigma_tpr, sigma_fpr, pp)
        delta = (pp, h-l)
        if delta[1] > max_delta[1]:
            max_delta = delta
    return max_delta

def conf_level_for_confidence_interval(tpr, fpr, s_tpr, s_fpr, positive_prior, cnt = 2000):
    """
    The (UB(\eta) - LB(\eta)) is a alpha^2 confidence interval which means that its confidence level is at least
    alpha^2. This function estimates the exact confidence level of the interval for given positive_prior. The
    estimation is performed by repeated sampling of tpr, fpr from normal distributions with standard deviations
    s_tpr and s_fpr and means at tpr,fpr which should represent the real values. The confidence level is computed
    by creating confidence intervals around sampled values of precision computed from sampled tpr and fpr. The
    confidence is the ratio of such confidence intervals that contain the true value of precision computed from
    real tpr, fpr.
    """
    sampled = _sample_precision(tpr, fpr, s_tpr, s_fpr, positive_prior, cnt)
    v, _, _ = precision_with_confidence_interval(tpr, fpr, s_tpr, s_fpr, positive_prior)
    return sum(map(lambda x : v >= x[1] and v <= x[2], sampled)) / cnt

def _sample_precision(tpr, fpr, s_tpr, s_fpr, pp, cnt):
    """
    Assumes normal distribution in tpr and fpr with standard deviations s_tpr and s_fpr.
    Samples 'cnt' precisions from it.
    """
    res = []
    for i in range(cnt):
        tpr_samp = min(1.0, np.random.normal(tpr, s_tpr))
        fpr_samp = max(0.0000000001, np.random.normal(fpr, s_fpr))
        res.append(precision_with_confidence_interval(tpr_samp, fpr_samp, s_tpr, s_fpr, pp))
    return res
