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
    Computes the max-width of precision alpha^2 confidence interval across range of imbalance rates [10^-7, 10^-1].
    Returns tuple (argmax_positive_prevalence, interval_width)
    """
    max_delta = (-1, -1)
    for pp in np.logspace(-7, 1, 10000):
        _, l, h = precision_with_confidence_interval(tpr, fpr, sigma_tpr, sigma_fpr, pp)
        delta = (pp, h-l)
        if delta[1] > max_delta[1]:
            max_delta = delta
    return max_delta
