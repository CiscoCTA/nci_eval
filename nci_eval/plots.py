import nci_eval.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC


class _PlotFromCurve(ABC):
    def __init__(self):
        self.classifiers_tprs = []
        self.classifiers_fprs = []
        self.classifiers_names = []

    def add(self, tprs, fprs, name):
        """
        Add new classifier to the plot.
        :param tprs: list of true-positive-rates for classifier working points (e.g. on ROC curve)
        :param fprs: list of false-positive-rates for classifier working points (e.g. on ROC curve)
        :param name: name of the classifier as it will be displayed in legend.
        """
        if len(tprs) != len(fprs):
            raise ValueError("len(tprs) and len(fprs) does not match for classifier: %s" % name)
        self.classifiers_tprs.append(tprs)
        self.classifiers_fprs.append(fprs)
        self.classifiers_names.append(name)
        return self

class _PlotFromOperatingPoint(ABC):
    def __init__(self):
        self.classifiers_tpr = []
        self.classifiers_fpr = []
        self.classifiers_sigma_tpr = []
        self.classifiers_sigma_fpr = []
        self.classifiers_names = []

    def add(self, tpr, fpr, name, sigma_tpr=None, sigma_fpr=None):
        """
        Add new classifier to the plot. Arguments related to sigma_tpr and sigma_fpr is set allow drawing of the
        (LB(\eta), UB(\eta)) confidence interval. More details in the paper.

        :param tpr: value of true positive rate
        :param fpr: value of false positive rate
        :param name: name of the classifier as it will be displayed in legend
        :param sigma_tpr: either value of sigma for true positive rate or None if it was not measured
        :param sigma_fpr: either value of sigma for false positive rate or None if it was not measured
        """
        self.classifiers_tpr.append(tpr)
        self.classifiers_fpr.append(fpr)
        self.classifiers_names.append(name)
        self.classifiers_sigma_tpr.append(0 if sigma_tpr is None else sigma_tpr)
        self.classifiers_sigma_fpr.append(0 if sigma_fpr is None else sigma_fpr)
        return self


class PositivePriorPraucPlot(_PlotFromCurve):
    """
    Creates plot for Precision-Recall-AUC vs Positive Prior. The plot supports multiple classifiers.

    Add them with the .add method and then draw the plot with the .plot method.
    """

    def plot(self, positive_priors=np.logspace(-7, -1, 500), figsize=(8, 3)):
        """
        :param positive_priors: specifies the points on the x-axis
        :param figsize: figure size as in matplotlib
        """
        fig = plt.figure(figsize=figsize)

        for i in range(len(self.classifiers_names)):
            tprs, fprs = self.classifiers_tprs[i], self.classifiers_fprs[i]
            aucs = [self._pr_auc(tprs, fprs, pp) for pp in positive_priors]
            plt.plot(positive_priors, aucs,
                label=self.classifiers_names[i], c=_color(i), ls=_line_style(i))

        plt.title('Impact of Positive prevalence on PR-AUC')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('PR-AUC')
        plt.semilogx()
        _set_plot_styling()
        return fig

    def _pr_auc(self, tprs, fprs, positive_prior):
            prec = [metrics.precision_from_tpr_fpr(tpr, fpr, positive_prior) for (tpr, fpr) in zip(tprs, fprs)]
            return np.trapz(prec, tprs)


class PositivePriorPrecisionPlot(_PlotFromOperatingPoint):
    """
    Creates plot for Precision vs Positive Prior (P3 curve). The plot supports multiple classifiers.

    Add them with the .add method and then draw the plot with the .plot method.
    """

    def plot(self, positive_priors=np.logspace(-7, -1, 500), figsize=(8, 3)):
        """
        :param positive_priors: specifies the points on the x-axis
        :param figsize: figure size as in matplotlib
        """
        fig = plt.figure(figsize=figsize)

        for i in range(len(self.classifiers_names)):
            self._create_ith_plot(i, positive_priors)

        plt.title('Impact of Positive prevalence on Precision')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('Precision')
        plt.semilogx()
        _set_plot_styling()
        return fig

    def _create_ith_plot(self, i, positive_priors):
        tpr, fpr = self.classifiers_tpr[i], self.classifiers_fpr[i]
        s_tpr, s_fpr = self.classifiers_sigma_tpr[i], self.classifiers_sigma_fpr[i]
        pci = [metrics.precision_with_confidence_interval(tpr, fpr, s_tpr, s_fpr, pp) for pp in positive_priors]
        plt.fill_between(positive_priors, [v[1] for v in pci], [v[2] for v in pci], facecolor='gray', alpha=0.5)
        plt.plot(positive_priors, [v[0] for v in pci],
            label='%s TPR=%.1f FPR=%g' % (self.classifiers_names[i], tpr, fpr), c=_color(i), ls=_line_style(i))


class PositivePriorF1ScorePlot(_PlotFromOperatingPoint):
    """
    Creates plot for F1-Score vs Positive Prior. The plot supports multiple classifiers.

    Add them with the .add method and then draw the plot with the .plot method.
    """

    def plot(self, positive_priors=np.logspace(-7, -1, 500), figsize=(8, 3)):
        """
        :param positive_priors: specifies the points on the x-axis
        :param figsize: figure size as in matplotlib
        """
        fig = plt.figure(figsize=figsize)

        for i in range(len(self.classifiers_names)):
            self._create_ith_plot(i, positive_priors)

        plt.title('Impact of Positive prevalence on F1 score')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('F1 score')
        plt.semilogx()
        _set_plot_styling()
        return fig

    def _create_ith_plot(self, i, positive_priors):
        tpr, fpr = self.classifiers_tpr[i], self.classifiers_fpr[i]
        s_tpr, s_fpr = self.classifiers_sigma_tpr[i], self.classifiers_sigma_fpr[i]
        f1ci = [self._f1_score_for_positive_prior_with_ci(tpr, fpr, s_tpr, s_fpr, pp) for pp in positive_priors]
        plt.fill_between(positive_priors, [v[1] for v in f1ci], [v[2] for v in f1ci], facecolor='gray', alpha=0.5)
        plt.plot(positive_priors, [v[0] for v in f1ci],
            label='%s TPR=%.1f FPR=%g' % (self.classifiers_names[i], tpr, fpr), c=_color(i), ls=_line_style(i))

    def _f1_score(self, tpr, prec):
        return 2.0 * tpr * prec / (tpr + prec)

    def _f1_score_for_positive_prior_with_ci(self, tpr, fpr, s_tpr, s_fpr, positive_prior):
        v, low, high = metrics.precision_with_confidence_interval(tpr, fpr, s_tpr, s_fpr, positive_prior)
        return self._f1_score(tpr, v), self._f1_score(tpr, low), self._f1_score(tpr, high)


class RocPlot(_PlotFromCurve):
    """
    Creates classic ROC plot with either log-scaled axis or not. The plot supports multiple classifiers.

    Add them with the .add method and then draw the plot with the .plot method.
    """

    def plot(self, log_x_axis=True, figsize=(6, 4)):
        """
        :param log_x_axis: Determines whether the x-axis will be log-scaled
            (advised for classifiers suited for imbalanced problems)
        :param figsize: figure size as in matplotlib
        """
        fig = plt.figure(figsize=figsize)

        for i in range(len(self.classifiers_names)):
            tprs, fprs = self.classifiers_tprs[i], self.classifiers_fprs[i]
            plt.plot(fprs, tprs, label=self.classifiers_names[i], c=_color(i), ls=_line_style(i))

        if log_x_axis:
            plt.semilogx()
        plt.title('ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        _set_plot_styling()
        return fig


class PrecisionRecallPlot(_PlotFromCurve):
    """
    Creates Precision-Recall curve for a given positive prior. The plot supports multiple classifiers.

    Add them with the .add method and then draw the plot with the .plot method.
    """

    def plot(self, positive_prior, figsize=(6, 4)):
        """
        :param positive_prior: desired positive prior for the plot
        :param figsize: figure size as in matplotlib
        """
        fig = plt.figure(figsize=figsize)

        for i in range(len(self.classifiers_names)):
            self._create_ith_plot(i, positive_prior)

        plt.title('Area under PR Curve at $\eta$ = %g' % (positive_prior))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        _set_plot_styling()
        return fig

    def _create_ith_plot(self, i, positive_prior):
        tprs, fprs, name = self.classifiers_tprs[i], self.classifiers_fprs[i], self.classifiers_names[i]
        prec = [metrics.precision_from_tpr_fpr(tpr, fpr, positive_prior) for (tpr, fpr) in zip(tprs, fprs)]
        auc = np.trapz(prec, tprs)
        plt.plot(tprs, prec, label="%s [AUPRC=%.2f]" % (name, auc), c=_color(i), ls=_line_style(i))


def _set_plot_styling():
    plt.legend()
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.grid(alpha=0.5)

def _line_style(i):
    styles = ['-', '--', '-.', ':']
    return styles[i % len(styles)]

def _color(i):
    style_cnt = 4
    colors = [
        "k", # black
        "b", # blue
        "g", # green
        "r", # red
        "c", # cyan
        "m", # magenta
        "y", # yellow
    ]
    return colors[(i // style_cnt) % len(colors)]
