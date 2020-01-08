import nci_eval.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

class PositivePriorPraucPlot:
    def plot(self,
        classifiers_tprs,
        classifiers_fprs,
        classifiers_names,
        positive_priors=np.logspace(-7, -1, 500),
        figsize=(8, 3)):
        """
        Creates plot for Precision-Recall-AUC vs Positive Prior. The plot supports multiple classifiers.
        For example if we had 2 classifiers A and B we need to set the arguments:
            classifiers_tprs = [[tpr_A1, tpr_A2, tpr_A3], [tpr_B1, tpr_B2, tpr_B3]]
            classifiers_fprs = [[fpr_A1, fpr_A2, fpr_A3], [fpr_B1, fpr_B2, fpr_B3]]
            classifiers_names = ["A", "B"]
        The argument positive_priors specifies the points on the x-axis.
        """
        if len(set([len(classifiers_tprs), len(classifiers_fprs), len(classifiers_names)])) > 1:
            raise ValueError("classifiers_tpr, classifiers_fpr, classifiers_names lenghts should be equal.")

        fig = plt.figure(figsize=figsize)

        for i in range(len(classifiers_names)):
            tprs, fprs = classifiers_tprs[i], classifiers_fprs[i]
            aucs = [self._pr_auc(tprs, fprs, pp) for pp in positive_priors]
            plt.plot(positive_priors, aucs,
                label=classifiers_names[i], c=_color(i), ls=_line_style(i))

        plt.title('Impact of Positive prevalence on PR-AUC')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('PR-AUC')
        plt.semilogx()
        _set_plot_styling()
        return fig

    def _pr_auc(self, tprs, fprs, positive_prior):
            prec = [metrics.precision_from_tpr_fpr(tpr, fpr, positive_prior) for (tpr, fpr) in zip(tprs, fprs)]
            return np.trapz(prec, tprs)


class PositivePriorPrecisionPlot:
    def plot(self,
        classifiers_tpr,
        classifiers_fpr,
        classifiers_names,
        classifiers_sigma_tpr=None,
        classifiers_sigma_fpr=None,
        positive_priors=np.logspace(-7, -1, 500),
        figsize=(8, 3)):
        """
        Creates plot for Precision vs Positive Prior (P3 curve). The plot supports multiple classifiers.
        For example if we had 2 classifiers A and B we need to set the arguments:
            classifiers_tpr = [tpr_A, tpr_B]
            classifiers_fpr = [fpr_A, fpr_B]
            classifiers_sigma_tpr = [sigma_tpr_A, sigma_tpr_B] or None
            classifiers_sigma_fpr = [sigma_fpr_A, sigma_fpr_B] or None
            classifiers_names = ["A", "B"]

        The argument positive_priors specifies the points on the x-axis.

        Arguments related to sigma_tpr and sigma_fpr is set allow drawing of the alpha^2 confidence interval.
        More details in the paper.
        """
        if len(set([len(classifiers_tpr), len(classifiers_fpr), len(classifiers_names)])) > 1:
            raise ValueError("classifiers_tpr, classifiers_fpr, classifiers_names lenghts should be equal.")
        if classifiers_sigma_tpr is not None and len(classifiers_sigma_tpr) != len(classifiers_names):
            raise ValueError("classifiers_sigma_tpr should be None or it's length must be equal" +
                " to the number of classifiers. If unknown for a specific classifier 0 value can be used.")
        if classifiers_sigma_fpr is not None and len(classifiers_sigma_fpr) != len(classifiers_names):
            raise ValueError("classifiers_sigma_fpr should be None or it's length must be equal" +
                " to the number of classifiers. If unknown for a specific classifier 0 value can be used.")

        if classifiers_sigma_tpr is None:
            classifiers_sigma_tpr = [0] * len(classifiers_names)
        if classifiers_sigma_fpr is None:
            classifiers_sigma_fpr = [0] * len(classifiers_names)

        fig = plt.figure(figsize=figsize)

        for i in range(len(classifiers_names)):
            tpr, fpr = classifiers_tpr[i], classifiers_fpr[i]
            s_tpr, s_fpr = classifiers_sigma_tpr[i], classifiers_sigma_fpr[i]
            pci = [metrics.precision_with_confidence_interval(tpr, fpr, s_tpr, s_fpr, pp) for pp in positive_priors]
            plt.plot(positive_priors, [v[0] for v in pci],
                label='%s TPR=%.1f FPR=%g' % (classifiers_names[i], tpr, fpr), c=_color(i), ls=_line_style(i))
            plt.fill_between(positive_priors, [v[1] for v in pci], [v[2] for v in pci], facecolor='gray', alpha=0.5)

        plt.title('Impact of Positive prevalence on Precision')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('Precision')
        plt.semilogx()
        _set_plot_styling()
        return fig


class PositivePriorF1ScorePlot:
    def plot(self,
        classifiers_tpr,
        classifiers_fpr,
        classifiers_names,
        positive_priors=np.logspace(-7, -1, 500),
        figsize=(8, 3)):
        """
        Creates plot for F1 score vs Positive Prior. The plot supports multiple classifiers.
        For example if we had 2 classifiers A and B we need to set the arguments:
            classifiers_tpr = [tpr_A, tpr_B]
            classifiers_fpr = [fpr_A, fpr_B]
            classifiers_names = ["A", "B"]
        The argument positive_priors specifies the points on the x-axis.
        """
        if len(set([len(classifiers_tpr), len(classifiers_fpr), len(classifiers_names)])) > 1:
            raise ValueError("classifiers_tpr, classifiers_fpr, classifiers_names lenghts should be equal.")

        fig = plt.figure(figsize=figsize)

        for i in range(len(classifiers_names)):
            tpr, fpr = classifiers_tpr[i], classifiers_fpr[i]
            f1 = [self._f1_score_for_positive_prior(tpr, fpr, pp) for pp in positive_priors]
            plt.plot(positive_priors, f1,
                label='%s TPR=%.1f FPR=%g' % (classifiers_names[i], tpr, fpr), c=_color(i), ls=_line_style(i))

        plt.title('Impact of Positive prevalence on F1 score')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('F1 score')
        plt.semilogx()
        _set_plot_styling()
        return fig

    def _f1_score(self, tpr, prec):
        return 2.0 * tpr * prec / (tpr + prec)

    def _f1_score_for_positive_prior(self, tpr, fpr, positive_prior):
        return self._f1_score(tpr, metrics.precision_from_tpr_fpr(tpr, fpr, positive_prior))


class RocPlot:
    def plot(self,
        classifiers_tprs,
        classifiers_fprs,
        classifiers_names,
        log_x_axis=True,
        figsize=(6, 4)):
        """
        Creates classic ROC plot with either log-scaled axis or not. The plot supports multiple classifiers.

        For example if we had 2 classifiers A and B we need to set the arguments:
            classifiers_tprs = [[tpr_A1, tpr_A2, tpr_A3], [tpr_B1, tpr_B2, tpr_B3]]
            classifiers_fprs = [[fpr_A1, fpr_A2, fpr_A3], [fpr_B1, fpr_B2, fpr_B3]]
            classifiers_names = ["A", "B"]
        """
        if len(set([len(classifiers_tprs), len(classifiers_fprs), len(classifiers_names)])) > 1:
            raise ValueError("classifiers_tprs, classifiers_fprs, classifiers_names lenghts should be equal.")

        fig = plt.figure(figsize=figsize)

        for i in range(len(classifiers_names)):
            tprs, fprs = classifiers_tprs[i], classifiers_fprs[i]
            if len(tprs) != len(fprs):
                raise ValueError("len(tprs) and len(fprs) does not match for classifier: %s" % (classifiers_names[i]))
            plt.plot(fprs, tprs, label=classifiers_names[i], c=_color(i), ls=_line_style(i))
        if log_x_axis:
            plt.semilogx()
        plt.title('ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        _set_plot_styling()
        return fig


class PrecisionRecallPlot:
    def plot(self,
        classifiers_tprs,
        classifiers_fprs,
        classifiers_names,
        positive_prior,
        figsize=(6, 4)):
        """
        Creates classic ROC plot with either log-scaled axis or not. The plot supports multiple classifiers.

        For example if we had 2 classifiers A and B we need to set the arguments:
            positive_prior = desired_prior
            classifiers_tprs = [[tpr_A1, tpr_A2, tpr_A3], [tpr_B1, tpr_B2, tpr_B3]]
            classifiers_fprs = [[fpr_A1, fpr_A2, fpr_A3], [fpr_B1, fpr_B2, fpr_B3]]
            classifiers_names = ["A", "B"]
        """
        if len(set([len(classifiers_tprs), len(classifiers_fprs), len(classifiers_names)])) > 1:
            raise ValueError("classifiers_tprs, classifiers_fprs, classifiers_names lenghts should be equal.")

        fig = plt.figure(figsize=figsize)

        for i in range(len(classifiers_names)):
            tprs, fprs = classifiers_tprs[i], classifiers_fprs[i]
            if len(tprs) != len(fprs):
                raise ValueError("len(tprs) and len(fprs) does not match for classifier: %s" % (classifiers_names[i]))
            prec = [metrics.precision_from_tpr_fpr(tpr, fpr, positive_prior) for (tpr, fpr) in zip(tprs, fprs)]
            auc = np.trapz(prec, tprs)
            plt.plot(tprs, prec, label="%s [AUPRC=%.2f]" % (classifiers_names[i], auc), c=_color(i), ls=_line_style(i))

        plt.title('Area under PR Curve at $\eta$ = %g' % (positive_prior))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        _set_plot_styling()
        return fig

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
