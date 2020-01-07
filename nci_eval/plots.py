import nci_eval.metrics as metrics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class PositivePriorF1ScorePlot:
    def plot(self, tpr1, fpr1, tpr2, fpr2, positive_priors):
        classifier1_f1 = [self._f1_score_for_positive_prior(tpr1, fpr1, pp) for pp in positive_priors]
        classifier2_f1 = [self._f1_score_for_positive_prior(tpr2, fpr2, pp) for pp in positive_priors]

        plt.plot(positive_priors, classifier1_f1,
            label='Classifier 1 TPR=%.1f FPR=%g' % (tpr1, fpr1), c='k')
        plt.plot(positive_priors, classifier2_f1,
            label='Classifier 2 TPR=%.1f FPR=%g' % (tpr2, fpr2), c='k', linestyle='--')

        plt.title('Impact of Positive prevalence on F1 score')
        plt.xlabel('Positive prevalence ($\eta$)')
        plt.ylabel('F1 score')
        plt.semilogx()
        plt.legend()
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

    def _f1_score(self, tpr, prec):
        return 2.0 * tpr * prec / (tpr + prec)

    def _f1_score_for_positive_prior(self, tpr, fpr, positive_prior):
        return self._f1_score(tpr, metrics.precision_from_tpr_fpr(tpr, fpr, positive_prior))