# nci_eval

**Supplementary code and materials for paper "On Model Evaluation under Non-constant Class Imbalance"**

## Abstract

Many real-world classification problems are significantly class-imbalanced to detriment of the class of interest. The standard set of proper evaluation metrics is well-known but the usual assumption is that the test dataset imbalance equals the real-world imbalance. In practice, this assumption is often broken for various reasons. The reported results are then often too optimistic and may lead to wrong conclusions about industrial impact and suitability of proposed techniques. We introduce methods focusing on evaluation under non-constant class imbalance. We show that not only the absolute values of commonly used metrics, but even the order of classifiers in relation to the evaluation metric used is affected by the change of the imbalance rate. Finally, we demonstrate that using subsampling in order to get a test dataset with class imbalance equal to the one observed in the wild is not necessary, and eventually can lead to significant errors in classifier's performance estimate.

## Repository structure

The repository is organized in the following way:

- nci_eval: Library code containing python implementation of metrics and plots in the paper.
  - metrics.py: Contains implementation of the metrics including confidence intervals where applicable. Also includes a Monte Carlo method for estimation of the exact confidence level of the UB(\eta) - LB(\eta) interval.
  - plots.py: Contains classes that are able to draw the plots that are in the paper and that might be also useful for evaluation of your classifiers. Although the plots allow some level of customization it is possible that you might want to customize your plots differently. In that case, you can use the code as inspiration for your own or if it would make sense to extend the methods create an Issue or pull request.
- notebooks: Contains examples of the uses of metrics and plots in the library.
  - plot_pr_auc.ipynb: Example of PositivePrevalence-PR-AUC plot. Also contains ROC curves and PR curves examples.
  - positive_prior_vs_f1_score_plot.ipynb: Example of PositivePrevalence-F1Score plot and P3 plots. Also contains example of P3 plot with confidence interval and the computation for the max width of confidence interval mentioned in the paper in footnote 8.
  - plot_subsampling_imagenet.ipynb: Code for the experiment in paper section 5.1 with ImageNet, ResNet-50 and the effects of subsampling on the evaluation.
- test: Tests for the code in nci_eval dir. Plots are not tested here, but for their usecases see the notebooks.

## Setup & Example

-- TODO: prepare pip package and example

## Contacts

Jan Brabec (janbrabe@cisco.com), Cisco Cognitive Intelligence

Tomas Komarek (tomkomar@cisco.com), Cisco Cognitive Intelligence
