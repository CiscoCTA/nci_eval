import pytest
import nci_eval.metrics as m
import numpy as np


class TestMetrics:
    @pytest.mark.parametrize("tpr, fpr, pp, expected", [
        (0.5, 0.01, 0.01, 0.3355),
        (0.5, 0.01, 0.001, 0.0477),
        (0.2, 0.001, 0.03, 0.8608)
    ])
    def test_precision_from_tpr_fpr(self, tpr, fpr, pp, expected):
        assert m.precision_from_tpr_fpr(tpr, fpr, pp) == pytest.approx(expected, 0.001)

    @pytest.mark.parametrize("tpr, fpr, s_tpr, s_fpr, pp, e_v, e_l, e_h", [
        (0.5, 0.01, 0, 0, 0.01, 0.3355, 0.3355, 0.3355),
        (0.5, 0.01, 0.1, 0, 0.01, 0.3355, 0.2877, 0.3773),
        (0.5, 0.01, 0.1, 0.001, 0.01, 0.3355, 0.2686, 0.4024)
    ])
    def test_precision_with_confidence_interval(self, tpr, fpr, s_tpr, s_fpr, pp, e_v, e_l, e_h):
        v, l, h = m.precision_with_confidence_interval(tpr, fpr, s_tpr, s_fpr, pp)
        assert v == pytest.approx(e_v, 0.001)
        assert l == pytest.approx(e_l, 0.001)
        assert h == pytest.approx(e_h, 0.001)

    @pytest.mark.parametrize("tpr, fpr, cv_tpr, cv_fpr, pp, e_v, e_l, e_h", [
        (0.5, 0.01, 0, 0, 0.01, 0.3355, 0.3355, 0.3355),
        (0.5, 0.01, 0.2, 0, 0.01, 0.3355, 0.2877, 0.3773),
        (0.5, 0.01, 0.2, 0.1, 0.01, 0.3355, 0.2686, 0.4024)
    ])
    def test_precision_with_confidence_interval_cv(self, tpr, fpr, cv_tpr, cv_fpr, pp, e_v, e_l, e_h):
        v, l, h = m.precision_with_confidence_interval_cv(tpr, fpr, cv_tpr, cv_fpr, pp)
        assert v == pytest.approx(e_v, 0.001)
        assert l == pytest.approx(e_l, 0.001)
        assert h == pytest.approx(e_h, 0.001)

    @pytest.mark.parametrize("tpr, fpr, s_tpr, s_fpr, e_pp, e_w", [
        (0.6, 0.001, 0.1*0.6, 0.5*0.001, 0.001047128, 0.31385),
    ])
    def test_max_width_of_precision_confidence_interval(self, tpr, fpr, s_tpr, s_fpr, e_pp, e_w):
        pp, width = m.max_width_of_precision_confidence_interval(tpr, fpr, s_tpr, s_fpr)
        pp == pytest.approx(e_pp, 0.006)
        width == pytest.approx(e_w, 0.001)

    @pytest.mark.parametrize("tpr, fpr, s_tpr, s_fpr, pp, min_conf, max_conf", [
        (0.6, 0.001, 0.1*0.6, 0.5*0.001, 0.001, 0.75, 0.76),
    ])
    def test_conf_level_for_confidence_interval(self, tpr, fpr, s_tpr, s_fpr, pp, min_conf, max_conf):
        np.random.seed(42)
        conf = m.conf_level_for_confidence_interval(tpr, fpr, s_tpr, s_fpr, pp, 3000)
        assert conf >= min_conf
        assert conf <= max_conf