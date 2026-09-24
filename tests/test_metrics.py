import numpy as np
import pytest

from otto_recommender.metrics import aggregate_recall, weighted_recall


def test_aggregate_recall_and_weighting():
    recalls = aggregate_recall([5, 6, 8], [10, 12, 10])
    np.testing.assert_allclose(recalls, [0.5, 0.5, 0.8])
    assert weighted_recall(recalls) == pytest.approx(0.68)


def test_zero_denominator_is_safe():
    recalls = aggregate_recall([0, 1, 0], [0, 2, 0])
    np.testing.assert_allclose(recalls, [0.0, 0.5, 0.0])


def test_invalid_metric_shapes_fail():
    with pytest.raises(ValueError):
        weighted_recall([0.5, 0.6])
