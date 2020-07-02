# -*- coding: utf-8 -*-

# License: GPL 3.0

import pickle
from os import path

import numpy as np
import numpy.testing as npt

from bhc import (BayesianHierarchicalClustering,
                 BayesianRoseTrees,
                 NormalInverseWishart)

BASE_PATH = path.abspath(path.dirname(path.abspath(__file__)))
SNAPSHOTS_PATH = path.join(BASE_PATH, 'snapshots')

DATA_EXAMPLE = np.array([[1.22474747, 1.90095847],
                         [1.67929293, 2.57188498],
                         [2.23484848, 3.40255591],
                         [2.96717172, 4.36102236],
                         [3.67424242, 5.28753994],
                         [4.17929293, 6.40575080],
                         [4.93686869, 6.94888179],
                         [5.69444444, 8.32268371],
                         [6.09848485, 9.15335463],
                         [6.75505051, 9.60063898],
                         [7.31060606, 3.75399361],
                         [6.88131313, 4.07348243],
                         [6.50252525, 4.23322684],
                         [6.04797980, 4.55271565],
                         [5.71969697, 4.74440895],
                         [5.29040404, 4.93610224],
                         [4.91161616, 5.15974441],
                         [2.58838384, 6.50159744]])


def test_bhc():
    """
    Simple test to validate that the result obtained from the
        current BHC implementation matches the expected snapshot.
    """
    model = NormalInverseWishart.create(DATA_EXAMPLE,
                                        g=20,
                                        scale_factor=0.001)

    bhc_result = BayesianHierarchicalClustering(DATA_EXAMPLE,
                                                model,
                                                alpha=1,
                                                cut_allowed=True).build()

    assert_result(path.join(SNAPSHOTS_PATH, 'bhc_example.obj'),
                  bhc_result)


def test_brt():
    """
    Simple test to validate that the result obtained from the
        current BRT implementation matches the expected snapshot.
    """
    model = NormalInverseWishart.create(DATA_EXAMPLE,
                                        g=10,
                                        scale_factor=0.001)

    brt_result = BayesianRoseTrees(DATA_EXAMPLE,
                                   model,
                                   alpha=0.5,
                                   cut_allowed=True).build()

    assert_result(path.join(SNAPSHOTS_PATH, 'brt_example.obj'),
                  brt_result)


def assert_result(file_path, result):
    with open(file_path, "rb") as f:
        expected_result = pickle.load(f)

        npt.assert_array_equal(
            result.arc_list, expected_result.arc_list)

        npt.assert_array_equal(
            result.node_ids, expected_result.node_ids)

        npt.assert_almost_equal(
            result.last_log_p, expected_result.last_log_p, decimal=10)

        npt.assert_almost_equal(
            result.weights, expected_result.weights, decimal=10)

        assert result.hierarchy_cut == expected_result.hierarchy_cut

        assert result.n_clusters == expected_result.n_clusters
