import unittest

from easycat.stats import montecarlo
import matplotlib.pyplot as plt
import numpy as np

class TestSpearmanr(unittest.TestCase):
    def test_calc_spearmanr(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 6, 8, 9, 10, 11, 17, 20, 21, 20])
        xerr = np.array([0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.1, 0.5, 0.4])
        yerr = np.array([0.4, 0.1, 0.3, 0.3, 0.25, 0.2, 0.4, 0.5, 0.5, 0.6])

        rho, z = montecarlo.spearmanr_mc(x, y, xerr, yerr, N=1000, method="resampling")