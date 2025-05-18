import os
from os.path import join as pjoin

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from easycat.lightcurve.reprocess import ReprocessFactory, LightcurveReprocessor, WiseReprocessor, ZtfReprocessor
from easycat.lightcurve.reprocess import util

class TestDbscan(unittest.TestCase):
    def test_calc_distance(self):
        raj2000 = [0,  0, 200]
        dej2000 = [30, 29, 10]

        positions = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")
        distance = util.calc_distance(positions)

        N = len(positions)
        for i in range(0, N):
            self.assertEqual(distance[i, i], 0)

        