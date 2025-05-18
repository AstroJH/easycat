import os
from os.path import join as pjoin

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from easycat.lightcurve.reprocess import ReprocessFactory, LightcurveReprocessor, WiseReprocessor, ZtfReprocessor


class TestWiseReprocessor(unittest.TestCase):
    def read_lcurve(self, filename):
        dirpath = os.path.split(os.path.realpath(__file__))[0]
        filepath = pjoin(dirpath, "..", "data", filename)

        lcurve = pd.read_csv(filepath, dtype={
            "cc_flags": str,
            "moon_masked": str
        })
        return lcurve


    def test_criteria_basic(self):
        reprocessor: WiseReprocessor = ReprocessFactory.get_reprocessor(None, {
            "telescope": "WISE"
        })

        # lcurve = self.read_lcurve("test_wise_criteria_basic.csv")
        # lcurve = reprocessor.criteria_basic(lcurve)

        # lcurve_correct = self.read_lcurve("test_wise_criteria_basic_correct.csv")
        # lcurve_correct = reprocessor.criteria_basic(lcurve_correct)
