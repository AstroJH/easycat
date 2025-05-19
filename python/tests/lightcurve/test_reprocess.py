import os
from os.path import join as pjoin

import unittest
import pandas as pd

from easycat.lightcurve.reprocess import ReprocessFactory, WiseReprocessor

class TestWiseReprocessor(unittest.TestCase):
    def get_reprocessor(self) -> WiseReprocessor:
        reprocessor = ReprocessFactory.get_reprocessor(None, {
            "telescope": "WISE"
        })
        return reprocessor
    

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

    def test_filter_outliers(self):
        reprocessor = self.get_reprocessor()

        lcurve = pd.DataFrame(data={
            "mjd":   [0.0, 0.1, 0.2, 0.3, 0.4],
            "w1mag": [14.274, 14.536, 14.504, 14.592, 14.366],
            "w2mag": [13.262, 13.257, 13.374, 13.251, 13.187]
        })

        answer = pd.DataFrame(data={
            "mjd":   [0.0, 0.1, 0.3],
            "w1mag": [14.274, 14.536, 14.592],
            "w2mag": [13.262, 13.257, 13.251]
        })
        output = reprocessor.filter_outliers(lcurve, 5, 1.2)

        self.assertTrue(answer.equals(output))
        
