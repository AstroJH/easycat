import os
from os.path import join as pjoin

import unittest
import pandas as pd
import numpy as np

from easycat.lightcurve.reprocess import ReprocessFactory, WISEReprocessor

class TestWiseReprocessor(unittest.TestCase):
    def get_reprocessor(self) -> WISEReprocessor:
        reprocessor = ReprocessFactory.get(None, {
            "telescope": "WISE"
        })
        return reprocessor
    

    def test_clean_epoch(self):
        repro = self.get_reprocessor()

        # Case 1
        lcurve = pd.DataFrame(data={
            "mjd": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        lcurve = repro.clean_epoch(lcurve, max_interval=0.5)
        self.assertEqual(len(lcurve), 0)

        # Case 2
        lcurve = pd.DataFrame(data={
            "mjd": [
                1, 2, 3, 4, 5,
                10, 11, 12,
                21, 22, 23, 24, 25
            ]
        })
        lcurve = repro.clean_epoch(lcurve, max_interval=1)
        self.assertTrue(
            np.all(lcurve.mjd == np.array([1, 2, 3, 4, 5, 21, 22, 23, 24, 25]))
        )

        # Case 3
        lcurve = pd.DataFrame(data={
            "mjd": [1, 2]
        })
        lcurve = repro.clean_epoch(lcurve, max_interval=2)
        self.assertEqual(len(lcurve), 0)


    def test_filter_outliers(self):
        repro = self.get_reprocessor()

        lcurve = pd.DataFrame(data={
            "mjd": [0.0, 0.1, 0.2, 0.3, 0.4],
            "w1mag": [14.274, 14.536, 14.504, 14.592, 14.366],
            "w2mag": [13.262, 13.257, 13.374, 13.251, 13.187]
        })

        answer = pd.DataFrame(data={
            "mjd": [0.0, 0.1, 0.3],
            "w1mag": [14.274, 14.536, 14.592],
            "w2mag": [13.262, 13.257, 13.251]
        })
        output = repro.filter_outliers(lcurve, 5, 1.2)

        self.assertTrue(answer.equals(output))
        
