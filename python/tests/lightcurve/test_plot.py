import os
from os.path import join as pjoin

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord

from easycat.lightcurve.reprocess import ReprocessFactory, WiseReprocessor
from easycat.lightcurve.plot import plot_wiselc, plot_positions

class TestWisePlot(unittest.TestCase):
    def get_reprocessor(self) -> WiseReprocessor:
        reprocessor = ReprocessFactory.get_reprocessor(None, {
            "telescope": "WISE"
        })
        return reprocessor
    

    def read_lcurve(self, filename):
        dirpath = os.path.split(os.path.realpath(__file__))[0]
        filepath = pjoin(dirpath, "..", "data", filename)

        with fits.open(filepath) as hdul:
            lcurve = Table(hdul[1].data).to_pandas()

        return lcurve
    
    def test_plot_wiselc(self):
        lcurve = self.read_lcurve("9626-57875-0685.fits")

        _, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=(2, 1))

        repro = self.get_reprocessor()
        lcurve = repro.reprocess(lcurve)

        plot_wiselc(axs[0], lcurve)

        positions = SkyCoord(lcurve.raj2000, lcurve.dej2000, unit="deg", frame="fk5")
        pos_ref = SkyCoord(
            np.median(lcurve.raj2000),
            np.median(lcurve.dej2000),
            unit="deg", frame="fk5")
        plot_positions(axs[1], positions, pos_ref)
        plt.show()