import unittest

from astropy.coordinates import SkyCoord
from easycat.util import dbscan

class TestDbscan(unittest.TestCase):
    def test_calc_distance(self):
        raj2000 = [0,  0, 200]
        dej2000 = [30, 29, 10]

        positions = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")
        distance = dbscan.calc_distance(positions)

        N = len(positions)
        for i in range(0, N):
            self.assertEqual(distance[i, i], 0)
