import os
from os.path import join as pjoin

import unittest
import pandas as pd
import numpy as np
import time

from easycat.stats import sf

def generate_lightcurve(n_points, time_scale=100.0, amplitude=1.0, noise_level=0.1):
    t = np.sort(np.random.uniform(0, time_scale, n_points))
    val = amplitude * np.sin(2 * np.pi * t / time_scale)
    val += np.random.normal(0, noise_level, n_points)
    err = np.full(n_points, noise_level)
    return t, val, err

class TestStructureFunction(unittest.TestCase):

    def test_sfdata(self):
        t = np.array([20, 30, 50, 90, 100])
        val = np.array([3, 1, 9, 4, 6])
        err = np.array([1, 3, 1, 2, 1])

        tau, dval, sig = sf.sfdata(t, val, err, 0)

        self.assertTrue(
            np.all(
                tau == \
                np.array([
                    10, 30, 70, 80, 20, 60, 70, 40, 50, 10
                ])
            )
        )

        self.assertTrue(
            np.all(
                dval == \
                np.array([
                    -2, 6, 1, 3, 8, 3, 5, -5, -3, 2
                ])
            )
        )

        self.assertTrue(
            np.all(
                sig == \
                np.array([
                    np.sqrt(10), np.sqrt(2), np.sqrt(5), np.sqrt(2),
                    np.sqrt(10), np.sqrt(13), np.sqrt(10),
                    np.sqrt(5), np.sqrt(2), np.sqrt(5)
                ])
            )
        )
    
    def test_esfdata(self):
        t_list = [
            np.array([20, 30, 50, 90, 100]),
            np.array([10, 50, 60, 65, 90, 105, 200]),
        ]

        val_list = [
            np.array([3, 1, 9, 4, 6]),
            np.array([1, 1, 4, 6, 3, 2, 5])
        ]

        err_list = [
            np.array([1, 3, 1, 2, 1]),
            np.array([2, 1, 1, 1, 2, 3, 1])
        ]

        tau, dval, sig = sf.esfdata(t_list, val_list, err_list)
        
        self.assertTrue(
            np.all(
                tau == \
                np.array([
                    10, 30, 70, 80, 20, 60, 70, 40, 50, 10, 40, 50, 55, 80,
                    95, 190, 10, 15, 40, 55, 150,  5, 30, 45, 140, 25, 40, 135,
                    15, 110, 95
                ])
            )
        )
    
    def test_speed(self):
        num_curves = 70
        # 生成测试数据
        t_list, val_list, err_list = [], [], []
        for _ in range(num_curves):
            t, val, err = generate_lightcurve(100)
            t_list.append(t)
            val_list.append(val)
            err_list.append(err)

        redshifts = [0.0] * num_curves

        start = time.time()
        sf.esfdata(t_list, val_list, err_list, redshifts)
        end = time.time()

        print(f"Python: {end-start:.3f}")

        start = time.time()
        rust_sf.esfdata(t_list, val_list, err_list, redshifts)
        end = time.time()

        print(f"Rust: {end-start:.3f}")