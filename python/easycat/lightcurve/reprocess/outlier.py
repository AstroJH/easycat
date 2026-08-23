from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


OutlierMask = NDArray[np.bool_]

class OutlierDetector(Protocol):
    """Protocol for outlier detection methods."""

    def detect(self, values: ArrayLike) -> OutlierMask:
        """Return a boolean mask identifying outliers."""
        ...


class MADDetector:
    """
    Outlier detector based on the median absolute deviation (MAD).

    The robust standard deviation is estimated as

        sigma = 1.4826 * MAD

    and values farther than ``threshold * sigma`` from the
    median are identified as outliers.
    """

    def __init__(self, threshold: float = 5.0):
        if threshold <= 0:
            raise ValueError(
                "MAD threshold must be positive"
            )

        self.threshold = threshold

    def detect(
        self,
        values: ArrayLike,
    ) -> OutlierMask:

        values = np.asarray(values)
        valid = np.isfinite(values) # Ignore NaN.

        outliers = np.zeros(len(values), dtype=bool)

        if not np.any(valid): # No valid values.
            return outliers
        
        valid_values = values[valid]

        median = np.median(valid_values)
        mad = np.median( # MAD: Median Absolute Deviation
            np.abs(valid_values - median)
        )

        # No measurable scatter.
        if mad == 0:
            return outliers

        # MAD to STD conversion (assuming a Gaussian distribution)
        std_est = 1.4826 * mad

        outliers[valid] = (
            np.abs(valid_values - median)
            > self.threshold * std_est
        )

        return outliers


class IQRDetector:
    """
    Outlier detector based on the interquartile range (IQR).
    """

    def __init__(self, multiplier: float = 1.5):
        if multiplier <= 0:
            raise ValueError(
                "IQR multiplier must be positive"
            )

        self.multiplier = multiplier

    def detect(
        self,
        values: ArrayLike,
    ) -> OutlierMask:

        values = np.asarray(values)
        valid = np.isfinite(values) # Ignore NaN.

        outliers = np.zeros(len(values), dtype=bool)

        if not np.any(valid): # No valid values.
            return outliers
        
        valid_values = values[valid]

        q1, q3 = np.percentile(
            valid_values,
            [25, 75],
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - self.multiplier * iqr
        )

        upper_bound = (
            q3 + self.multiplier * iqr
        )

        outliers[valid] = (
            (valid_values < lower_bound)
            | (valid_values > upper_bound)
        )

        return outliers
