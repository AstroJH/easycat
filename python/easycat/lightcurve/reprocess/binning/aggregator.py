from typing import Protocol, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

class Aggregator(Protocol):

    def aggregate(
        self,
        values: ArrayLike,
        errors: Optional[ArrayLike] = None
    ) -> Tuple[float, float]:
        ...

class MeanAggregator:

    def aggregate(
        self,
        values: ArrayLike,
        errors: Optional[ArrayLike] = None,
    ) -> Tuple[float, float]:

        values = np.asarray(values, dtype=float)

        valid = np.isfinite(values)

        if errors is not None:
            errors = np.asarray(errors, dtype=float)
            valid &= np.isfinite(errors)

        values = values[valid]

        if errors is not None:
            errors = errors[valid]

        N = len(values)

        if N == 0:
            return np.nan, np.nan

        value = np.mean(values)

        if errors is not None:
            error = np.sqrt(
                np.sum(errors**2)
            ) / N

        else:
            error = np.std(
                values,
                ddof=1,
            ) / np.sqrt(N)

        return value, error

class MedianAggregator:

    def aggregate(
        self,
        values: ArrayLike,
        errors: Optional[ArrayLike] = None,
    ) -> Tuple[float, float]:

        values = np.asarray(values, dtype=float)

        values = values[np.isfinite(values)]

        N = len(values)

        if N == 0:
            return np.nan, np.nan

        value = np.median(values)

        error = (
            1.4826
            * np.median(
                np.abs(values - value)
            )
        )

        return value, error

class WeightedMeanAggregator:

    def aggregate(
        self,
        values: ArrayLike,
        errors: Optional[ArrayLike] = None,
    ) -> Tuple[float, float]:

        values = np.asarray(values, dtype=float)

        if errors is None:
            return MeanAggregator().aggregate(values)

        errors = np.asarray(errors, dtype=float)

        valid = (
            np.isfinite(values)
            & np.isfinite(errors)
            & (errors > 0)
        )

        values = values[valid]
        errors = errors[valid]

        if len(values) == 0:
            return np.nan, np.nan

        weights = 1.0 / errors**2

        value = (
            np.sum(values * weights)
            / np.sum(weights)
        )

        error = 1.0 / np.sqrt(
            np.sum(weights)
        )

        return value, error


# @staticmethod
# def default_binner(values: ArrayLike, errors: Optional[ArrayLike] = None, **kwargs) -> Tuple[float, float]:
#     method = kwargs.get('method')

#     N = len(values)
    
#     if method == 'mean':
#         binned_value = np.nanmean(values)
#         if errors is not None:
#             binned_error = np.sqrt(np.nansum(errors**2)) / N
#         else:
#             binned_error = np.nanstd(values) / np.sqrt(N)
#     elif method == 'median':
#         binned_value = np.nanmedian(values)
#         if errors is not None:
#             # For median, use median absolute deviation
#             binned_error = 1.4826 * np.nanmedian(np.abs(values - binned_value))
#         else:
#             binned_error = 1.4826 * np.nanmedian(np.abs(values - binned_value))
    
#     elif method == 'weighted_mean':
#         if errors is None:
#             # Fall back to simple mean
#             binned_value = np.nanmean(values)
#             binned_error = np.nanstd(values) / np.sqrt(len(values))
#         else:
#             # Weighted mean
#             weights = 1.0 / (errors**2)
#             binned_value = np.nansum(values * weights) / np.nansum(weights)
#             binned_error = 1.0 / np.sqrt(np.nansum(weights))
    
#     else:
#         raise ValueError(f"Unknown binning method: {method}")
    
#     return binned_value, binned_error