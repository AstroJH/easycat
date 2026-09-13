"""Summary statistics used by two-dimensional binning."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np


DEFAULT_STATS = (
    "count",
    "mean",
    "median",
    "std",
    "error_weighted_mean",
    "error_weighted_std",
)


def _as_float_array(values: Iterable[Any]) -> np.ndarray:
    """Convert values to a one-dimensional float array."""
    return np.asarray(values, dtype=float).reshape(-1)


def summarize(
    values: Iterable[Any],
    *,
    errors: Optional[Iterable[Any]] = None,
    weights: Optional[Iterable[Any]] = None,
) -> Mapping[str, float | int]:
    """Compute common statistics for one bin or selected subset.

    Invalid values are removed consistently across values, errors and weights.
    Standard deviation uses ``ddof=1``.  Error-weighted means use inverse
    variance weights; a caller-supplied weight column takes precedence for
    ``weighted_mean`` but does not replace the error-weighted result.
    """
    values_arr = _as_float_array(values)
    total_count = int(values_arr.size)
    finite = np.isfinite(values_arr)

    errors_arr = None
    if errors is not None:
        errors_arr = _as_float_array(errors)
        if errors_arr.size != values_arr.size:
            raise ValueError("errors and values must have the same length")
        finite &= np.isfinite(errors_arr) & (errors_arr > 0)

    weights_arr = None
    if weights is not None:
        weights_arr = _as_float_array(weights)
        if weights_arr.size != values_arr.size:
            raise ValueError("weights and values must have the same length")
        finite &= np.isfinite(weights_arr) & (weights_arr > 0)

    clean_values = values_arr[finite]
    count = int(clean_values.size)
    result: dict[str, float | int] = {
        "total_count": total_count,
        "count": count,
        "empty": int(total_count - count),
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "sem": np.nan,
        "weighted_mean": np.nan,
        "weighted_std": np.nan,
        "error_weighted_mean": np.nan,
        "error_weighted_std": np.nan,
    }
    if count == 0:
        return result

    result["mean"] = float(np.mean(clean_values))
    result["median"] = float(np.median(clean_values))
    if count > 1:
        result["std"] = float(np.std(clean_values, ddof=1))
        result["sem"] = float(result["std"] / np.sqrt(count))

    if weights_arr is not None:
        clean_weights = weights_arr[finite]
        weight_sum = float(np.sum(clean_weights))
        weighted_mean = float(np.sum(clean_weights * clean_values) / weight_sum)
        result["weighted_mean"] = weighted_mean
        result["weighted_std"] = float(
            np.sqrt(
                np.sum(clean_weights * (clean_values - weighted_mean) ** 2)
                / weight_sum
            )
        )

    if errors_arr is not None:
        clean_errors = errors_arr[finite]
        variance_weights = 1.0 / clean_errors**2
        weight_sum = float(np.sum(variance_weights))
        error_weighted_mean = float(
            np.sum(variance_weights * clean_values) / weight_sum
        )
        result["error_weighted_mean"] = error_weighted_mean
        result["error_weighted_std"] = float(
            np.sqrt(1.0 / weight_sum)
        )

    return result


def select_stats(
    summary: Mapping[str, float | int],
    names: Optional[Iterable[str]] = None,
) -> dict[str, float | int]:
    """Select and validate requested statistics from a summary mapping."""
    requested = tuple(names or DEFAULT_STATS)
    unknown = [name for name in requested if name not in summary]
    if unknown:
        raise KeyError(f"unknown statistics: {unknown}")
    return {name: summary[name] for name in requested}


__all__ = ["DEFAULT_STATS", "select_stats", "summarize"]
