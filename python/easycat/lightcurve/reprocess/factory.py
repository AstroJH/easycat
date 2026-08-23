from typing import Dict, Any, Type

from .outlier import (
    OutlierDetector,
    MADDetector,
    IQRDetector
)

from .binning.grouper import (
    Grouper,
    MaxBinwidthGrouper,
    MinBinwidthGrouper,
    MaxIntervalGrouper
)

from .binning.aggregator import (
    Aggregator,
    MeanAggregator,
    MedianAggregator,
    WeightedMeanAggregator
)


GROUPERS = {
    "max_interval": MaxIntervalGrouper,
    "min_binwidth": MinBinwidthGrouper,
    "max_binwidth": MaxBinwidthGrouper,
}

AGGREGATORS = {
    "mean": MeanAggregator,
    "median": MedianAggregator,
    "weighted_mean": WeightedMeanAggregator,
}

OUTLIER_DETECTORS = {
    "mad": MADDetector,
    "iqr": IQRDetector,
}


def register_grouper(
    name: str,
    cls: Type[Grouper],
) -> None:

    if name in GROUPERS:
        raise ValueError(
            f"Grouper '{name}' is already registered."
        )

    GROUPERS[name] = cls


def register_aggregator(
    name: str,
    cls: Type[Aggregator],
) -> None:

    if name in AGGREGATORS:
        raise ValueError(
            f"Aggregator '{name}' is already registered."
        )

    AGGREGATORS[name] = cls


def register_outlier_detector(
    name: str,
    cls: Type[OutlierDetector],
) -> None:

    if name in OUTLIER_DETECTORS:
        raise ValueError(
            f"OutlierDetector '{name}' is already registered."
        )

    OutlierDetector[name] = cls


def create_grouper(
    config: Dict[str, Any],
) -> Grouper:

    method = config.get("method")

    if method not in GROUPERS:
        raise ValueError(
            f"Unknown time grouping method: '{method}'. "
            f"Available methods: "
            f"{list(GROUPERS)}"
        )

    grouper_class = GROUPERS[method]

    params = {
        key: value
        for key, value in config.items()
        if key != "method"
    }

    return grouper_class(**params)


def create_aggregator(
    config: Dict[str, Any],
) -> Aggregator:

    method = config.get("method")

    if method not in AGGREGATORS:
        raise ValueError(
            f"Unknown aggregation method: '{method}'. "
            f"Available methods: {list(AGGREGATORS)}"
        )

    aggregator_class = AGGREGATORS[method]

    params = {
        key: value
        for key, value in config.items()
        if key != "method"
    }

    return aggregator_class(**params)


def create_outlier_detector(config: Dict[str, Any]) -> OutlierDetector:
    """
    Create an outlier detector from configuration.

    Parameters
    ----------
    config : dict
        Detector configuration. Examples:
    ```
    {
        "method": "mad",
        "threshold": 5.0,
    }

    {
        "method": "iqr",
        "multiplier": 1.5,
    }
    ```
    """

    method = config.get("method")

    if method not in OUTLIER_DETECTORS:
        raise ValueError(
            f"Unknown outlier method: {method}. "
            f"Available methods: "
            f"{list(OUTLIER_DETECTORS)}"
        )

    detector_class = OUTLIER_DETECTORS[method]

    params = {
        key: value
        for key, value in config.items()
        if key != "method"
    }

    return detector_class(**params)
