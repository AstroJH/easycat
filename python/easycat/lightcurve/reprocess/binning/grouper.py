from typing import Protocol, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


BinBoundaries = Tuple[
    NDArray[np.intp],
    NDArray[np.intp],
]

class Grouper(Protocol):

    def group(
        self,
        data: ArrayLike,
    ) -> BinBoundaries:
        ...

class MaxIntervalGrouper:

    def __init__(
        self,
        max_interval: float = 1.2,
    ):
        if max_interval <= 0:
            raise ValueError(
                "max_interval must be positive."
            )

        self.max_interval = max_interval

    def group(
        self,
        data: ArrayLike,
    ) -> BinBoundaries:

        data = np.asarray(data)

        lo = []
        hi = []

        n = len(data)

        if n == 0:
            return (
                np.asarray(lo, dtype=np.intp),
                np.asarray(hi, dtype=np.intp),
            )

        lo.append(0)

        for i in range(1, n):
            if data[i] - data[i - 1] > self.max_interval:
                hi.append(i - 1)
                lo.append(i)

        hi.append(n - 1)

        return (
            np.asarray(lo, dtype=np.intp),
            np.asarray(hi, dtype=np.intp),
        )


class MinBinwidthGrouper:

    def __init__(
        self,
        min_binwidth: float = 1.0,
        giveup_last: bool = False,
    ):
        if min_binwidth <= 0:
            raise ValueError(
                "min_binwidth must be positive."
            )

        self.min_binwidth = min_binwidth
        self.giveup_last = giveup_last

    def group(
        self,
        data: ArrayLike,
    ) -> BinBoundaries:

        data = np.asarray(data)

        los = []
        his = []

        n = len(data)
        i = 0

        while i < n:

            start_idx = i
            start_time = data[i]

            if i == n - 1:

                if not self.giveup_last:
                    los.append(start_idx)
                    his.append(start_idx)

                break

            j = i + 1

            while (
                j < n
                and data[j] - start_time
                < self.min_binwidth
            ):
                j += 1

            if j < n:

                los.append(start_idx)
                his.append(j)

                i = j + 1

            else:

                if not self.giveup_last:
                    los.append(start_idx)
                    his.append(n - 1)

                break

        return (
            np.asarray(los, dtype=np.intp),
            np.asarray(his, dtype=np.intp),
        )

class MaxBinwidthGrouper:

    def __init__(
        self,
        max_binwidth: float = 1.0,
    ):
        if max_binwidth <= 0:
            raise ValueError(
                "max_binwidth must be positive."
            )

        self.max_binwidth = max_binwidth

    def group(
        self,
        data: ArrayLike,
    ) -> BinBoundaries:

        data = np.asarray(data)

        los = []
        his = []

        n = len(data)
        i = 0

        while i < n:

            start = i

            while (
                i + 1 < n
                and data[i + 1] - data[start]
                <= self.max_binwidth
            ):
                i += 1

            los.append(start)
            his.append(i)

            i += 1

        return (
            np.asarray(los, dtype=np.intp),
            np.asarray(his, dtype=np.intp),
        )