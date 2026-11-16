# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba

from ..numba import njit


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from .. import types


__all__ = ["matmul_csr_csr", "matmul_csr_dense"]


# Each row of `x` only adds to its own row of `out`, in a fixed order, so no thread races or reorders sums.


@njit
def matmul_csr_dense(x: types.CSBase, y: NDArray[Any], out: NDArray[Any]) -> None:
    for i in numba.prange(out.shape[0]):
        for p in range(x.indptr[i], x.indptr[i + 1]):
            w, row = x.data[p], x.indices[p]
            for j in range(out.shape[1]):
                out[i, j] += w * y[row, j]


@njit
def matmul_csr_csr(x: types.CSBase, y: types.CSBase, out: NDArray[Any]) -> None:
    for i in numba.prange(out.shape[0]):
        for p in range(x.indptr[i], x.indptr[i + 1]):
            w, row = x.data[p], x.indices[p]
            for q in range(y.indptr[row], y.indptr[row + 1]):
                out[i, y.indices[q]] += w * y.data[q]
