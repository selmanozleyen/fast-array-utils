# SPDX-License-Identifier: MPL-2.0
"""Linear algebra utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


__all__ = ["matmul"]


def matmul(x: types.CSBase, y: NDArray[Any] | types.CSBase, /) -> NDArray[Any]:
    """Multiply a CSR matrix with a dense or CSR matrix into a dense result, in parallel over the rows of ``x``.

    Equivalent to ``x @ y`` (densified), which :mod:`scipy.sparse` computes on a single thread.
    A typical ``x`` is a k-nearest-neighbors weight matrix, making the result a weighted sum
    of each row’s neighbors in ``y``.
    The result doesn’t depend on the number of threads.

    Parameters
    ----------
    x
        CSR array or matrix of shape ``(n, m)``.
    y
        :class:`numpy.ndarray` or CSR array or matrix of shape ``(m, k)``.

    Returns
    -------
    Dense product of shape ``(n, k)``, with the dtype that ``x @ y`` has.

    Example
    -------
    >>> import numpy as np
    >>> import scipy.sparse as sps
    >>> x = sps.csr_array([[0.5, 0.5, 0], [0, 0, 0]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> matmul(x, y)
    array([[2., 3.],
           [0., 0.]])

    See Also
    --------
    :data:`numpy.matmul`
    """
    if not (isinstance(x, types.CSBase) and x.format == "csr"):
        msg = f"`x` must be a CSR array or matrix, got {type(x).__name__}."
        raise TypeError(msg)
    if not (isinstance(y, np.ndarray) or (isinstance(y, types.CSBase) and y.format == "csr")):
        msg = f"`y` must be a numpy array or a CSR array or matrix, got {type(y).__name__}."
        raise TypeError(msg)
    if (x.ndim, y.ndim) != (2, 2) or x.shape[1] != y.shape[0]:
        msg = f"Shapes {x.shape} and {y.shape} are not aligned for a 2D matrix product."
        raise ValueError(msg)

    from ._matmul import matmul_csr_csr, matmul_csr_dense

    out = np.zeros((x.shape[0], y.shape[1]), dtype=np.result_type(x.dtype, y.dtype))
    if isinstance(y, np.ndarray):
        matmul_csr_dense(x, y, out)
    else:
        matmul_csr_csr(x, y, out)
    return out
