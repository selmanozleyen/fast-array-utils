# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.sparse as sps

from fast_array_utils import types
from fast_array_utils.conv import to_dense
from fast_array_utils.linalg import matmul
from testing.fast_array_utils import Flags, random_mat


pytest.importorskip("numba")

import numba


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray
    from pytest_codspeed import BenchmarkFixture

    from fast_array_utils.typing import CpuArray
    from testing.fast_array_utils import ArrayType


def random_exact(rng: np.random.Generator, shape: tuple[int, int], dtype: type[np.number[Any]]) -> NDArray[np.number[Any]]:
    """Sparse-ish values in {-1.5, -1, …, 1.5}, so float sums are exact in any order."""
    arr = rng.integers(-3, 4, shape) * (rng.random(shape) < 0.3)
    return (arr / 2 if np.dtype(dtype).kind == "f" else arr).astype(dtype)


@pytest.mark.array_type(skip=Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("x_cls", [sps.csr_array, sps.csr_matrix])
@pytest.mark.parametrize(
    ("dtype_x", "dtype_y"),
    [(np.float32, np.float32), (np.float64, np.float64), (np.int64, np.int64), (np.float32, np.float64)],
    ids=["f32", "f64", "i64", "f32-f64"],
)
def test_matmul(
    array_type: ArrayType[CpuArray, None],
    x_cls: type[types.csr_array | types.csr_matrix],
    dtype_x: type[np.number[Any]],
    dtype_y: type[np.number[Any]],
) -> None:
    rng = np.random.default_rng(0)
    x_np = random_exact(rng, (30, 20), dtype_x)
    x_np[[0, -1]] = 0  # empty rows
    x = x_cls(x_np)
    y = array_type(random_exact(rng, (20, 10), dtype_y), dtype=dtype_y)
    if isinstance(y, types.CSBase) and y.format == "csc":
        with pytest.raises(TypeError, match=r"`y` must be"):
            matmul(x, y)
        return

    np.testing.assert_array_equal(matmul(x, y), to_dense(x @ y), strict=True)


@pytest.mark.parametrize(
    ("x", "y", "err", "match"),
    [
        pytest.param(sps.csc_array(np.eye(3)), np.eye(3), TypeError, r"`x` must be", id="x-csc"),
        pytest.param(np.eye(3), np.eye(3), TypeError, r"`x` must be", id="x-dense"),
        pytest.param(sps.csr_array(np.eye(3)), np.eye(4), ValueError, r"not aligned", id="shape"),
        pytest.param(sps.csr_array(np.eye(3)), np.ones(3), ValueError, r"not aligned", id="y-1d"),
    ],
)
def test_matmul_errors(x: object, y: object, err: type[Exception], match: str) -> None:
    with pytest.raises(err, match=match):
        matmul(x, y)  # type: ignore[arg-type]


@pytest.mark.parametrize("y_format", ["dense", "csr"])
def test_matmul_thread_count(y_format: Literal["dense", "csr"]) -> None:
    """Results must not depend on how rows are split across threads."""
    if (n_threads := numba.get_num_threads()) < 2:
        pytest.skip("needs at least 2 threads")
    rng = np.random.default_rng(0)
    x = random_mat((2_000, 500), density=0.05, dtype=np.float32, rng=rng)
    y = random_mat((500, 50), density=0.3, dtype=np.float32, rng=rng)
    y_in = y if y_format == "csr" else y.toarray()

    numba.set_num_threads(1)
    try:
        serial = matmul(x, y_in)
    finally:
        numba.set_num_threads(n_threads)

    np.testing.assert_array_equal(matmul(x, y_in), serial, strict=True)
    np.testing.assert_allclose(serial, to_dense(x @ y), rtol=1e-5)


@pytest.mark.benchmark
@pytest.mark.parametrize("y_format", ["dense", "csr"])
def test_matmul_benchmark(benchmark: BenchmarkFixture, y_format: Literal["dense", "csr"]) -> None:
    # a kNN weight matrix with 30 neighbors per row, as used to impute or smooth over neighbors
    n, k = 10_000, 30
    rng = np.random.default_rng(0)
    x = sps.csr_array((rng.random(n * k, dtype=np.float32), rng.integers(0, n, n * k), np.arange(0, n * k + 1, k)), shape=(n, n))
    y_csr = random_mat((n, 100), density=0.1, dtype=np.float32, rng=rng)
    y = y_csr if y_format == "csr" else y_csr.toarray()
    matmul(x, y)  # warmup: numba compile

    @benchmark
    def call() -> None:
        matmul(x, y)
