from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

# A layer launches once per process and the choice is then frozen, so each case needs its own
# interpreter. `threading_layer()` is the prediction; `numba.threading_layer()` is the truth.
PROBE = """
import numba
import numpy as np

import fast_array_utils.numba as fa_numba

predicted = fa_numba.threading_layer()


@numba.njit(parallel=True, cache=False)
def total(v):
    t = 0.0
    for i in numba.prange(v.shape[0]):
        t += v[i]
    return t


total(np.zeros(4))
print(predicted, numba.threading_layer())
"""


def _tbb_loadable() -> bool:
    try:
        importlib.import_module("numba.np.ufunc.tbbpool")
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _tbb_loadable(), reason="without tbb both resolve to workqueue and agree vacuously")
@pytest.mark.parametrize("category", ["forksafe", "threadsafe"])
def test_predicted_layer_matches_launched(category: str) -> None:
    env = {
        **os.environ,
        "NUMBA_THREADING_LAYER": category,
        "NUMBA_THREADING_LAYER_PRIORITY": "omp workqueue tbb",
    }
    done = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    predicted, launched = done.stdout.split()

    assert predicted == launched, f"{category!r}: predicted {predicted!r}, numba launched {launched!r}"
