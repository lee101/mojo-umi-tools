"""ctypes access to the standalone Mojo UMI kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_UMI_TOOLS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-umi-tools.so"
)
I = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_UMI_TOOLS_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_UMI_TOOLS_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than the kernel."""
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.environ.get("MOJO_UMI_TOOLS_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        _mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        fn = _loaded.mut_pair_edges
        fn.argtypes = [I, I, I, I, I, I, I, I]
        fn.restype = None
        dense_fn = _loaded.mut_dense_pair_edges
        dense_fn.argtypes = [I, I, I, I, I, I, I, I]
        dense_fn.restype = None
    return _loaded


def pair_edges(
    sequences: np.ndarray, pairs: np.ndarray, counts: np.ndarray,
    threshold: int, directional: bool,
) -> np.ndarray:
    """Return directed edge bitmasks for a checked, contiguous native call.

    This is deliberately strict about dtypes.  Converting an index or count at
    this boundary could silently change the graph that the Mojo code sees.
    Non-contiguous arrays of the right dtype are copied and kept alive for the
    entire ctypes call.
    """
    if not isinstance(sequences, np.ndarray) or sequences.dtype != np.uint8 or sequences.ndim != 2:
        raise TypeError("sequences must be a two-dimensional uint8 numpy array")
    if not isinstance(pairs, np.ndarray) or pairs.dtype != np.int64 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise TypeError("pairs must be an (n, 2) int64 numpy array")
    if not isinstance(counts, np.ndarray) or counts.dtype != np.int64 or counts.ndim != 1:
        raise TypeError("counts must be a one-dimensional int64 numpy array")
    if counts.shape[0] != sequences.shape[0]:
        raise ValueError("counts must contain one value per sequence")
    if not isinstance(threshold, (int, np.integer)) or isinstance(threshold, (bool, np.bool_)) or threshold < 0:
        raise ValueError("threshold must be a non-negative integer")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= sequences.shape[0])):
        raise ValueError("pair indices must refer to sequences")

    # These copies, where needed, are local references and therefore cannot be
    # collected until mut_pair_edges has returned.
    sequences = np.ascontiguousarray(sequences)
    pairs = np.ascontiguousarray(pairs)
    counts = np.ascontiguousarray(counts)
    if pairs.size == 0:
        return np.empty(0, dtype=np.uint8)
    edges = np.empty(pairs.shape[0], dtype=np.uint8)
    lib().mut_pair_edges(
        sequences.ctypes.data, pairs.ctypes.data, pairs.shape[0], sequences.shape[1],
        threshold, counts.ctypes.data, int(directional), edges.ctypes.data,
    )
    return edges


def dense_pair_edges(
    sequences: np.ndarray, counts: np.ndarray, threshold: int, directional: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all upper-triangle indices and their edge masks."""
    if not isinstance(sequences, np.ndarray) or sequences.dtype != np.uint8 or sequences.ndim != 2:
        raise TypeError("sequences must be a two-dimensional uint8 numpy array")
    if not isinstance(counts, np.ndarray) or counts.dtype != np.int64 or counts.ndim != 1:
        raise TypeError("counts must be a one-dimensional int64 numpy array")
    if counts.shape[0] != sequences.shape[0]:
        raise ValueError("counts must contain one value per sequence")
    if not isinstance(threshold, (int, np.integer)) or isinstance(threshold, (bool, np.bool_)) or threshold < 0:
        raise ValueError("threshold must be a non-negative integer")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    sequences = np.ascontiguousarray(sequences)
    counts = np.ascontiguousarray(counts)
    pair_count = sequences.shape[0] * (sequences.shape[0] - 1) // 2
    if pair_count == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.uint8)
    pairs = np.empty((pair_count, 2), dtype=np.int64)
    edges = np.empty(pair_count, dtype=np.uint8)
    lib().mut_dense_pair_edges(
        sequences.ctypes.data, sequences.shape[0], sequences.shape[1], threshold,
        counts.ctypes.data, int(directional), pairs.ctypes.data, edges.ctypes.data,
    )
    return pairs, edges
