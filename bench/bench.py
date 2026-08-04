"""Benchmark end-to-end UMI clustering against umi-tools on identical maps."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojoumitools as mut  # noqa: E402
from umi_tools import UMIClusterer as ReferenceClusterer  # noqa: E402


def make_counts(n: int = 1_000, length: int = 12) -> dict[bytes, int]:
    """Generate dense candidate buckets, exercising the native Hamming kernel."""
    rng = np.random.default_rng(7)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    result = {}
    prefix = b"AAAAAA"
    while len(result) < n:
        suffix = bytes(rng.choice(alphabet, size=length - len(prefix)).tolist())
        result.setdefault(prefix + suffix, int(rng.integers(1, 10_000)))
    return result


def best_time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    counts = make_counts()
    print(f"Machine: {platform.platform()} | Python {platform.python_version()} | {len(counts)} 12bp UMIs")
    print("| case | mojo-umi-tools | umi-tools 1.1.6 | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for method in ("directional", "adjacency", "cluster"):
        ours = lambda: mut.UMIClusterer(cluster_method=method)(counts, 1)
        theirs = lambda: ReferenceClusterer(cluster_method=method)(counts, 1)
        ours()
        theirs()
        ours_time = best_time(ours)
        reference_time = best_time(theirs)
        label = "faster" if ours_time < reference_time else "slower"
        print(
            f"| `{method}` (1,000 dense candidates) | {ours_time * 1e3:.1f} ms | "
            f"{reference_time * 1e3:.1f} ms | {reference_time / ours_time:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
