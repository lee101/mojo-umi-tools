"""Mojo-accelerated, API-compatible UMI-tools network deduplication."""

from ._lib import build
from .network import (
    UMIClusterer,
    breadth_first_search,
    build_substr_idx,
    edit_distance,
    get_substr_slices,
    iter_nearest_neighbours,
    remove_umis,
)

__version__ = "0.1.0"
__all__ = [
    "UMIClusterer", "edit_distance", "breadth_first_search", "remove_umis",
    "get_substr_slices", "build_substr_idx", "iter_nearest_neighbours", "build",
]
