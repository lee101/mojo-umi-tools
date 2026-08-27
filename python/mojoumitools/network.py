"""UMI-tools-compatible network clustering backed by a Mojo edge kernel."""

from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from statistics import median
from typing import Iterable, Mapping

import numpy as np

from ._lib import dense_pair_edges, pair_edges

UMI = bytes


def edit_distance(umi1: bytes, umi2: bytes) -> int:
    """Return the Hamming distance used by UMI-tools for equal-length UMIs."""
    if len(umi1) != len(umi2):
        raise ValueError("UMIs must have equal length")
    return sum(left != right for left, right in zip(umi1, umi2))


def breadth_first_search(node: UMI, adj_list: Mapping[UMI, Iterable[UMI]]) -> set[UMI]:
    """Return all nodes reachable from *node*, matching umi_tools.network."""
    searched = {node}
    queue = [node]
    while queue:
        current = queue.pop()
        for child in adj_list[current]:
            if child not in searched:
                searched.add(child)
                queue.append(child)
    return searched


def remove_umis(adj_list: Mapping[UMI, Iterable[UMI]], cluster: Iterable[UMI],
                nodes: Iterable[UMI]) -> set[UMI]:
    """Remove selected nodes and their immediate neighbours from a cluster."""
    nodes = list(nodes)
    removed = set(nodes)
    for node in nodes:
        removed.update(adj_list[node])
    return set(cluster) - removed


def get_substr_slices(umi_length: int, idx_size: int) -> list[tuple[int, int]]:
    """Split a UMI into the pigeonhole index slices used upstream."""
    quotient, remainder = divmod(umi_length, idx_size)
    sizes = [quotient + 1] * remainder + [quotient] * (idx_size - remainder)
    offset = 0
    slices = []
    for size in sizes:
        slices.append((offset, offset + size))
        offset += size
    return slices


def build_substr_idx(umis: Iterable[UMI], umi_length: int, min_edit: int):
    """Build the exact substring candidate index used by UMI-tools."""
    index = defaultdict(lambda: defaultdict(set))
    for bounds in get_substr_slices(umi_length, min_edit + 1):
        start, stop = bounds
        for umi in umis:
            index[bounds][umi[start:stop]].add(umi)
    return index


def iter_nearest_neighbours(umis: list[UMI], substr_idx):
    """Yield each pair that shares a pigeonhole-index substring once."""
    seen: set[UMI] = set()
    for umi in umis:
        neighbours: set[UMI] = set()
        for bounds, substr_map in substr_idx.items():
            start, stop = bounds
            neighbours.update(substr_map[umi[start:stop]])
        neighbours.difference_update(seen)
        neighbours.discard(umi)
        for neighbour in neighbours:
            yield umi, neighbour
        seen.add(umi)


class UMIClusterer:
    """Cluster a ``{UMI: read_count}`` mapping with UMI-tools' five methods."""

    def __init__(self, cluster_method="directional"):
        self.max_umis_per_position = 0
        self.total_umis_per_position = 0
        self.positions = 0
        methods = {
            "adjacency": (self._get_adj_list_adjacency,
                          self._get_connected_components_adjacency, self._group_adjacency),
            "directional": (self._get_adj_list_directional,
                            self._get_connected_components_adjacency, self._group_directional),
            "cluster": (self._get_adj_list_adjacency,
                        self._get_connected_components_adjacency, self._group_cluster),
            "percentile": (self._get_adj_list_null,
                           self._get_connected_components_null, self._group_percentile),
            "unique": (self._get_adj_list_null,
                       self._get_connected_components_null, self._group_unique),
        }
        if cluster_method not in methods:
            raise ValueError("cluster_method must be adjacency, directional, cluster, percentile, or unique")
        self.cluster_method = cluster_method
        self.get_adj_list, self.get_connected_components, self.get_groups = methods[cluster_method]

    @staticmethod
    def _candidate_pair_indices(umis: list[UMI], threshold: int) -> np.ndarray:
        """Return candidate pairs as native-ready ``int64`` row indices."""
        size = len(umis)
        if size <= 25:
            left, right = np.triu_indices(size, k=1)
            return np.column_stack((left, right)).astype(np.int64, copy=False)
        index = build_substr_idx(umis, len(umis[0]), threshold)
        if any(len(bucket) == size for substr_map in index.values() for bucket in substr_map.values()):
            left, right = np.triu_indices(size, k=1)
            return np.column_stack((left, right)).astype(np.int64, copy=False)
        positions = {umi: position for position, umi in enumerate(umis)}
        flat_indices = np.fromiter(
            (positions[umi] for pair in iter_nearest_neighbours(umis, index) for umi in pair),
            dtype=np.int64,
        )
        return flat_indices.reshape((-1, 2))

    @staticmethod
    def _uses_complete_pair_set(umis: list[UMI], threshold: int) -> bool:
        size = len(umis)
        if size <= 25:
            return True
        index = build_substr_idx(umis, len(umis[0]), threshold)
        return any(
            len(bucket) == size
            for substr_map in index.values()
            for bucket in substr_map.values()
        )

    @staticmethod
    def _candidate_pairs(umis: list[UMI], threshold: int) -> list[tuple[UMI, UMI]]:
        indices = UMIClusterer._candidate_pair_indices(umis, threshold)
        return [(umis[left], umis[right]) for left, right in indices]

    @staticmethod
    def _edge_list(umis: list[UMI], counts: Mapping[UMI, int], threshold: int,
                   directional: bool) -> list[tuple[UMI, UMI, int]]:
        if (not isinstance(threshold, Integral) or isinstance(threshold, bool)
                or threshold < 0):
            raise ValueError("threshold must be a non-negative integer")
        limit = np.iinfo(np.int64).max
        if any(not isinstance(counts[umi], Integral) or isinstance(counts[umi], bool)
               or counts[umi] < 0 or counts[umi] > limit for umi in umis):
            raise ValueError("UMI counts must be non-negative signed 64-bit integers")
        encoded = np.frombuffer(b"".join(umis), dtype=np.uint8).reshape(len(umis), len(umis[0]))
        frequencies = np.fromiter((counts[umi] for umi in umis), dtype=np.int64, count=len(umis))
        if UMIClusterer._uses_complete_pair_set(umis, threshold):
            indices, edge_masks = dense_pair_edges(
                encoded, frequencies, threshold, directional
            )
        else:
            indices = UMIClusterer._candidate_pair_indices(umis, threshold)
            if indices.size == 0:
                return []
            edge_masks = pair_edges(encoded, indices, frequencies, threshold, directional)
        return [
            (umis[int(indices[position, 0])], umis[int(indices[position, 1])], int(edge_masks[position]))
            for position in np.flatnonzero(edge_masks)
        ]

    def _get_adj_list_adjacency(self, umis, counts, threshold):
        graph = {umi: [] for umi in umis}
        for left, right, _ in self._edge_list(umis, counts, threshold, False):
            graph[left].append(right)
            graph[right].append(left)
        return graph

    def _get_adj_list_directional(self, umis, counts, threshold=1):
        graph = {umi: [] for umi in umis}
        for left, right, mask in self._edge_list(umis, counts, threshold, True):
            if mask & 1:
                graph[left].append(right)
            if mask & 2:
                graph[right].append(left)
        return graph

    @staticmethod
    def _get_adj_list_null(umis, counts, threshold):
        return None

    @staticmethod
    def _get_connected_components_adjacency(umis, graph, counts):
        found: set[UMI] = set()
        components = []
        for node in sorted(graph, key=lambda umi: counts[umi], reverse=True):
            if node not in found:
                component = sorted(breadth_first_search(node, graph))
                found.update(component)
                components.append(component)
        return components

    @staticmethod
    def _get_connected_components_null(umis, adj_list, counts):
        return umis

    @staticmethod
    def _get_best_min_account(cluster, adj_list, counts):
        if len(cluster) == 1:
            return list(cluster)
        nodes = sorted(cluster, key=lambda umi: counts[umi], reverse=True)
        cluster_set = set(cluster)
        covered: set[UMI] = set()
        for index in range(len(nodes) - 1):
            node = nodes[index]
            covered.add(node)
            covered.update(adj_list[node])
            if cluster_set.issubset(covered):
                return nodes[:index + 1]
        return nodes

    @staticmethod
    def _get_best_percentile(cluster, counts):
        if len(cluster) == 1:
            return list(cluster)
        cutoff = median(counts.values()) / 100
        return [umi for umi in cluster if counts[umi] > cutoff]

    @staticmethod
    def _group_unique(clusters, adj_list, counts):
        return [clusters] if len(clusters) == 1 else [[umi] for umi in clusters]

    @staticmethod
    def _group_directional(clusters, adj_list, counts):
        observed: set[UMI] = set()
        groups = []
        for cluster in clusters:
            if len(cluster) == 1:
                groups.append(list(cluster))
                observed.update(cluster)
            else:
                group = []
                for umi in sorted(cluster, key=lambda umi: counts[umi], reverse=True):
                    if umi not in observed:
                        group.append(umi)
                        observed.add(umi)
                groups.append(group)
        return groups

    def _group_adjacency(self, clusters, adj_list, counts):
        groups = []
        for cluster in clusters:
            if len(cluster) == 1:
                groups.append(cluster)
                continue
            observed: set[UMI] = set()
            leads = self._get_best_min_account(cluster, adj_list, counts)
            observed.update(leads)
            for lead in leads:
                connected = set(adj_list[lead])
                groups.append([lead] + list(connected - observed))
                observed.update(connected)
        return groups

    @staticmethod
    def _group_cluster(clusters, adj_list, counts):
        return [sorted(cluster, key=lambda umi: counts[umi], reverse=True) for cluster in clusters]

    def _group_percentile(self, clusters, adj_list, counts):
        return [[umi] for umi in self._get_best_percentile(clusters, counts)]

    def __call__(self, umis, threshold):
        counts = umis
        umis = list(umis.keys())
        self.positions += 1
        self.total_umis_per_position += len(umis)
        self.max_umis_per_position = max(self.max_umis_per_position, len(umis))
        if not umis:
            return []
        if not all(isinstance(umi, bytes) for umi in umis):
            raise TypeError("UMI keys must be bytes, as in umi_tools")
        lengths = {len(umi) for umi in umis}
        if len(lengths) != 1:
            raise AssertionError("not all umis are the same length(!)")
        graph = self.get_adj_list(umis, counts, threshold)
        clusters = self.get_connected_components(umis, graph, counts)
        return [list(group) for group in self.get_groups(clusters, graph, counts)]
