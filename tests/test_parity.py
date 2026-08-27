"""Behavioural parity with umi-tools 1.1.6 on the same UMI count maps."""

from __future__ import annotations

import numpy as np
import pytest

from umi_tools import UMIClusterer as ReferenceClusterer
from umi_tools import network as reference_network

import mojoumitools as mut
from mojoumitools._lib import dense_pair_edges, pair_edges


def canonical(groups):
    return frozenset(frozenset(group) for group in groups)


def leaders(groups):
    return {frozenset(group): group[0] for group in groups}


def umi_map(n: int, length: int, seed: int) -> dict[bytes, int]:
    rng = np.random.default_rng(seed)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    result = {}
    while len(result) < n:
        umi = bytes(rng.choice(alphabet, size=length).tolist())
        result.setdefault(umi, int(rng.integers(1, 100)))
    return result


@pytest.mark.parametrize("method", ["unique", "percentile", "cluster", "adjacency", "directional"])
def test_every_method_matches_upstream_on_small_graph(method):
    counts = {
        b"AAAA": 50, b"AAAT": 25, b"AATT": 12, b"ATTT": 6,
        b"CCCC": 9, b"CCCA": 2, b"GGGG": 1,
    }
    expected = ReferenceClusterer(cluster_method=method)(counts, 1)
    actual = mut.UMIClusterer(cluster_method=method)(counts, 1)
    assert canonical(actual) == canonical(expected)
    assert leaders(actual) == leaders(expected)


@pytest.mark.parametrize("method", ["cluster", "adjacency", "directional"])
@pytest.mark.parametrize("threshold", [1, 2])
def test_network_methods_match_upstream_with_substring_index(method, threshold):
    counts = umi_map(80, 10, seed=threshold)
    expected = ReferenceClusterer(cluster_method=method)(counts, threshold)
    actual = mut.UMIClusterer(cluster_method=method)(counts, threshold)
    assert canonical(actual) == canonical(expected)
    assert leaders(actual) == leaders(expected)


def test_directional_documented_example_and_statistics():
    counts = {b"ATAT": 10, b"GTAT": 5, b"CCAT": 3}
    clusterer = mut.UMIClusterer(cluster_method="directional")
    assert clusterer(counts, 1) == [[b"ATAT", b"GTAT"], [b"CCAT"]]
    assert (clusterer.positions, clusterer.total_umis_per_position, clusterer.max_umis_per_position) == (1, 3, 3)


def test_adjacency_lists_match_upstream():
    counts = {b"AAAA": 9, b"AAAT": 4, b"AATT": 2, b"CCCC": 8}
    for method in ("adjacency", "directional"):
        expected = ReferenceClusterer(cluster_method=method).get_adj_list(list(counts), counts, 1)
        actual = mut.UMIClusterer(cluster_method=method).get_adj_list(list(counts), counts, 1)
        assert {key: set(value) for key, value in actual.items()} == {
            key: set(value) for key, value in expected.items()
        }


def test_helpers_match_upstream_semantics():
    assert mut.edit_distance(b"ACGT", b"AGGT") == 1
    assert mut.get_substr_slices(10, 3) == [(0, 4), (4, 7), (7, 10)]
    graph = {b"A": [b"B"], b"B": [b"A", b"C"], b"C": [b"B"]}
    assert mut.breadth_first_search(b"A", graph) == {b"A", b"B", b"C"}
    assert mut.remove_umis(graph, graph, [b"B"]) == set()


def test_substring_index_helpers_match_upstream_candidates():
    umis = [b"AAAA", b"AAAT", b"AATT", b"CCCC"]
    actual_index = mut.build_substr_idx(umis, 4, 1)
    expected_index = reference_network.build_substr_idx(umis, 4, 1)
    assert {
        bounds: {key: set(value) for key, value in buckets.items()}
        for bounds, buckets in actual_index.items()
    } == {
        bounds: {key: set(value) for key, value in buckets.items()}
        for bounds, buckets in expected_index.items()
    }
    assert set(mut.iter_nearest_neighbours(umis, actual_index)) == set(
        reference_network.iter_nearest_neighbours(umis, expected_index)
    )


def test_edit_distance_rejects_different_lengths():
    with pytest.raises(ValueError):
        mut.edit_distance(b"A", b"AA")


def test_invalid_mismatched_lengths_match_upstream_failure():
    counts = {b"AAA": 2, b"AAAA": 1}
    with pytest.raises(AssertionError):
        mut.UMIClusterer()(counts, 1)


def test_empty_mapping_is_a_useful_empty_result():
    assert mut.UMIClusterer()({}, 1) == []


def test_dense_substring_bucket_uses_complete_index_pairs():
    umis = [b"AAAAAA" + suffix for suffix in (b"AAAAAA", b"AAAAAC", b"AAAACA", b"AAACAA")]
    indices = mut.UMIClusterer._candidate_pair_indices(umis, 1)
    assert indices.dtype == np.int64
    assert {tuple(pair) for pair in indices} == {
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    }


def test_adjacency_leader_coverage_matches_incremental_union():
    cluster = [b"AAAA", b"AAAT", b"AATT"]
    graph = {
        b"AAAA": [b"AAAT"],
        b"AAAT": [b"AAAA", b"AATT", b"OUTSIDE"],
        b"AATT": [b"AAAT"],
    }
    counts = {b"AAAA": 9, b"AAAT": 4, b"AATT": 2}
    assert mut.UMIClusterer._get_best_min_account(cluster, graph, counts) == [b"AAAA", b"AAAT"]


def test_pair_edges_simd_tail_matches_scalar_hamming_distance():
    sequences = np.frombuffer(
        b"A" * 131 + b"A" * 130 + b"C" + b"C" + b"A" * 129 + b"C",
        dtype=np.uint8,
    ).reshape(3, 131)
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int64)
    counts = np.array([10, 1, 1], dtype=np.int64)
    assert pair_edges(sequences, pairs, counts, threshold=1, directional=True).tolist() == [1, 0]


@pytest.mark.parametrize("size", [362, 363])
def test_dense_pair_edges_matches_numpy_across_parallel_threshold(size):
    sequences = np.resize(np.frombuffer(b"AC", dtype=np.uint8), size).reshape(size, 1)
    counts = np.ones(size, dtype=np.int64)
    pairs, edges = dense_pair_edges(sequences, counts, threshold=0, directional=False)
    left, right = np.triu_indices(size, k=1)
    expected_pairs = np.column_stack((left, right)).astype(np.int64, copy=False)
    assert np.array_equal(pairs, expected_pairs)
    assert np.array_equal(edges, np.where(sequences[left, 0] == sequences[right, 0], 3, 0))


def test_pair_edges_rejects_unsafe_or_narrowing_ffi_inputs():
    sequences = np.zeros((2, 4), dtype=np.uint8)
    counts = np.ones(2, dtype=np.int64)
    with pytest.raises(TypeError, match="int64"):
        pair_edges(sequences, np.array([[0, 1]], dtype=np.int32), counts, 1, False)
    with pytest.raises(ValueError, match="pair indices"):
        pair_edges(sequences, np.array([[0, 2]], dtype=np.int64), counts, 1, False)
    with pytest.raises(ValueError, match="counts"):
        pair_edges(sequences, np.array([[0, 1]], dtype=np.int64), np.array([1], dtype=np.int64), 1, False)
    with pytest.raises(ValueError, match="threshold"):
        pair_edges(sequences, np.array([[0, 1]], dtype=np.int64), counts, -1, False)


def test_zero_length_umis_are_safe_for_the_native_path():
    counts = {b"": 4, b"": 4}
    # A mapping cannot contain two distinct empty byte strings, but the direct
    # API still supports its valid zero-width matrix shape.
    edges = pair_edges(np.empty((2, 0), dtype=np.uint8), np.array([[0, 1]], dtype=np.int64),
                       np.array([4, 1], dtype=np.int64), 0, True)
    assert edges.tolist() == [1]
