# mojo-umi-tools

`mojo-umi-tools` is a standalone Mojo port of the compute-heavy UMI network
deduplication core from [UMI-tools](https://umi-tools.readthedocs.io/).  It
keeps the public `UMIClusterer(cluster_method=...)` constructor and
`clusterer(umis, threshold)` call contract, under the import name
`mojoumitools` so it can live beside the upstream package during migration.

## Covered subset

`UMIClusterer` implements all five UMI-tools grouping methods:

| method | behaviour |
| --- | --- |
| `unique` | one group per exact UMI |
| `percentile` | retain UMIs above 1% of the median count |
| `cluster` | connected components of the edit-distance graph |
| `adjacency` | count-ranked, one-hop adjacency groups |
| `directional` | count-aware directed graph, UMI-tools' default |

The public network helpers `edit_distance`, `breadth_first_search`,
`remove_umis`, `get_substr_slices`, `build_substr_idx`, and
`iter_nearest_neighbours` are included as well. The Cython/Mojo work is the
Hamming-distance comparison and directional-edge construction; the Python
layer retains UMI-tools' graph/group semantics and its substring index.

This does not port the BAM/FASTQ command-line programs (`dedup`, `group`,
`count`, `extract`, `whitelist`), read selection, or barcode-whitelist logic.

## Install and use

```bash
pixi install
pixi run build
```

```python
from mojoumitools import UMIClusterer

umis = {b"ATAT": 10, b"GTAT": 5, b"CCAT": 3}
groups = UMIClusterer(cluster_method="directional")(umis, threshold=1)
assert groups == [[b"ATAT", b"GTAT"], [b"CCAT"]]
```

`pixi run test` builds the library and runs parity tests against upstream
`umi-tools` 1.1.6. The Mojo nightly is pinned to Python 3.13, while upstream
1.1.6 only ships a compatible Conda build for Python 3.9, so Pixi resolves an
isolated `upstream` test environment for that package. The shared library is
ABI-independent and is built first by the default environment.

## Benchmarks

Measured with `pixi run bench` on Linux 6.8.0-136-generic x86_64, glibc 2.39,
using Python 3.9.23 and 1,000 12-base UMIs chosen to form dense substring
candidate buckets. Values are best of three full end-to-end clustering calls.

| case | mojo-umi-tools | umi-tools 1.1.6 | ratio |
| --- | ---: | ---: | ---: |
| `directional` (1,000 dense candidates) | 10.1 ms | 161.7 ms | 16.08x faster |
| `adjacency` (1,000 dense candidates) | 12.7 ms | 349.6 ms | 27.60x faster |
| `cluster` (1,000 dense candidates) | 13.2 ms | 173.2 ms | 13.13x faster |

The edge kernel uses SIMD for full byte-vector blocks and a scalar tail. Dense
candidate sets are generated directly into their final NumPy buffers by Mojo,
avoiding temporary triangle and stacked-index arrays. Independent pair work is
parallelized at 65,536 pairs and remains serial below that threshold. No GPU mode
is provided: Hamming edge construction has under two operations per byte moved
and is branch-heavy, so host-device transfer would lose to the CPU path. Re-run
the measured table with `pixi run bench`; the task takes a machine-wide flock
before timing.

## How it works

`src/capi.mojo` is one compilation unit exported as
`dist/libmojo-umi-tools.so`. NumPy packs equal-length byte UMIs into an
`(n, length)` contiguous `uint8` matrix and candidate pairs into contiguous
`int64` indices. ctypes passes their addresses as C `Int` values; Mojo rebuilds
typed pointers inside the ABI wrapper and writes one `uint8` edge bitmask per
pair. Bit 1 is left-to-right and bit 2 is right-to-left, which represents both
undirected and directional network methods without allocating in Mojo.

The Python layer owns every allocation and turns edge masks back into the
same adjacency dictionaries, connected components, and representative-first
groups used by UMI-tools. `build/build.sh` is the explicit build route;
`mojoumitools` also rebuilds a stale or absent shared library on first use.

## Verification

```bash
pixi run build && pixi run test && pixi run bench
```

The test suite has direct upstream parity coverage for every clustering method,
both edit thresholds, the large-UMI substring index path, adjacency lists, and
the documented directional example.

## License

MIT. See [LICENSE](LICENSE).
