"""Native pairwise UMI edge construction for the Python API."""

from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.uint8]()


@export("mut_pair_edges")
def mut_pair_edges(seqs_addr: Int, pairs_addr: Int, pair_count: Int, umi_length: Int,
                   threshold: Int, counts_addr: Int, directional: Int,
                   edges_addr: Int) abi("C"):
    """Set an edge bitmask for each candidate UMI pair.

    Bit 1 denotes first-to-second and bit 2 second-to-first.  For undirected
    adjacency both bits are set.  The caller owns every buffer.
    """
    seqs = BPtr(unsafe_from_address=seqs_addr)
    pairs = IPtr(unsafe_from_address=pairs_addr)
    counts = IPtr(unsafe_from_address=counts_addr)
    edges = BPtr(unsafe_from_address=edges_addr)
    def process_pair(p: Int) {seqs, pairs, counts, edges, umi_length, threshold, directional}:
        var left = Int(pairs.load(2 * p))
        var right = Int(pairs.load(2 * p + 1))
        var distance = 0
        var k = 0
        while k + W <= umi_length and distance <= threshold:
            var differing = seqs.load[width=W](left * umi_length + k) ^ seqs.load[width=W](right * umi_length + k)
            var zero_bits = ((differing - SIMD[DType.uint8, W](1)) & ~differing) & SIMD[DType.uint8, W](128)
            distance += W - Int(zero_bits.cast[DType.int64]().reduce_add()) // 128
            k += W
        while k < umi_length and distance <= threshold:
            if seqs.load(left * umi_length + k) != seqs.load(right * umi_length + k):
                distance += 1
            k += 1
        var edge: UInt8 = 0
        if distance <= threshold:
            if directional == 0:
                edge = 3
            else:
                # For non-negative Int64 counts, this is exactly
                # left >= 2 * right - 1 without overflowing at Int64.max.
                if counts.load(left) // 2 + counts.load(left) % 2 >= counts.load(right):
                    edge |= 1
                if counts.load(right) // 2 + counts.load(right) % 2 >= counts.load(left):
                    edge |= 2
        edges.store(p, edge)

    var p = 0
    while p < pair_count:
        process_pair(p)
        p += 1
