"""Native pairwise UMI edge construction for the Python API."""

from std.runtime.asyncrt import TaskGroup, initialize_runtime, parallelism_level
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.uint8]()
comptime PARALLEL_PAIR_THRESHOLD = 65536
comptime PARALLEL_TASKS = 16


@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int, max_workers: Int):
    var worker_count = min(num_work_items, min(max_workers, parallelism_level()))
    var chunk_size, extra_items = divmod(num_work_items, worker_count)

    async def work_chunk(worker: Int, chunk_size: Int, extra_items: Int):
        var begin = worker * chunk_size + min(worker, extra_items)
        var count = chunk_size + Int(worker < extra_items)
        for item in range(begin, begin + count):
            func(item)

    var tasks = TaskGroup()
    for worker in range(worker_count):
        tasks.create_task(work_chunk(worker, chunk_size, extra_items))
    tasks.wait()


def pair_edge(seqs: BPtr, left: Int, right: Int, umi_length: Int,
              threshold: Int, counts: IPtr, directional: Int) -> UInt8:
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
            var left_count = counts.load(left)
            var right_count = counts.load(right)
            if left_count // 2 + left_count % 2 >= right_count:
                edge |= 1
            if right_count // 2 + right_count % 2 >= left_count:
                edge |= 2
    return edge


@export("mut_pair_edges")
def mut_pair_edges(seqs_addr: Int, pairs_addr: Int, pair_count: Int, umi_length: Int,
                   threshold: Int, counts_addr: Int, directional: Int,
                   edges_addr: Int) abi("C"):
    """Set an edge bitmask for each candidate UMI pair.

    Bit 1 denotes first-to-second and bit 2 second-to-first.  For undirected
    adjacency both bits are set.  The caller owns every buffer.
    """
    var seqs = BPtr(unsafe_from_address=seqs_addr)
    var pairs = IPtr(unsafe_from_address=pairs_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var edges = BPtr(unsafe_from_address=edges_addr)

    @parameter
    def process_pair(p: Int):
        var left = Int(pairs.load(2 * p))
        var right = Int(pairs.load(2 * p + 1))
        edges.store(p, pair_edge(seqs, left, right, umi_length, threshold, counts, directional))

    if pair_count >= PARALLEL_PAIR_THRESHOLD:
        initialize_runtime()
        var task_count = min(PARALLEL_TASKS, pair_count)

        @parameter
        def process_chunk(task: Int):
            var first = task * pair_count // task_count
            var last = (task + 1) * pair_count // task_count
            for p in range(first, last):
                process_pair(p)

        parallelize[process_chunk](task_count, task_count)
    else:
        for p in range(pair_count):
            process_pair(p)


@export("mut_dense_pair_edges")
def mut_dense_pair_edges(seqs_addr: Int, umi_count: Int, umi_length: Int,
                         threshold: Int, counts_addr: Int, directional: Int,
                         pairs_addr: Int, edges_addr: Int) abi("C"):
    """Generate every upper-triangle pair and its edge without NumPy temporaries."""
    var seqs = BPtr(unsafe_from_address=seqs_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var pairs = IPtr(unsafe_from_address=pairs_addr)
    var edges = BPtr(unsafe_from_address=edges_addr)
    var pair_count = umi_count * (umi_count - 1) // 2

    @parameter
    def process_left(left: Int):
        var position = left * (2 * umi_count - left - 1) // 2
        for right in range(left + 1, umi_count):
            pairs.store(2 * position, Int64(left))
            pairs.store(2 * position + 1, Int64(right))
            edges.store(position, pair_edge(
                seqs, left, right, umi_length, threshold, counts, directional
            ))
            position += 1

    if pair_count >= PARALLEL_PAIR_THRESHOLD:
        initialize_runtime()
        parallelize[process_left](umi_count, PARALLEL_TASKS)
    else:
        for left in range(umi_count):
            process_left(left)
