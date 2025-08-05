import heapq
from collections import defaultdict

class PairHeap:
    def __init__(self, pair_counts: defaultdict[tuple[bytes, bytes], int]):
        self.pair_counts = pair_counts
        self.heap = [(-freq, (-pair[0][0], -pair[1][0]), pair) for pair, freq in self.pair_counts.items() if freq > 0]
        heapq.heapify(self.heap)
        self.removed = set()

    def pop(self) -> tuple[bytes, bytes] | None:
        while self.heap:
            neg_freq, neg_pair_key, pair = heapq.heappop(self.heap)
            if pair in self.removed:
                self.removed.remove(pair)
                continue
            if neg_freq != -self.pair_counts[pair]:
                continue
            return pair
        return None

    def update_freq(self, pair: tuple[bytes, bytes], new_freq: int):
        self.pair_counts[pair] = new_freq
        heapq.heappush(self.heap, (-new_freq, (-pair[0][0], -pair[1][0]), pair))

    def remove(self, pair: tuple[bytes, bytes]):
        self.pair_counts[pair] = 0
        self.removed.add(pair)

if __name__ == "__main__":
    pair_counts = {
        (b'a', b'b'): 5,
        (b'b', b'c'): 5,
        (b'c', b'd'): 2,
    }
    pair_heap = PairHeap(defaultdict(int, pair_counts))
    print(pair_heap.pop())
    pair_heap.update_freq((b'c', b'd'), 4)
    print(pair_heap.pop())
