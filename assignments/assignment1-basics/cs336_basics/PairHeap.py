import heapq

class PrioritizedPair:
    def __init__(self, freq, pair):
        self.freq = freq
        self.pair = pair  # e.g., ('a', 'b')

    def __lt__(self, other):
        # Higher frequency comes first
        if self.freq != other.freq:
            return self.freq > other.freq
        # Lexicographically greater pair comes first
        # print(self.pair, other.pair)
        self.bytes = b''.join(self.pair)
        other.bytes = b''.join(other.pair)
        return self.bytes > other.bytes

    def __repr__(self):
        return f"({self.pair}, {self.freq})"

class PairHeap:
    def __init__(self, pair_count):
        self.heap = []
        for pair, freq in pair_count.items():
            heapq.heappush(self.heap, PrioritizedPair(freq, pair))
        self.pair_count = pair_count
        self.removed_pairs = set()

    def remove_pair(self, pair):
        self.removed_pairs.add(pair)

    def clear_heap(self):
        while True:
            if len(self.heap) == 0:
                break
            item = self.heap[0]
            if item.pair in self.removed_pairs:
                heapq.heappop(self.heap)
            elif item.freq != self.pair_count[item.pair]:
                heapq.heappop(self.heap)
            else:
                break

    def pop(self):
        self.clear_heap()
        if not self.heap:
            return None
        item = heapq.heappop(self.heap)
        return item
    
    def peek(self):
        self.clear_heap()
        return self.heap[0] if self.heap else None
    
    def update_count(self, pair, new_freq):
        self.pair_count[pair] = new_freq
        heapq.heappush(self.heap, PrioritizedPair(new_freq, pair))

if __name__ == "__main__":
    pair_count = {(b'a', b'b'): 5, (b'a', b'c'): 5, (b'c', b'd'): 2}
    pair_heap = PairHeap(pair_count)
    
    item = pair_heap.pop()
    print(item)  # Should print the pair with the highest frequency
    pair_heap.update_count((b'c', b'd'), 10)
    item = pair_heap.pop()
    print(item)  # Should print the updated pair with frequency 10
