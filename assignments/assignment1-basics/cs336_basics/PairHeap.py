import heapq
from collections import defaultdict

class PairHeap:
    def __init__(self, pair_counts: defaultdict[tuple[bytes, bytes], int]):
        self.pair_counts = pair_counts
        self.heap = [(-freq, self._neg_lex(pair), pair)
                     for pair, freq in self.pair_counts.items() if freq > 0]
        heapq.heapify(self.heap)
        self.removed = set()

    def _neg_lex(self, pair: tuple[bytes, bytes]) -> tuple[int, ...]:
        combined = pair[0] + pair[1]
        # 反转字节序列，然后取负值
        # 这样字典序大的会变成字典序小的（在反转后的序列中）
        reversed_combined = combined[::-1]
        return tuple(-b for b in reversed_combined)

    def pop(self) -> tuple[bytes, bytes] | None:
        while self.heap:
            neg_freq, neg_pair_key, pair = heapq.heappop(self.heap)
            if pair in self.removed:
                self.removed.remove(pair)
                continue
            # 检查频率是否仍然匹配（避免过期条目）
            if neg_freq != -self.pair_counts[pair]:
                continue
            return pair
        return None

    def update_freq(self, pair: tuple[bytes, bytes], new_freq: int):
        if new_freq > 0:
            self.pair_counts[pair] = new_freq
            heapq.heappush(self.heap, (-new_freq, self._neg_lex(pair), pair))
        else:
            # 如果频率为0或负数，相当于删除
            self.remove(pair)

    def remove(self, pair: tuple[bytes, bytes]):
        """标记pair为已删除"""
        self.pair_counts[pair] = 0
        self.removed.add(pair)

if __name__ == "__main__":
    # 测试用例
    pair_counts = {
        (b'a', b'b'): 5,
        (b'a', b'br'): 5,  # 频率相同，但字典序更大
        (b'x', b'y'): 3
    }
    pair_heap = PairHeap(defaultdict(int, pair_counts))
    print(pair_heap.pop())