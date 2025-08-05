from tracemalloc import start
import regex as re
import multiprocessing
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.DoublyLinkedList import DoublyLinkedList, Node
from cs336_basics.PairHeap import PairHeap

bytes_pair = tuple[bytes, bytes]

class BPETokenizer:
    def __init__(self):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def pretokenize_chunk(self, input_path: str, start: int, end: int, special_tokens: list[str]) -> defaultdict[bytes, int]:
        """
        Reads a chunk of the input file and returns a pretokenized dictionary.
        """
        word_counts = defaultdict(int)
        with open(input_path, "rb") as file:
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            parts = re.split(f"({'|'.join(map(re.escape, special_tokens))})", chunk)
            for part in parts:
                if part in special_tokens:
                    continue
                if part == "":
                    continue
                for token in re.finditer(self.pattern, part):
                    token = token.group()
                    if token == "":
                        continue
                    elif token in special_tokens:
                        continue
                    else:
                        word_counts[token.encode("utf-8")] += 1
        return word_counts
    
    def init_vocab(self, special_tokens: list[str]) -> dict[int, bytes]:
        """
        Initializes the vocabulary with special tokens.
        """
        vocab = {}
        for idx, token in enumerate(special_tokens):
            vocab[idx] = token.encode("utf-8")  # Offset by 256 for special tokens
        _len = len(vocab)
        for i in range(256):
            vocab[_len + i] = bytes([i])  # Add single byte tokens
        return vocab
    
    def init_doubly_linked_list(self, word_count: defaultdict[bytes, int]) -> tuple[DoublyLinkedList, dict[bytes_pair, set[Node]], dict[bytes_pair, int]]:
        """
        Initializes a DoublyLinkedList with the word counts and prepares for merging.
        """
        dll = DoublyLinkedList()
        pair_nodes = defaultdict(set)
        pair_counts = defaultdict(int)

        # Add each word to the doubly linked list
        for word, count in word_count.items():
            word_bytes = word
            for i in range(len(word_bytes)):
                curr_bytes = word_bytes[i:i+1]
                dll.append(curr_bytes, count, word_end=(i == len(word_bytes) - 1))
        curr = dll.head
        while curr is not None and curr.next is not None:
            a = curr.data
            b = curr.next.data
            pair = (a, b)
            if curr.word_end == True:
                curr = curr.next
                continue
            pair_nodes[pair].add(curr)
            pair_counts[pair] += curr.count
            curr = curr.next
        return dll, pair_nodes, pair_counts
    
    def merge_pairs(self, dll: DoublyLinkedList, 
                    pair_nodes: dict[bytes_pair, set[Node]], 
                    pair_counts: dict[bytes_pair, int], 
                    vocab_size: int, vocab: dict[int, bytes]
    ) -> tuple[dict[int, bytes], list[bytes_pair]]:
        merges = []
        pair_heap = PairHeap(pair_counts)
        
        while len(vocab) < vocab_size:
            max_pair = pair_heap.pop()
            if max_pair is None:
                break
                
            b, c = max_pair
            new_token = b + c
            merges.append((b, c))
            vocab[len(vocab)] = new_token
            
            # Update the doubly linked list
            nodes_to_update = list(pair_nodes[max_pair])
            updated_freq = defaultdict(int)
            
            for node in nodes_to_update:
                if node.next is None or node.word_end:
                    continue
                    
                # 创建新的合并节点
                bc_node = Node(data=new_token, count=node.count, word_end=node.next.word_end)
                
                # 更新前向连接 a -> bc
                if node.prev is not None and not node.prev.word_end:
                    a_node = node.prev
                    a = a_node.data
                    
                    # 从旧pair中移除
                    pair_nodes[(a, b)].discard(a_node)
                    updated_freq[(a, b)] -= a_node.count
                    
                    # 添加到新pair
                    pair_nodes[(a, new_token)].add(a_node)
                    updated_freq[(a, new_token)] += a_node.count
                    
                    # 更新链接
                    a_node.next = bc_node
                    bc_node.prev = a_node
                else:
                    # bc_node是第一个节点
                    dll.head = bc_node if dll.head == node else dll.head
                
                # 更新后向连接 bc -> d
                if node.next.next is not None and not node.next.word_end:
                    d_node = node.next.next
                    d = d_node.data
                    
                    # 从旧pair中移除
                    pair_nodes[(c, d)].discard(node.next)
                    updated_freq[(c, d)] -= node.next.count
                    
                    # 添加到新pair
                    pair_nodes[(new_token, d)].add(bc_node)
                    updated_freq[(new_token, d)] += bc_node.count
                    
                    # 更新链接
                    bc_node.next = d_node
                    d_node.prev = bc_node
                else:
                    # bc_node是最后一个节点
                    bc_node.next = None
                    if node.next == dll.tail:
                        dll.tail = bc_node
            
            # 清除旧的pair
            pair_nodes[max_pair].clear()
            pair_heap.remove(max_pair)
            
            # 更新频率
            for pair, freq_change in updated_freq.items():
                if freq_change != 0:
                    current_freq = pair_heap.pair_counts.get(pair, 0)
                    new_freq = max(0, current_freq + freq_change)
                    pair_heap.update_freq(pair, new_freq)
            
        return vocab, merges

    def tokenize(self, 
                 input_path: str,
                 vocab_size: int,
                 special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[bytes_pair]]:
        """
        Tokenizes the input text file and returns a vocabulary and merges.
        """
        # Step 1: Read the file and pretokenize it
        num_process = 4
        with open(input_path, "rb") as file:
            # Find chunk boundaries
            boundaries = find_chunk_boundaries(file, num_process, b"<|endoftext|>")
            with multiprocessing.Pool(processes=num_process) as pool:
                # Read each chunk in parallel
                results = pool.starmap(self.pretokenize_chunk,
                                   [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
            word_count = defaultdict(int)
            for r in results:
                for word, count in r.items():
                    word_count[word] += count
        
        # Step 2: Create the vocabulary
        vocab = self.init_vocab(special_tokens)
        
        # Step 3: Initial DoublyLinkedList used for merging
        dll, pair_nodes, pair_counts = self.init_doubly_linked_list(word_count)
        
        # Step 4: Start Merge
        vocab, merges = self.merge_pairs(dll, pair_nodes, pair_counts, vocab_size, vocab)
        return vocab, merges
            

if __name__ == "__main__":
    bpe = BPETokenizer()
    input_path = "test.txt"
    vocab, merges = bpe.tokenize(
        input_path=input_path,
        vocab_size=263,
        special_tokens=["<|endoftext|>"]
    )
    print("Vocabulary:", vocab)
    print("Merges:", merges)