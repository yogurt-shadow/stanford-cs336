import json
import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
from typing import Set

def str2bytes(s: str) -> list[bytes]:
    """
    Convert a string to a list of bytes.
    """
    return [bytes([c]) for c in s.encode('utf-8')]


class BPETrainer:
    def __init__(self):
        self.train_naive = True
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def init_vocab(self, special_tokens: list[str]) -> dict:
        """
        Initialize the vocabulary with special tokens.
        Returns a dictionary mapping indices to tokens.
        """
        vocab = {i: bytes([i]) for i in range(256)}  # Initialize with single-byte tokens
        for idx, token in enumerate(special_tokens, start=256):
            vocab[idx] = token.encode('utf-8') if isinstance(token, str) else token
        return vocab

    def count_words_in_chunk(self,
                                input_path: str,
                                start: int,
                                end: int,
                                special_tokens: list[str]
    ) -> dict[tuple[bytes], int]:
        """
        Count word frequencies in a chunk of the input file.
        Returns a dictionary mapping tokens to their frequencies.
        """
        word_count = defaultdict(int)
        with open(input_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            pattern = "|".join(re.escape(token) for token in special_tokens)
            for part in re.split(f"({pattern})", chunk):
                if part in special_tokens:
                    continue
                else:
                    for match in re.finditer(self.PAT, part):
                        word = match.group(0)
                        token = tuple(str2bytes(word))
                        word_count[token] += 1
        return word_count

    def init_pair_count(self, word_count: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
        """
        Initialize pair counts from the word count.
        Returns a dictionary mapping pairs of tokens to their frequencies.
        """
        pair_count = defaultdict(int)
        for token, freq in word_count.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_count[pair] += freq
        return pair_count

    def merge_pairs_naive(self,
                    vocab: dict[int, bytes],
                    pair_count: dict[tuple[bytes, bytes], int],
                    word_count: dict[tuple[bytes], int],
                    vocab_size: int
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Merge the most frequent pairs in the vocabulary.
        Returns the updated vocabulary and a list of merges.
        Naive implementation that does not handle frequency updates.
        """
        merges = []
        while len(vocab) < vocab_size:
            max_freq = max(pair_count.values())
            best_pair = max([pair for pair, freq in pair_count.items() if freq == max_freq])
            if best_pair is None:
                break
            idx = len(vocab)
            vocab[idx] = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            new_word_count, new_pair_count = defaultdict(int), defaultdict(int)
            for word, count in word_count.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_count[tuple(new_word)] += count
                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j + 1])
                    new_pair_count[new_pair] += count
            word_count, pair_count = new_word_count, new_pair_count
        return vocab, merges

    def train_bpe(self, input_path: str,
                    vocab_size: int,
                    special_tokens: list[str]              
    ) -> tuple[dict, list]:
        """
        Train a BPE tokenizer on the input text file.
        Returns a vocabulary mapping and a list of merges.
        """
        # Step 1. Read the input file and count word frequencies
        num_process = 4
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
        with mp.Pool(processes=num_process) as pool:
            results = pool.starmap(self.count_words_in_chunk, [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
        word_count = defaultdict(int)
        for result in results:
            for token, count in result.items():
                word_count[token] += count
        # Step 2. Initialize the vocabulary
        vocab = self.init_vocab(special_tokens)
        # Step 3. Init pair-counting
        pair_count = self.init_pair_count(word_count)
        # Step 4. Merge pairs
        vocab, merges = self.merge_pairs_naive(vocab, pair_count, word_count, vocab_size)
        return vocab, merges
    