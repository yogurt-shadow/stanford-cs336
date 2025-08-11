import regex as re
import pickle
from typing import Iterable, Iterator
from cs336_basics.BPETrainer import str2bytes

def split_by_special_tokens(
    text: str, special_tokens: list[str]
) -> list[str]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    if not special_tokens_sorted:
        return [text]
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)
    special_chunks = re.split(f"({pattern})", text)

    return special_chunks

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = []
    ):
        self.vocab = vocab
        self.merges = merges
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # self.register_special_tokens(special_tokens)
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        
        if special_tokens is None:
            self.special_tokens = {}
            self.bytes_special_tokens = []
        else:
            self.special_tokens = {token: i for i, token in enumerate(special_tokens, start=len(self.vocab))}
            self.bytes_special_tokens = [token.encode("utf-8") for token in special_tokens if isinstance(token, str)]
        
    def _pre_tokenize(self, text) -> list[bytes]:
        """
        Pre-tokenize the input text into bytes.
        """
        parts = split_by_special_tokens(text, list(self.special_tokens.keys()))
        token_list = []
        
        for part in parts:
            if part in self.special_tokens.keys():
                token_list.append(part.encode("utf-8"))
            else:
                tokens = re.findall(self.PAT, part)
                token_list.extend(str2bytes(token) for token in tokens)

        return token_list
    

    
    def encode(self, text: str) -> list[int]:
        byte_tokens = self._pre_tokenize(text)
        # Convert byte tokens to indices
        token_ids = []
        for byte_token in byte_tokens:
            # print(f"Processing byte token: {byte_token}")
            if byte_token in self.bytes_special_tokens:
                token_ids.append([self.vocab_inv[byte_token]])
            else:
                token_ids.append([self.vocab_inv[b] for b in byte_token]) #type: ignore

        for i, pretoken in enumerate(token_ids):
            for merge in self.merges:
                new_index = self.vocab_inv.get(merge[0] + merge[1], None)
                if new_index is None:
                    continue

                merged = []
                j = 0
                while j < len(pretoken):
                    if (
                        j < len(pretoken) - 1
                        and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]]) == merge
                    ):
                        merged.append(new_index)
                        j += 2
                    else:
                        merged.append(pretoken[j])
                        j += 1
                        
                pretoken = merged
            token_ids[i] = pretoken[:]

        return [i for pre in token_ids for i in pre]
        

    

    def encode_iterable(self, iterable: Iterable[str], batch_size: int = 1024) -> Iterator[int]:
        """
        Encode lines of text from an iterable using buffered batching.
        This version preserves newlines by assuming the input was split with `splitlines(keepends=True)`.
        """
        batch = []
        for line in iterable:
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                for encoded in map(self.encode, batch):
                    yield from encoded
                batch.clear()
                
        if batch:
            for encoded in map(self.encode, batch):
                yield from encoded
    
    def decode(self, ids: list[int]) -> str:
        # https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character
        
        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace")
    
    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_path, 'rb') as vf:
            raw_vocab = pickle.load(vf)

        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}

        with open(merges_path, 'rb') as mf:
            raw_merges = pickle.load(mf)

        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)