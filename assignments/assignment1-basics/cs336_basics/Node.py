"""
Doubly Linked List Node class for use in a tokenizer.
"""
from collections import defaultdict

class Node:
    def __init__(self, value: bytes, word_end: bool, word_count: int):
        self.value = value  # bytes 类型
        self.prev = None
        self.next = None
        self.word_end = word_end
        self.word_count = word_count

    def __str__(self):
        try:
            val_str = self.value.decode("utf-8")
        except:
            val_str = str(self.value)
        return f"Node(value={val_str}, word_end={self.word_end}, word_count={self.word_count})"

    def __repr__(self):
        return self.__str__()


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def build_words(self, word_count: defaultdict[bytes, int], special_tokens: set[bytes]):
        """
        Build a doubly linked list from word_count dict.
        Each word is a tuple of bytes tokens (each token is a full byte like b'a')
        """
        for word, occr in word_count.items():
            if word in special_tokens:
                new_node = Node(value=word, word_end=True, word_count=occr)
                if self.head is None:
                    self.head = new_node
                    self.tail = new_node
                else:
                    self.tail.next = new_node
                    new_node.prev = self.tail
                    self.tail = new_node
            else:
                for i in range(len(word)):
                    byte_token = word[i]  # 这是 int
                    new_node = Node(
                        value=bytes([byte_token]),  # 转为真正的 bytes 单字符
                        word_end=(i == len(word) - 1),
                        word_count=occr
                    )
                    if self.head is None:
                        self.head = new_node
                        self.tail = new_node
                    else:
                        self.tail.next = new_node
                        new_node.prev = self.tail
                        self.tail = new_node
