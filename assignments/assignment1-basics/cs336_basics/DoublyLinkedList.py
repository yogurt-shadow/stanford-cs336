from collections import defaultdict

class Node:
    def __init__(self, data: bytes, count: int, word_end: bool):
        self.data = data
        self.prev = None
        self.next = None
        self.count = count
        self.word_end = word_end  # Indicates if this node represents a complete word

    def __repr__(self):
        return f"[{self.data!r} (count: {self.count}, word_end: {self.word_end})]"

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data: bytes, count: int, word_end: bool):
        new_node = Node(data, count, word_end)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def __repr__(self):
        nodes = []
        current = self.head
        while current:
            nodes.append(repr(current))
            current = current.next
        return " <-> ".join(nodes)
