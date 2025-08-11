from collections import defaultdict

class Node:
    def __init__(self, data, word_end, freq):
        self.data = data
        self.prev = None
        self.next = None
        self.word_end = word_end
        self.freq = freq

    def __repr__(self):
        return f"Node(data={self.data}, word_end={self.word_end}, freq={self.freq})"
    
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def add_node(self, data, word_end, freq):
        """
        Add a new node to the end of the list.
        """
        new_node = Node(data, word_end, freq)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.size += 1

    def add_from_words(self, word_count: dict[list[bytes], int]):
        """
        Add nodes from a word count dictionary.
        Each key is a tuple of bytes, and the value is the frequency.
        """
        for word, freq in word_count.items():
            for i in  range(len(word)):
                data = word[i]
                word_end = (i == len(word) - 1)
                self.add_node(data, word_end, freq)

    def __repr__(self):
        """
        String representation of the doubly linked list.
        """
        nodes = []
        current = self.head
        while current:
            nodes.append(f"{current.data}({current.freq})")
            current = current.next
        return " <-> ".join(nodes)

if __name__ == "__main__":
    # Example usage
    dll = DoublyLinkedList()
    word_count = {(b'h', b'r', b'e'): 3, (b'w', b'o', b'r', b'l', b'd'): 2}
    dll.add_from_words(word_count)
    print(dll)

    a = b'a'
    b = b'b'
    print(a + b)