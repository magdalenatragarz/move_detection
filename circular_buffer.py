class RingBuffer:
    def __init__(self, size):
        self.data = [-1 for i in range(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data