from miniflow.node import Node
class Op:
    def __call__(self):
        node  = Node()
        node.op = self
        return node
    def compute(self):
        raise NotImplementedError
    def gradient(self):
        raise NotImplementedError