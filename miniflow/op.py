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
    
    
class AddOp(Op):
    def __call__(self,node_A, node_B):
        new_node = super().__call__()
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] + input_vals[1]
    def gradient(self, node, output_grad):
        return [output_grad,output_grad]

class MulOp(Op):
    def __call__(self,node_A, node_B):
        new_node = super().__call__()
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] * input_vals[1]
    def gradient(self, node, output_grad):
        return [output_grad*node.inputs[1],output_grad*node.inputs[0]]