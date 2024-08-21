from typing import List
import numpy as np
from node import Node

class Op:
    def __call__(self):
        node = Node()
        node.op = self
        return node
    def compute(self, node, input_vals:List):
        raise NotImplementedError
    def gradient(self, node, output_grad):
        raise NotImplementedError

def Variable(name):
    node = PlaceHolderOp()()
    node.name = name
    return node

class PlaceHolderOp(Op):
    def __call__(self):
        node = Op.__call__(self)
        node.op = self
        return node
    def compute(self, node, input_vals:List):
        return None
    def gradient(self, node, output_grad):
        return None

class AddOp(Op):
    def __call__(self,node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name  = f"{node_A.name} + {node_B.name}"
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals:List):
        return input_vals[0] + input_vals[1]
    def gradient(self, node, output_grad):
        return [output_grad,output_grad]
    
class AddConstOp(Op):
    def __call__(self,node_A, const_attr):
        new_node = Op.__call__(self)
        new_node.name  = f"{node_A.name} + {const_attr}"
        new_node.inputs = [node_A]
        new_node.const_attr = const_attr
        return new_node
    def compute(self,node,input_vals):
        return input_vals[0] + node.const_attr
    def gradient(self,node,output_grad):
        return [output_grad]
    
class MulOp(Op):
    def __call__(self,node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name  = f"{node_A.name} * {node_B.name}"
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals):
        return input_vals[0] * input_vals[1]
    def gradient(self, node, output_grad):
        grad_1 = mul_op(node.inputs[0],output_grad)
        grad_2 = mul_op(node.inputs[1],output_grad)
        return [grad_1,grad_2]

class MulConstOp(Op):
    def __call__(self,node,const_attr):
        new_node = Op.__call__(self)
        new_node.name  = f"{node.name} * {const_attr}"
        new_node.inputs = [node]
        new_node.const_attr = const_attr
        return new_node
    def compute(self,node,input_vals):
        return input_vals[0] * node.const_attr
    def gradient(self,node,output_grad):
        return [mul_const_op(output_grad,node.const_attr)]
    
class OnesLikeOp(Op):
    def __call__(self,node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node
    def compute(self, node, input_vals):
        return np.ones(input_vals[0].shape)
    def gradient(self, node, output_grad):
        return [zerolike_op(node.inputs[0])]
    
class ZeroLikeOp(Op):
    def __call__(self,node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node
    def compute(self, node, input_vals):
        return np.zeros(input_vals[0].shape)
    def gradient(self, node, output_grad):
        return [zerolike_op(node.inputs[0])]
    
add_op = AddOp()
add_const_op = AddConstOp()
mul_op = MulOp()
mul_const_op = MulConstOp()
oneslike_op = OnesLikeOp()
zerolike_op = ZeroLikeOp()