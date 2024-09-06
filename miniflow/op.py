from typing import List
import numpy as np
from miniflow.node import Node


class Op:  # abstract class, you should inherit this class to implement your own operator
    def __call__(self):  # Differ it from __init__()
        node = Node()
        node.op = self
        return node

    def compute(self, node, input_vals: List):
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

    def compute(self, node, input_vals: List):
        return None

    def gradient(self, node, output_grad):
        return None


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} + {node_B.name}"
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals: List):
        # y = a + b
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        # y = a+b => da = dy, db = dy
        # according to the formula above, you should cal grad_a and grad_b
        # and return [grad_a,grad_b]
        return [output_grad, output_grad]


class AddConstOp(Op):
    def __call__(self, node_A, const_attr):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} + {const_attr}"
        new_node.inputs = [node_A]
        new_node.const_attr = const_attr
        return new_node

    def compute(self, node, input_vals):
        # y = a + const
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        # da = dy
        return [output_grad]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} * {node_B.name}"
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        # y = a * b
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad) -> List[Node]:
        # y = a*b => da = dy*b, db = dy*a
        # according to the formula above, you should cal grad_a and grad_b
        # and return [grad_a,grad_b]

        # TODO: Write your code below
        grad_a = mul_op(node.inputs[1], output_grad)
        grad_b = mul_op(node.inputs[0], output_grad)
        return [grad_a, grad_b]


class MulConstOp(Op):
    def __call__(self, node, const_attr):
        new_node = Op.__call__(self)
        new_node.name = f"{node.name} * {const_attr}"
        new_node.inputs = [node]
        new_node.const_attr = const_attr
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad) -> List[Node]:
        # y = const*a => da = dy*const
        # according to the formula above, you should cal grad_a and grad_b
        # and return [grad_a]

        # TODO: Write your code below
        return [mul_const_op(output_grad, node.const_attr)]


class OnesLikeOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node

    def compute(self, node, input_vals):
        # y = np.full_like(y,1)
        # according to the formula above, you should cal grad_y
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad) -> List[Node]:
        # y = np.full_like(y,1) => dy = 0
        return [zerolike_op(node.inputs[0])]


class ZeroLikeOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node

    def compute(self, node, input_vals):
        # y = np.zeros_like(y)
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad) -> List[Node]:
        # y = np.zeros_like(y) => dy = 0
        return [zerolike_op(node.inputs[0])]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} MatMal {node_B.name}"
        new_node.trans_A = trans_A  # A or A^T
        new_node.trans_B = trans_B  # B or B^T
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        # y = a * b
        val_a, val_b = input_vals[0], input_vals[1]
        # according to the attr to decide the value of a and b
        # and return the result
        # TODO: Write your code below

        if node.trans_A:
            val_a = input_vals[0].T  # np.transpose(input_vals[0])
        if node.trans_B:
            val_b = input_vals[1].T

        return np.matmul(val_a, val_b)

    def gradient(self, node, output_grad) -> List[Node]:
        #  Y = A B => dA = dY B^T, dB = A^T dY
        #  According to the formula above, you should cal grad_a and grad_b
        #  and reuturn [grad_a, grad_b]

        # TODO: Write your code below
        grad_a = matmul_op(output_grad, node.inputs[1], trans_B=True)
        grad_b = matmul_op(node.inputs[0], output_grad, trans_A=True)

        return [grad_a, grad_b]


class ReduceSumOp(Op):
    def __call__(self, node, axis=0, keepdims=False):
        new_node = Op.__call__(self)
        new_node.name = f"ReduceSum({node.name})"
        new_node.inputs = [node]
        new_node.axis = axis  # default 0
        new_node.keepdims = keepdims  # default False
        return new_node

    def compute(self, node, input_vals):
        # y = np.sum(x, axis=axis, keepdims=keepdims)
        return np.sum(input_vals[0], axis=0)

    def gradient(self, node, output_grad):
        # y = np.sum(x, axis=axis, keepdims=keepdims) => dx = dy
        return [broadcast_op(output_grad, node.inputs[0])]


class BroadcastOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name = f"(Broadcast{node_A.name} to {node_B.name}'shape)"
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        # y = B + a # broadcast a to B's shape
        a_val = input_vals[0]
        B_val = input_vals[1]
        return np.broadcast_to(a_val, B_val.shape)

    def gradient(self, node, output_grad):
        # y = B + a => dB = dy, da = dy
        grad_a = reducesum_op(output_grad, axis=0, keepdims=False)
        grad_B = zerolike_op(node.inputs[1])
        return [grad_a, grad_B]


class ReLuOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.name = f"ReLu({node.name})"
        new_node.inputs = [node]
        return new_node

    def compute(self, node, input_vals):
        return np.maximum(input_vals[0], 0)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]


class ReLuGradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name = f"ReLuGradient({node_A.name})"
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        # y = Relu(a) => dy = 1 if a>0 else 0
        sign = np.where(input_vals[0] > 0, 1, 0)
        # sign = (np.sign(input_vals[0])+1)  # (-1 or 0 or 1) + 1 => (0 or 1 or 2)
        return input_vals[1]*sign

    def gradient(self, node, output_grad):
        pass  # ReLuGradientOp is the end of the gradient chain, no gradient to pass back


def softmax(x, axis=-1):
    # x shape (N,q)
    # return shape (N,q)
    new_x = x - np.max(x, axis=axis, keepdims=True)  # avoid overflow
    exp_x = np.exp(new_x)
    # numpy's broadcast mechanism
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SoftmaxOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.name = f"Softmax({node.name})"
        new_node.inputs = [node]
        return new_node
    def compute(self,node,input_vals):
        return softmax(input_vals[0])
    def gradient(self,node,output_grad):
        pass  # SoftmaxOp is the end of the gradient chain, no gradient to pass back
    
class SoftmaxCrossEntropyLoss(Op):
    def __call__(self, node_A, node_B) -> Node:
        new_node = Op.__call__(self)
        new_node.name = f"CrossEntryopyLoss({node_A.name},{node_B.name})"
        new_node.inputs = [node_A, node_B]
        new_node.is_mean = True
        return new_node

    def compute(self, node, input_vals):
        ''' Cross Entropy Loss
        loss(y,y_hat) = -sum(y*log(y_hat))
        y_hat = softmax(output)
        How to cal gradient of output?
        '''
        y_true = input_vals[0] 
        o = input_vals[1]
        y_hat = softmax(o) 
        cross_entropy_loss = -np.sum(y_true*np.log(y_hat), axis=1)
        if node.is_mean:
            cross_entropy_loss = np.mean(cross_entropy_loss)
        return cross_entropy_loss

    def gradient(self, node, output_grad):
        '''
        loss(y,y_hat) = -sum(y*log(y_hat))
        y_hat = softmax(o)
        
        do  = (softmax(o) - y)*d(loss)
        we donn't neet to calculate uptate of y
        dy = 0
        
        grad_o = (softmax(o) - y)*d(loss)
        '''
        grad_o = mul_op(add_op(softmax_op(node.inputs[1]),mul_const_op(node.inputs[0],-1)),output_grad)
        grad_y_true = zerolike_op(node.inputs[1])
        
        return [grad_y_true, grad_o]

# NOTION: Here, Instantiate the your operators
add_op = AddOp()
add_const_op = AddConstOp()
mul_op = MulOp()
mul_const_op = MulConstOp()
matmul_op = MatMulOp()
oneslike_op = OnesLikeOp()
zerolike_op = ZeroLikeOp()
broadcast_op = BroadcastOp()
reducesum_op = ReduceSumOp()
relu_op = ReLuOp()
relu_gradient_op = ReLuGradientOp()
softmax_op = SoftmaxOp()
softmax_cross_entropy_loss = SoftmaxCrossEntropyLoss()
