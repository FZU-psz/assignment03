from .op import *
from typing import Dict
class Executor:
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list
        self.graph = find_topo_sort(self.eval_node_list)

    def run(self, feed_dict:Dict):
        
        # calculate the value of each node in the graph
        node_to_val_map = {}
        for varaible, varaible_val in feed_dict.items():
            node_to_val_map[varaible] = varaible_val
        
        # Now you have the value of the input nodes(feed_dict) and computation graph(self.graph)
        # TODO:Traverse graph in topological order and compute values for all nodes,write you code below
        # hint: use node.op.compute(node, input_vals) to get value of node
        for node in self.graph:
            if node in node_to_val_map:
                continue
            
            input_vals = [node_to_val_map[inp] for inp in node.inputs]
            node_to_val_map[node] = node.op.compute(node, input_vals)

        # return the val of each node 
        return [node_to_val_map[node] for node in self.eval_node_list]

def gradient(output_node:Node, node_list:List[Node])->List[Node]:

    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    
    # Traverse the graph in reverse topological order and calculate the gradient for each variable
    for node in reverse_topo_order:
        #TODO: Sum the adjoints from all output nodes(hint: use sum_node_list)
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        
        #TODO: Calculate the gradient of the node with respect to its inputs
        
        input_grads_list = node.op.gradient(node, output_grad)
        
        #TODO: Traverse the inputs of node and add the partial adjoints to the list
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    # return the gradient of each node in node_list
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

# ========================
# NOTION: Helper functions
# ========================
def find_topo_sort(node_list)->List[Node]:
    """Given a list of nodes, return a topo ordering of nodes ending in them.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.
    """
    visited = set()
    topo_order = []

    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node:Node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from functools import reduce
    return reduce(add_op, node_list)
