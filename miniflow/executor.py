from op import *

class Executor:
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list
        self.graph = find_topo_sort(self.eval_node_list)

    def run(self, feed_dict):

        # calculate the value of each node in the graph
        node_to_val_map = {}
        for varaible, varaible_val in feed_dict.items():
            node_to_val_map[varaible] = varaible_val

        # forwad pass the graph
        for node in self.graph:
            if node in node_to_val_map:
                continue
            input_vals = [node_to_val_map[inp] for inp in node.inputs]
            node_to_val_map[node] = node.op.compute(node, input_vals)

        return [node_to_val_map[node] for node in self.eval_node_list]


def gradient(output_node, node_list):

    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        # sum the adjoints from all output nodes
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

# ========================
# Below: Helper functions
# ========================
def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.
    """
    visited = set()
    topo_order = []

    for node in node_list:
        # print(node)
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
    return reduce(mul_op, node_list)
