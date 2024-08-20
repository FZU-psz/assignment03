
class Executor:
    def __init__(self, eval_node_list):
        self.eval_node_list =  eval_node_list 
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
            # 
            node_to_val_map[node] = node.op.compute(node, input_vals)
        
        return [node_to_val_map[node] for node in self.eval_node_list] 

def gradient(output_node, node_list):
    pass


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
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)