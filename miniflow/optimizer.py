


class SGD:
    def __init__(self, learning_rate=1e-2,params={}):
        self.learning_rate = learning_rate
        self.params = params
    def update(self,grad_map):
        # param :(node,val)
        for node,val in self.params.items():
            self.params[node] = self.params[node] - self.learning_rate * grad_map[node]
            
    def print_params(self):
        for node,val in self.params.items():
            print(f'{node}:\n{val}')
            
def sgd_update_param( val, grad_val, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

    `trainables`: A list of `Input` nodes representing weights/biases.
    `learning_rate`: The learning rate.
    """
    val = val - learning_rate * grad_val