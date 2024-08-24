


class SGD:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate

    def update(self, params):
        # param :(val, grad_val)
        for param in params:
            param[0]+= -self.learning_rate * param[1]

def sgd_update_param( val, grad_val, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

    `trainables`: A list of `Input` nodes representing weights/biases.
    `learning_rate`: The learning rate.
    """
    val = val - learning_rate * grad_val