
from miniflow.executor import *
from miniflow.optimizer import SGD
import numpy as np

def yield_infinite_data():
    def yield_data():
        while True:
            yield np.random.rand(2, 2)
    return yield_data()

def train_loop():
    # define variables
    W = Variable('W')
    b = Variable('b')
    # define target 
    X = Variable('X')
    z = matmul_op(X, W)
    y = add_op(z, b)
    
    # cal gradient
    grad_W, grad_b = gradient(y, [W, b])
    
    # initialize 
    W_val = np.array([[2,2], [2, 2]],dtype=np.float32)
    b_val = np.array([1, 1],dtype=np.float32)
    

    # initialize optimizer
    sgd = SGD(learning_rate=0.01,params = {W: W_val, b: b_val})
    # prepare data
    train_iter = yield_infinite_data()
    
    # train
    for epoch in range(num_epoch):
        X_batch = next(train_iter)
        y_val, grad_W_val, grad_b_val = Executor([y, grad_W, grad_b]).run({X: X_batch, W: W_val, b: b_val})
        sgd.update({W: grad_W_val, b: grad_b_val})
        sgd.print_params()
        print('-----------------')




if __name__ == "__main__":
    
    num_epoch = 2
    batch_size = 2
    learning_rate = 0.01
    
    train_loop()