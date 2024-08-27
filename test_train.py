
from miniflow.executor import *
from miniflow.optimizer import SGD
import numpy as np


def yield_infinite_data():
    def yield_data():
        while True:
            yield np.random.rand(batch_size,3)
    return yield_data()


def train_loop():
    # define variables
    W = Variable('W')
    b = Variable('b')
    X = Variable('X')
    y_true = Variable('y_true')
    
    z = matmul_op(X, W)
    o = add_op(z,broadcast_op(b, z))
    # loss(y,y_hat) = -sum(y*log(y_hat))
    loss = softmax_cross_entropy_loss(y_true,o)
    
    # cal gradient
    grad_W, grad_b = gradient(loss, [W, b])
    # initialize params 
    W_val = np.array([[2, 2,2], [2, 2,2],[2,2,2]], dtype=np.float32)
    b_val = np.array([1, 1,1], dtype=np.float32)

    # initialize optimizer and engine
    sgd = SGD(learning_rate=0.01, params={W: W_val, b: b_val})
    executor = Executor([loss, grad_W, grad_b,o])

    # prepare data
    train_iter = yield_infinite_data()

    # train
    for epoch in range(num_epoch):
        X_batch = next(train_iter)
        # print(X_batch)
        Y_batch = np.array([[1,0,0],[0,0,1]])
        
        # model.forward
        loss_val, grad_W_val, grad_b_val, o_val = executor.run(
            {X: X_batch, W: W_val, b: b_val,y_true:Y_batch})
        
        # optimizer.step
        sgd.update({W: grad_W_val, b: grad_b_val})
        sgd.print_params()
        
        # cal loss
        print(f'loss:{loss_val}')
        print(f'\033[92m---------epoch:{epoch}--------\033[0m')




if __name__ == "__main__":
    num_epoch = 10
    batch_size = 2
    learning_rate = 0.01
    train_loop()
