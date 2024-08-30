
from miniflow.executor import *
from miniflow.optimizer import SGD
import numpy as np
from miniflow.data import DataLoader,Dataset,load_mnist_data

def yield_infinite_data():
    def yield_data():
        while True:
            yield np.random.rand(batch_size, 3)
    return yield_data()

def convert_to_one_hot(vals):
    """Helper method to convert label array to one-hot array."""
    one_hot_vals = np.zeros((vals.size, 10))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

def train_loop_test():
    # define variables
    W = Variable('W')
    b = Variable('b')
    X = Variable('X')
    y_true = Variable('y_true')

    z = matmul_op(X, W)
    o = add_op(z, broadcast_op(b, z))
    # loss(y,y_hat) = -sum(y*log(y_hat))
    loss = softmax_cross_entropy_loss(y_true, o)

    # cal gradient
    grad_W, grad_b = gradient(loss, [W, b])
    # initialize params
    W_val = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=np.float32)
    b_val = np.array([1, 1, 1], dtype=np.float32)

    # initialize optimizer and engine
    sgd = SGD(learning_rate=0.01, params={W: W_val, b: b_val})
    executor = Executor([loss, grad_W, grad_b, o])
    # prepare data
    train_iter = yield_infinite_data()
    # train
    for epoch in range(num_epoch):
        X_batch = next(train_iter)
        # print(X_batch)
        Y_batch = np.array([[1, 0, 0], [0, 0, 1]])

        # model.forward
        loss_val, grad_W_val, grad_b_val, o_val = executor.run(
            {X: X_batch, W: W_val, b: b_val, y_true: Y_batch})

        # optimizer.step
        sgd.update({W: grad_W_val, b: grad_b_val})
        sgd.print_params()

        # cal loss
        print(f'loss:{loss_val}')
        print(f'\033[92m---------epoch:{epoch}--------\033[0m')

def train_loop_minist():
    
    #===================prepare data======================
    data = load_mnist_data('data/mnist.pkl.gz')
    train_data, train_labal = data[0]
    valid_data, valid_labal = data[1]
    test_data, test_labal = data[2]
    
    train_dataset = Dataset(train_data,train_labal)
    train_dataloader = DataLoader(train_dataset,batch_size)
    
    #===================define variables======================
    X = Variable('X')
    W = Variable('W')
    b = Variable('b')
    
    y_true = Variable('y_true')
    
    z = matmul_op(X,W)
    y = add_op(z,broadcast_op(b,z))
    loss = softmax_cross_entropy_loss(y_true,y)
    
    #===================initialize engine ======================
    gradient_W,gradient_b = gradient(loss,[W,b])
    executor = Executor([loss,gradient_W,gradient_b,y])
    #===================initialize params======================
    W_val = np.random.rand(784,10)
    b_val = np.random.rand(10)
    
 
    
    #===================initialize optimizer======================
    sgd = SGD(learning_rate=learning_rate,params={W:W_val,b:b_val})
    

    #===================train======================
    for i in range(num_epoch):
        print(f'\033[92m---------epoch:{i}--------\033[0m')
        cnt =0
        loss = 0
        for X_batch,y_batch in train_dataloader:
            
            cnt+=1
            y_batch = convert_to_one_hot(y_batch)
            # model.forward
            loss_val,grad_W_val,grad_b_val,y_val = executor.run({X:X_batch,W:W_val,b:b_val,y_true:y_batch})
            # optimizer.step
            sgd.update({W:grad_W_val,b:grad_b_val})
            if(cnt==1000):
                break
        # cal loss
        print(f'loss:{loss_val}')

    

if __name__ == "__main__":
    
    
    num_epoch = 20
    batch_size = 16
    learning_rate = 0.01
    train_loop_minist()
