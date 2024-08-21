
from node import *
from executor import *

def test_var_add_var():
    x1 = Variable('x1')
    x2 = Variable('x2')
    y = add_op(x1,x2)

    grad_x1,grad_x2 = gradient(y, [x1,x2])
    
    # print(type(grad_x1), type(grad_x2))
    x1_val = np.array([2,1])
    x2_val = np.array([1,1])
    y_val, x1_grad_val, x2_grad_val= Executor([y, grad_x1,grad_x2]).run({x1:x1_val,x2:x2_val})
    print(y_val, x1_grad_val, x2_grad_val)
    
def test_var_add_const():
    x1 = Variable('x1')
    y = add_const_op(x1,3) 
    grad_x1=gradient(y, [x1])[0]

    x1_val = np.array([1,1])
    y_val,x1_grad_val = Executor([y, grad_x1]).run({x1:x1_val})
    print(y_val,x1_grad_val)

def test_var_mul_var():
    x1 = Variable('x1')
    x2 = Variable('x2')
    y = mul_op(x1,x2)

    grad_x1,grad_x2 = gradient(y, [x1,x2])
    
    x1_val = np.array([1,2,3])
    x2_val = np.array([4,5,6])
    y_val, x1_grad_val, x2_grad_val= Executor([y, grad_x1,grad_x2]).run({x1:x1_val,x2:x2_val})
    print(y_val, x1_grad_val, x2_grad_val)

def test_var_mul_const():
    x1 = Variable('x1')
    y = mul_const_op(x1,4) 
    grad_x1=gradient(y, [x1])[0]
    x1_val = np.array([1,2,3])
    y_val,x1_grad_val = Executor([y, grad_x1]).run({x1:x1_val})
    print(y_val,x1_grad_val)

def test_add_mul_mix_1():
    x1 = Variable('x1')
    x2 = Variable('x2')
    # y = x1+x1*x2
    y = add_op(x1, mul_op(x1,x2))
    
    grad_x1,grad_x2 = gradient(y, [x1,x2])
    
    x1_val = 1*np.array([1,1,1])
    x2_val = 2*np.array([1,1,1])
    y_val, x1_grad_val, x2_grad_val= Executor([y, grad_x1,grad_x2]).run({x1:x1_val,x2:x2_val})
    print(y_val,x1_grad_val,x2_grad_val)
    
def test_add_mul_mix_2():
    x1 = Variable('x1')
    x2 = Variable('x2')
    x3 = Variable('x3') 
    # y = x1*x2 + x2*x3
    y = add_op(mul_op(x1,x2),mul_op(x2,x3))
    
    grad_x1,grad_x2,grad_x3 = gradient(y, [x1,x2,x3])
    
    x1_val = 1*np.array([1,1,1])
    x2_val = 2*np.array([1,1,1])
    x3_val = 3*np.array([1,1,1])
    y_val, x1_grad_val, x2_grad_val , x3_grad_val= Executor([y, grad_x1,grad_x2,grad_x3]).run({x1:x1_val,x2:x2_val,x3:x3_val})
    print(y_val,x1_grad_val,x2_grad_val,x3_grad_val)
    
def test_add_mul_mix_3():
    x1 = Variable('x1')
    x2 = Variable('x2')
    x3 = Variable('x3')
    # y = x1+x1*x2*x3
    y = add_op(x1, mul_op(x1,mul_op(x2,x3)))
    
    grad_x1,grad_x2,grad_x3= gradient(y, [x1,x2,x3])
    
    x1_val = 1*np.array([1,1,1])
    x2_val = 2*np.array([1,1,1])
    x3_val = 2*np.array([1,1,1])
    y_val, x1_grad_val, x2_grad_val , x3_grad_val= Executor([y, grad_x1,grad_x2,grad_x3]).run({x1:x1_val,x2:x2_val,x3:x3_val})
    print(y_val,x1_grad_val,x2_grad_val,x3_grad_val)

def test_matmul_two_vars():
    x1 = Variable('x1')
    x2 = Variable('x2')
    y = matmul_op(x1,x2)
    
    grad_x1,grad_x2 = gradient(y, [x1,x2])
    x1_val  = np.array([[1,2],[3,4]])
    x2_val  = np.array([[5,6],[7,8]])

    y_val, x1_grad_val, x2_grad_val= Executor([y, grad_x1,grad_x2]).run({x1:x1_val,x2:x2_val})
    print(y_val, x1_grad_val, x2_grad_val,sep='\n')

if __name__ == "__main__":
#    test_var_add_var() 
#    test_var_add_const()
    # test_var_mul_var()
    # test_var_mul_const()
    # test_add_mul_mix_1()
    # test_add_mul_mix_2()
    # test_add_mul_mix_3()
    test_matmul_two_vars()
    