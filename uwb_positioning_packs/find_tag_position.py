import numpy as np
#定義圓球方程式
def f(x,center,r):
    return np.array([ (x[0]-center[0][0])**2 + (x[1]-center[0][1])**2 + (x[2]-center[0][2])**2 - r[0]**2   ,
             (x[0]-center[1][0])**2 + (x[1]-center[1][1])**2 + (x[2]-center[1][2])**2 - r[1]**2   ,
             (x[0]-center[2][0])**2 + (x[1]-center[2][1])**2 + (x[2]-center[2][2])**2 - r[2]**2   ])
#定義方程式的梯度
def grad_f(x,center):
    return np.array([[ 2*(x[0]-center[0][0]) , 2*(x[1]-center[0][1]) , 2*(x[2]-center[0][2]) ],
                     [ 2*(x[0]-center[1][0]) , 2*(x[1]-center[1][1]) , 2*(x[2]-center[1][2]) ],
                     [ 2*(x[0]-center[2][0]) , 2*(x[1]-center[2][1]) , 2*(x[2]-center[2][2]) ]])

#梯度下降找近似解
def gradient_descent(X,center,r):
    esp=1e-2
    N=1000
    for i in range(N):
        # print("iter : {:d}".format(i))
        fx=f(X,center,r)
        grad=grad_f(X,center)
        dx=np.dot(grad.T,fx)
        X=X-(1e-6)*dx
        if  np.sqrt(np.sum(dx**2)) < esp:
            break
    
    print("iter {:d} done dx={:}".format(i,dx))
    # return [int(X[0]),int(X[1]),int(X[2])]
    #
    if X[0]==np.nan:
        return False
    else:
        # 3D
        return [int(X[0]),int(X[1]),int(X[2])]
