import numpy as np
import matplotlib.pyplot as plt

def generate_xor_data(n=200):
    # 生成随机点
    X = np.random.uniform(-5, 5, (n, 2))

    Y = np.zeros((n, 1))
    Y[X[:,0] * X[:,1] > 0] = 1

    # noise
    # X += np.random.normal(0, 0.2, X.shape)

    return X, Y

X, Y = generate_xor_data()
#print(X.shape, Y.shape)
plt.scatter(X[:,0], X[:,1], c=Y.ravel(), cmap='bwr', edgecolors='k')
plt.title("XOR Data: Can you draw ONE line to separate them?")
#plt.show()

'''
结构 ：输入层（两个特征） =》 hidden layer（四个节点） 》 激活函数sigmoid 》 输出层 sigmoid

'''
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#forward
#parmas
W1 = np.random.randn(2,4)
B1 = np.zeros((1, 4))
W2 = np.random.randn(4,1)
B2 = np.zeros((1,1))
ln = 0.0001 # 我不知道设置技巧，先任意设置一个
epochs = 50000
def epoch(i):
    global W1, B1, W2, B2
    # layer1
    Z1 = X @ W1 + B1
    A1 = sigmoid(Z1)
    #layer2
    Z2 = A1 @ W2 + B2
    A2 = sigmoid(Z2)
    
    ''' 求导
    前置知识/经验/技巧 
    # bce + sigmoid  (y' - y)
    # sigmoid y' * (1 - y')
    # A 激活层
    # Z 线性加权层
    Z很特别，是hub，编程时要作为变量
    注意矩阵形状，可以打印下shape，矩阵乘法会用到，后续换位/转置都跟这个相关
    
    
    W求导
    dL/W2 = dL/A2 * dA2/Z2 * dZ2/W2
    dL/W1 = dL/A2 * dA2/Z2 * dZ2/A1 * dA1/dZ1 * dZ1 /W1
    
    自己得出结论：梯度 dL/dW 的形状必须和参数 W 的形状一致
    换位置/装置是帮手
    
    '''
    
    # 代码
    # 求W2 dL/W2 = dL/dZ2 * dZ2/W2
    dZ2  = A2 - Y # y' - y
    dZ2W2 = A1
    '''
    调试形状问题。
    print("\n W2 shape \n", W2.shape)
    print("\n dZ2 shape: \n", dZ2.shape)
    print("\n dZ2W2 shape: \n", dZ2W2.shape)
    print("\n dZ2W2.T shape: \n", dZ2W2.T.shape)
    '''
    dW2 = dZ2W2.T @ dZ2
    
    # 求W1
    # dL/W1 = dL/A2 * dA2/Z2 * dZ2/A1 * dA1/dZ1 * dZ1 /W1
    # dz2已知，先求dz1
    # dZ2/A1 = W2
    # dA1/dZ1 = y' * (1-y') = Z1 * (1 - Z1)  y'对应哪个？y'是预测值，是上个输入Z1? 正确答案：A1
    #print(Z1.shape, dZ2.shape, W2.shape)
    dZ1 = (dZ2 @ W2.T) * (A1 * (1-A1))
    #print(W1.shape, dZ1.shape, X.shape)
    dW1 = X.T @ dZ1
    
    '''
    求b1,b2
    dL/b2 = dL/z2 * dz2/db2(1)
    dL/b1 = dL/z1 * dz1/db1(1)
    '''
    #print(b2.shape,b1.shape)
    #print(dZ2.shape, dZ1.shape)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)
    #print(dB2.shape, dB1.shape)

    #更新参数
    W1 = W1 - ln * dW1
    W2 = W2 - ln * dW2
    B1 = B1 - ln * dB1
    B2 = B2 - ln * dB2

    if i % 500 ==0:
        loss = -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1-A2)) 
        print(loss)

def train(epochs):
    for i in range(epochs):
        epoch(i)

train(epochs)


def plot_decision_boundary(X, y, model_predict):
    # 设定边界范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 预测网格中所有点的分类
    # 模拟一次 Forward
    Z1 = np.c_[xx.ravel(), yy.ravel()] @ W1 + B1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + B2
    A2 = sigmoid(Z2)
    Z = A2.reshape(xx.shape)

    # 绘图
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

# 调用绘图
plot_decision_boundary(X, Y, None)
