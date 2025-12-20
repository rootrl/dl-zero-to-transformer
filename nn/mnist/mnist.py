import data
import numpy as np
# 执行加载
(X_train, Y_train), (X_test, Y_test) = data.load_mnist()

print("\nData Loaded Success!")
print(f"X_train shape: {X_train.shape}") # 期待 (60000, 784)
print(f"Y_train shape: {Y_train.shape}") # 期待 (60000, 1)

#one_hot
def one_hot(Y, n = 10):
    return np.eye(n)[Y.flatten()]

Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)
print(Y_train.shape)

#sigmoid
def sigmoid(Z):
    return 1/(1+np.exp(-Z))


#softmax
def softmax(Z):
    shift_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shift_Z)
    return exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)

#init
W1 = np.random.standard_normal((784, 128)) * np.sqrt(1 / 784) # 784 * 128
B1 = np.zeros((1,128))# 1 * 128
W2 = np.random.standard_normal((128, 10)) * np.sqrt(1 / 128) # 128 * 10
B2 = np.zeros((1, 10)) # 1 * 10
ln = 0.0001
epoches = 5000

def epoch(i):
    global W1,B1,W2,B2
    #forward
    #Hidden layer
    Z1 = X_train@W1 + B1 # N*128
    A1 = sigmoid(Z1) # N * 128

    # 输出层
    Z2 = A1 @ W2 + B2 # N * 10
    A2 = softmax(Z2) # N * 10 


'''
    warning: @ shape
    默认*,跟W有关：@ 本质：线性层权重
    dZ2 = dL/A2 * dA2/Z2 = A2 - Y_train
    dW2 = dZ2 @ A1

    dZ1 = (dZ2 @ W2) * ((A1) * (1-A1))
    dW1 = dZ1 @ X_train
    
'''
    # 求导
    # L = ?
    # softmax = ?
    # dL/Z2 = dL/A2 * dA2/Z2
    #backward
    dZ2 = A2 - Y_train
    dW2 = A1.T@dZ2
    #print(Z1.shape, dZ2.shape, W2.shape)
    dZ1 = (dZ2 @ W2.T) * (A1 * (1 - A1))
    #print(W1.shape, dZ1.shape, X_train.shape)
    dW1 = X_train.T@dZ1
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    # 更新
    W2 = W2 - ln * dW2
    W1 = W1 - ln * dW1
    
    B2 = B2 - ln * dB2
    B1 = B1 - ln * dB1

    if i % 8 == 0:
        loss = np.mean(np.abs(dZ2))
        print(f"Epoch {i}, Error: {loss:.6f}")
    
for i in range(epoches):
    epoch(i)
