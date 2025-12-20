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
B1  = np.zeros((1,128))# 1 * 128
W2 = np.random.standard_normal((128, 10)) * np.sqrt(1 / 128) # 128 * 10
B2 = np.zeros((1, 10)) # 1 * 10
ln = 0.01
epoches = 50
batch_size = 128  # 每次只看 128 张图
steps = X_train.shape[0] // batch_size # 一轮有多少步

def epoch(i):
    global W1,B1,W2,B2

    # 洗牌（可选，但推荐）：打乱数据顺序
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]

    for step in range(steps):
        # 1. 切片：取出一小批数据
        start = step * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]

        #forward
        #Hidden layer
        Z1 = X_batch@W1 + B1 # N*128
        A1 = sigmoid(Z1) # N * 128

        # 输出层
        Z2 = A1 @ W2 + B2 # N * 10
        A2 = softmax(Z2) # N * 10 

        # 求导
        # L = ?
        # softmax = ?
        # dL/Z2 = dL/A2 * dA2/Z2
        #backward
        dZ2 = A2 - Y_batch
        dW2 = A1.T@dZ2
        #print(Z1.shape, dZ2.shape, W2.shape)
        dZ1 = (dZ2 @ W2.T) * (A1 * (1 - A1))
        #print(W1.shape, dZ1.shape, X_train.shape)
        dW1 = X_batch.T@dZ1
        dB2 = np.sum(dZ2, axis=0, keepdims=True)
        dB1 = np.sum(dZ1, axis=0, keepdims=True)

        # 更新
        W2 = W2 - ln * dW2
        W1 = W1 - ln * dW1
        
        B2 = B2 - ln * dB2
        B1 = B1 - ln * dB1

    loss = np.mean(np.abs(dZ2))

    # 计算准确率 (Accuracy) - 这比 Loss 更直观！
    predictions = np.argmax(A2, axis=1) # 选概率最大的那个类
    labels = np.argmax(Y_batch, axis=1)
    accuracy = np.mean(predictions == labels)

    print(f"Epoch {i}, Loss: {loss:.4f}, Acc: {accuracy:.2%}")

def evaluate_test():
    
    # 1. Forward
    Z1 = X_test @ W1 + B1
    A1 = sigmoid(Z1)
    
    Z2 = A1 @ W2 + B2
    A2 = softmax(Z2)
    
    # 2. 计算准确率
    predictions = np.argmax(A2, axis=1)
    labels = np.argmax(Y_test, axis=1)
    
    accuracy = np.mean(predictions == labels)
    print(f"\n======== 最终测试集成绩 ========")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"==============================")


def train():
    for i in range(epoches):
        epoch(i)

train()
evaluate_test()
