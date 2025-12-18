from train import data
import numpy as np
from mtorch.celoss import CrossEntropyLoss
from mtorch.nn import Linear, Sequential, Module, ReLU
from mtorch.tensor import Tensor
from mtorch.optimizer import Adam

# 执行加载
(X_train, Y_train), (X_test, Y_test) = data.load_mnist()

print("\nData Loaded Success!")
print(f"X_train shape: {X_train.shape}") # 期待 (60000, 784)
print(f"Y_train shape: {Y_train.shape}") # 期待 (60000, 1)

data_size = len(X_train)
batch_size = 128
epoches = 50
steps = X_train.shape[0]

#one_hot
def one_hot(Y, n = 10):
    return np.eye(n)[Y.flatten()]

Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)
print(Y_train.shape)

model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
])


criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(epoches):
    indices = np.random.permutation(data_size)
    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]
    
    total_loss = 0
    
    # --- Mini-batch Loop ---
    for i in range(0, data_size, batch_size):
        x_batch_np = X_shuffled[i : i + batch_size]
        y_batch_np = Y_shuffled[i : i + batch_size]
        
        X_batch = Tensor(x_batch_np, label="Batch_X")
        Y_batch = Tensor(y_batch_np, label="Batch_Y")
        
        # 2. Forward
        logits = model(X_batch)
        
        # 3. Loss
        loss = criterion(logits, Y_batch)
        total_loss += loss.data
        
        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Avg Loss: {total_loss / (data_size / batch_size):.4f}") 

# 1. Forward
test_input = Tensor(X_test, label="Test_X")
A2 = model(test_input)

# 2. 计算准确率
predictions = np.argmax(A2.data, axis=1)
labels = np.argmax(Y_test, axis=1)

accuracy = np.mean(predictions == labels)
print(f"\n======== 最终测试集成绩 ========")
print(f"Test Accuracy: {accuracy:.2%}")
print(f"==============================")

