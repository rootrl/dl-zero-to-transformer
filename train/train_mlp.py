import numpy as np
from mtorch.nn import Tensor, Linear, Sequential, Module, Sigmoid, ReLU
from mtorch.tensor import Tensor
from mtorch.optimizer import Adam, SGD
# 1. 数据准备
X = Tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], label="Input")
Y = Tensor([[0.], [1.], [1.], [0.]], label="Target")

# 重新定义模型
model = Sequential([
    Linear(2, 8),
    Sigmoid(),
    Linear(8, 1),
    Sigmoid()
])

# 3. 定义优化器
# 自动收集所有参数 (W1, b1, W2, b2)
optimizer = Adam(model.parameters())

print("Start Training...")

for i in range(2000):
    #__call__ forward
    y_pred = model(X)
    
    # Loss
    diff = y_pred - Y
    loss = (diff ** 2).sum()
    
    # --- 2. Backward ---
    optimizer.zero_grad()  # 一键清空
    loss.backward()        # 一键求导
    optimizer.step()       # 一键更新
    
    if i % 20 == 0:
        print(f"Epoch {i}, Loss: {loss.data:.4f}")

print("\nFinal Predictions:")
print(y_pred.data)
