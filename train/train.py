import numpy as np
from mtorch.tensor import Tensor  # 假设你的类保存在 microtorch.py

# 1. 准备数据 (XOR 问题)
# 4个样本，2个特征
X = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], label="Input")

# 期望输出 (0, 1, 1, 0) -> 形状 (4, 1)
Y = Tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], label="Target")

# 2. 定义模型权重
# 隐层: 2 -> 4
W1 = Tensor(np.random.randn(2, 4), label="W1")
b1 = Tensor(np.zeros((1, 4)), label="b1")

# 输出层: 4 -> 1
W2 = Tensor(np.random.randn(4, 1), label="W2")
b2 = Tensor(np.zeros((1, 1)), label="b2")

# 3. 训练循环
print("Start Training...")
for i in range(1000):
    
    # --- Forward (架构设计) ---
    # 就像写 PyTorch 一样，只写正向逻辑
    
    # Layer 1
    z1 = X @ W1 + b1
    a1 = z1.sigmoid() 
    
    # Layer 2
    z2 = a1 @ W2 + b2
    a2 = z2.sigmoid()
    
    # Loss (MSE)
    diff = a2 - Y
    loss = (diff ** 2).sum()
    
    # --- Backward (自动微分) ---
    # 魔法发生的地方：一键求导
    
    # 记得先清空梯度！(模拟 optimizer.zero_grad)
    W1.zero_grad(); b1.zero_grad()
    W2.zero_grad(); b2.zero_grad()
    
    loss.backward()
    
    # --- Update (SGD) ---
    # 简单的梯度下降
    lr = 1.0
    W1.data -= lr * W1.grad
    b1.data -= lr * b1.grad
    W2.data -= lr * W2.grad
    b2.data -= lr * b2.grad
    
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.data:.4f}")

print("\nFinal Predictions:")
print(a2.data)
