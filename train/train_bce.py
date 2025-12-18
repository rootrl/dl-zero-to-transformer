import numpy as np
from mtorch.celoss import BCEWithLogitsLoss
from mtorch.nn import Linear, Sequential, Module, ReLU
from mtorch.tensor import Tensor
from mtorch.optimizer import Adam

# --- 1. 数据准备 (XOR, 标量标签版) ---
# Input: (4, 2)
X = Tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], label="Input")

# Target: Scalar (4, 1)
# 0/1 直接表示类别
Y = Tensor([
    [0.],  # 0 xor 0 = 0
    [1.],  # 0 xor 1 = 1
    [1.],  # 1 xor 0 = 1
    [0.]   # 1 xor 1 = 0
], label="Target_Scalar")

# --- 2. 定义模型 ---
# 结构：Input(2) -> Hidden(8) -> ReLU -> Output(1) !!!
# 注意：输出层只有 1 个神经元，因为 BCE 是对这一个值的 Yes/No 判断
model = Sequential([
    Linear(2, 8),
    ReLU(),
    Linear(8, 1)   # Output shape: (N, 1)
])

# --- 3. 训练配置 ---
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.01)

print("Start Training XOR with BCEWithLogits...")

for i in range(500):
    # A. Forward
    logits = model(X) # (4, 1)
    
    # B. Loss
    loss = criterion(logits, Y)
    
    # C. Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Epoch {i}, Loss: {loss.data:.4f}")

# --- 4. 验证结果 ---
print("\nTraining Complete.")
final_logits = model(X)
print("Final Logits:\n", final_logits.data)

# Sigmoid 转换为概率
probs = 1 / (1 + np.exp(-final_logits.data))
print("\nPredicted Probabilities:")
print(probs)

# 阈值判断 ( > 0.5 为 1)
predictions = np.where(probs > 0.5, 1.0, 0.0)
print("\nPredicted Class:\n", predictions)
print("Target Class:    \n[[0.]\n [1.]\n [1.]\n [0.]]")
