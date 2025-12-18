import numpy as np
from mtorch.celoss import CrossEntropyLoss
from mtorch.nn import Linear, Sequential, Module, ReLU
from mtorch.tensor import Tensor
from mtorch.optimizer import Adam

# --- 2. 数据准备 (One-Hot 版本的 XOR) ---
# Input: (4, 2)
X = Tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], label="Input")

# Target: One-Hot Encoding (4, 2)
# [1, 0] 代表 类别0 (False)
# [0, 1] 代表 类别1 (True)
Y = Tensor([
    [1., 0.],  # 0 xor 0 = 0
    [0., 1.],  # 0 xor 1 = 1
    [0., 1.],  # 1 xor 0 = 1
    [1., 0.]   # 1 xor 1 = 0
], label="Target_OneHot")

# --- 3. 定义模型 ---
# 结构：Input(2) -> Hidden(8) -> ReLU -> Output(2)
# 注意：输出层是 2！因为我们要输出两个类别的 Logits
model = Sequential([
    Linear(2, 8),
    ReLU(),        # 使用 ReLU 替代 Sigmoid，收敛更快
    Linear(8, 2)   # 输出 2 个值：[Score_for_0, Score_for_1]
])

# --- 4. 训练配置 ---
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

print("Start Training XOR with CrossEntropy...")

for i in range(3000):
    # A. Forward
    # 注意：这里输出的是 logits (未经过 Softmax 的原始分数)
    logits = model(X)
    
    # B. Loss
    # CrossEntropy 内部会自动做 Softmax + Log + NLL
    loss = criterion(logits, Y)
    
    # C. Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Epoch {i}, Loss: {loss.data:.4f}")

# --- 5. 验证结果 ---
print("\nTraining Complete.")
final_logits = model(X)
print("Final Logits:\n", final_logits.data)

# 手动把 Logits 转成概率看一眼 (Softmax)
exp_logits = np.exp(final_logits.data)
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

print("\nPredicted Probabilities (Softmax):")
print(probs)

# 打印最终分类结果
predictions = np.argmax(probs, axis=1)
print("\nPredicted Class:", predictions)
print("Target Class:    [0 1 1 0]")
