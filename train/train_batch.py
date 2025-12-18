import numpy as np
from mtorch.celoss import CrossEntropyLoss
from mtorch.tensor import Tensor
from mtorch.nn import Linear, Sequential, Module, ReLU
from mtorch.optimizer import Adam

# 1. 模拟更多的数据 (比如 100 个样本)
# 假设我们有 100 条数据，特征维度 10，分类 2
data_size = 100
batch_size = 10

X_raw = np.random.randn(data_size, 10)
# 随机生成一些 0/1 标签 (One-Hot)
labels = np.random.randint(0, 2, size=(data_size,))
Y_raw = np.zeros((data_size, 2))
Y_raw[np.arange(data_size), labels] = 1

# 2. 定义模型
model = Sequential([
    Linear(10, 16),
    ReLU(),
    Linear(16, 2)
])

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

print(f"Start Training with Batch Size = {batch_size}...")

# 3. 训练循环 Mini-batch 
for epoch in range(500): # 跑 5 个 Epoch
    
    # 手动 Shuffle 数据 
    indices = np.random.permutation(data_size)
    X_shuffled = X_raw[indices]
    Y_shuffled = Y_raw[indices]
    
    total_loss = 0
    
    # --- Mini-batch Loop ---
    for i in range(0, data_size, batch_size):
        # 1. 切片 (Slicing) - 制作 Batch
        # numpy 切片出来还是 numpy，需要包成 Tensor
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
