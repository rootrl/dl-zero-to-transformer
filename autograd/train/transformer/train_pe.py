import os
import numpy as np
from mtorch.tokenizer import Tokenizer 
from mtorch.tensor import Tensor
# 假设 PositionalEncoding 已经在 nn 里了 (下一步实现)
from mtorch.nn import Sequential, Linear, Embedding, PositionalEncoding 
from mtorch.celoss import CrossEntropyLoss
from mtorch.optimizer import Adam

# --- 1. 数据准备 ---
file_path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
if not os.path.exists(file_path):
    print("Error: data/input.txt not found.")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 截取一部分训练，加快速度
train_text = text 
tokenizer = Tokenizer(train_text)
full_ids = tokenizer.encode(train_text)
vocab_size = len(tokenizer.stoi)

print(f"Vocab Size: {vocab_size}")
print(f"Total Tokens: {len(full_ids)}")

# --- 2. 关键升级：处理序列数据 ---
BLOCK_SIZE = 8 # 上下文窗口：一次看8个词
BATCH_SIZE = 128

def one_hot(indices, vocab_size):
    """
    Input: (Batch, Seq) -> (32, 8)
    Output: (Batch, Seq, Vocab) -> (32, 8, 65)
    """
    # 展平处理，然后再 reshape 回去
    flat_indices = indices.flatten()
    n = len(flat_indices)
    res = np.zeros((n, vocab_size), dtype=np.float32)
    res[np.arange(n), flat_indices] = 1
    
    # Reshape 回 (Batch, Seq, Vocab)
    original_shape = indices.shape
    return res.reshape(*original_shape, vocab_size)


def get_batch(ids, batch_size=32, block_size=8):
    """
    X: (Batch, Block_Size) -> 输入序列
    Y: (Batch, Block_Size) -> 目标序列 (每个位置预测下一个)
    """
    # 随机选 batch_size 个起始点
    max_idx = len(ids) - block_size - 1
    ix = np.random.randint(0, max_idx, (batch_size,))
    
    x_batch = []
    y_batch = []
    
    for i in ix:
        # 取一段: ids[i : i+block_size]
        x_batch.append(ids[i : i + block_size])
        # 预测下一段: ids[i+1 : i+block_size+1]
        y_batch.append(ids[i+1 : i + block_size + 1])
    
    # 转 numpy
    x_np = np.array(x_batch) # (32, 8)
    y_np = np.array(y_batch) # (32, 8)
    
    x_tensor = Tensor(x_np, label="Input_Seq")
    
    # 目标转 One-Hot
    y_onehot_np = one_hot(y_np, vocab_size)
    y_tensor = Tensor(y_onehot_np, label="Target_Seq")
    
    return x_tensor, y_tensor

fixed_X, fixed_Y = get_batch(full_ids, BATCH_SIZE, BLOCK_SIZE)

# --- 3. 模型定义 ---
EMB_DIM = 32

# 这里的 Sequential 逻辑是：
# 1. Embedding: (B, T) -> (B, T, C)
# 2. PositionalEncoding: (B, T, C) -> (B, T, C) [注入位置信息]
# 3. Linear: (B, T, C) -> (B, T, V) [对每个位置独立做投影]
model = Sequential([
    Embedding(vocab_size, EMB_DIM),
    PositionalEncoding(EMB_DIM, max_len=500), # 预计算足够长
    Linear(EMB_DIM, vocab_size)
])

# --- 4. 训练配置 ---
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

print(f"\nStart Training with Block_Size={BLOCK_SIZE}...")

for step in range(1000):
    # 1. Get Batch (Sequence)
    #X, Y = fixed_X, fixed_Y # <--- 用锁死的数据 测试loss
    X, Y = get_batch(full_ids, BATCH_SIZE, BLOCK_SIZE)
    
    # 2. Forward
    # Logits Shape: (B, T, V)
    logits = model(X) 
    
    # 3. Reshape for Loss (关键步骤)
    # 我们的 CrossEntropyLoss 期望 (N, C)
    # 所以要把 Batch 和 Time 维度合并 -> (B*T, V)
    B, T, V = logits.data.shape

    logits_flat = logits.reshape(-1, V)
    targets_flat = Y.reshape(-1, V)
    
    # 4. Backward
    loss = criterion(logits_flat, targets_flat)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.data:.4f}")

print("Training Complete.")

# --- 验证：位置编码是否生效？ ---
# 我们可以打印一下 PE 层的输出，看看是否加上了非零值
print("\n--- Debug: Check PE Values ---")
pe_layer = model.layers[1] # 获取 PE 层
print("PE Matrix Sample (First 5 positions, First 5 dims):")
# 打印内部的 pe 矩阵 (只读数据)
print(pe_layer.pe.data[:5, :5])
