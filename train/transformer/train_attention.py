import os
import numpy as np
from mtorch.tokenizer import Tokenizer 
from mtorch.tensor import Tensor
from mtorch.nn import Sequential, Linear, Embedding, PositionalEncoding, CausalSelfAttention
from mtorch.celoss import CrossEntropyLoss
from mtorch.optimizer import Adam

# --- 1. 数据准备 (保持不变) ---
file_path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
if not os.path.exists(file_path):
    print("Error: data/input.txt not found.")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

train_text = text[:10000] # 调试用小数据
tokenizer = Tokenizer(train_text)
full_ids = tokenizer.encode(train_text)
vocab_size = len(tokenizer.stoi)

print(f"Vocab Size: {vocab_size}")

# --- 2. Batch 处理 (保持不变) ---
BLOCK_SIZE = 8  # 上下文窗口
BATCH_SIZE = 32

def one_hot(indices, vocab_size):
    flat_indices = indices.flatten()
    n = len(flat_indices)
    res = np.zeros((n, vocab_size), dtype=np.float32)
    res[np.arange(n), flat_indices] = 1
    original_shape = indices.shape
    return res.reshape(*original_shape, vocab_size)

def get_batch(ids, batch_size=32, block_size=8):
    max_idx = len(ids) - block_size - 1
    ix = np.random.randint(0, max_idx, (batch_size,))
    x_batch = [ids[i : i + block_size] for i in ix]
    y_batch = [ids[i+1 : i + block_size + 1] for i in ix]
    
    x_np = np.array(x_batch)
    y_np = np.array(y_batch)
    
    x_tensor = Tensor(x_np, label="Input_Seq")
    y_onehot_np = one_hot(y_np, vocab_size)
    y_tensor = Tensor(y_onehot_np, label="Target_Seq")
    return x_tensor, y_tensor

# --- 3. 模型定义 (架构升级！) ---
EMB_DIM = 32
HEAD_SIZE = 32 # 单头注意力，通常设为和 Embedding 一样大

# 架构：Emb -> PE -> SelfAttention -> Linear
model = Sequential([
    # 1. 把词 ID 变成向量 (B, T, C)
    Embedding(vocab_size, EMB_DIM),
    
    # 2. 注入位置信息 (B, T, C)
    PositionalEncoding(EMB_DIM, max_len=BLOCK_SIZE*2),
    
    # 3. 自注意力层 (B, T, C) -> (B, T, C)
    # 这一层负责"回头看"，聚合上下文信息
    CausalSelfAttention(emb_dim=EMB_DIM, head_size=HEAD_SIZE, block_size=BLOCK_SIZE),
    
    # 4. 输出层 (B, T, C) -> (B, T, V)
    Linear(HEAD_SIZE, vocab_size)
])

# --- 4. 训练循环 ---
criterion = CrossEntropyLoss()
# 注意：Attention 层参数较多，学习率通常要调低一点，但在手搓阶段 0.005 或 0.001 都可以
optimizer = Adam(model.parameters(), lr=0.005) 

print(f"\nStart Training Attention Model (Dummy Mode)...")

for step in range(500): # 跑几步测试数据流
    # 1. Get Batch
    X, Y = get_batch(full_ids, BATCH_SIZE, BLOCK_SIZE)
    
    # 2. Forward
    logits = model(X) 
    
    # 3. Reshape (利用你新写的 Tensor.reshape)
    B, T, V = logits.data.shape
    logits_flat = logits.reshape(B*T, V)
    targets_flat = Y.reshape(B*T, V)
    
    # 4. Backward
    loss = criterion(logits_flat, targets_flat)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.data:.4f}")

print("Pipeline Check Complete.")
