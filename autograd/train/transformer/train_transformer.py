import os
import numpy as np
from mtorch.tokenizer import Tokenizer 
from mtorch.tensor import Tensor
from mtorch.nn import Sequential, Linear, Embedding, PositionalEncoding, CausalSelfAttention, MultiHeadAttention, FeedForward, Block, LayerNorm
from mtorch.celoss import CrossEntropyLoss
from mtorch.optimizer import Adam

# --- 1. 数据准备 ---
# (保持不变，省略以节省篇幅，请保留你原来的代码)
file_path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
train_text = text 
#train_text = text[:10000]
tokenizer = Tokenizer(train_text)
full_ids = tokenizer.encode(train_text)
vocab_size = len(tokenizer.stoi)
print(f"Vocab Size: {vocab_size}")

# --- 2. Batch 处理 (保持不变) ---
BLOCK_SIZE = 16
BATCH_SIZE = 64 # 减小一点 Batch Size 方便调试

def one_hot(indices, vocab_size):
    flat_indices = indices.flatten()
    n = len(flat_indices)
    res = np.zeros((n, vocab_size), dtype=np.float32)
    res[np.arange(n), flat_indices] = 1
    return res.reshape(*indices.shape, vocab_size)

def get_batch(ids, batch_size=32, block_size=8):
    max_idx = len(ids) - block_size - 1
    ix = np.random.randint(0, max_idx, (batch_size,))
    x_batch = [ids[i : i + block_size] for i in ix]
    y_batch = [ids[i+1 : i + block_size + 1] for i in ix]
    x_tensor = Tensor(np.array(x_batch), label="Input_Seq")
    y_tensor = Tensor(one_hot(np.array(y_batch), vocab_size), label="Target_Seq")
    return x_tensor, y_tensor

# --- 3. 模型定义 (关键升级！) ---
# 超参数配置
EMB_DIM = 64
N_HEAD = 4   # 新增：我们要用 4 个头
# 只要 EMB_DIM 能被 N_HEAD 整除即可 (32 / 4 = 8)
# block_size = head_dim
HEAD_SIZE = EMB_DIM // N_HEAD 
N_LAYER = 6

print(f"Model Config: Emb={EMB_DIM}, Heads={N_HEAD}, Head_Size={HEAD_SIZE}")

model = Sequential([
    Embedding(vocab_size, EMB_DIM),
    PositionalEncoding(EMB_DIM, max_len=BLOCK_SIZE*2), # output (B, T, C)
    *[Block(EMB_DIM, N_HEAD, BLOCK_SIZE) for _ in range(N_LAYER)],
    LayerNorm(EMB_DIM),
    Linear(EMB_DIM, vocab_size)
])

# --- 4. 终极测试：单批次过拟合 (Overfit Check) ---
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

print(f"\n=== Start MHA Sanity Check (Overfitting One Batch) ===")
# 1. 锁死数据
#fixed_X, fixed_Y = get_batch(full_ids, BATCH_SIZE, BLOCK_SIZE)

for step in range(5000):
    # 2. 永远使用同一个 Batch
    #X, Y = fixed_X, fixed_Y
    X, Y = get_batch(full_ids, BATCH_SIZE, BLOCK_SIZE)
    
    # Forward
    logits = model(X)
    
    # Reshape
    B, T, V = logits.data.shape
    logits_flat = logits.reshape(B*T, V)
    targets_flat = Y.reshape(B*T, V)
    
    # Backward
    loss = criterion(logits_flat, targets_flat)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.data:.4f}")

print("Test Complete.")

# --- 5. 见证奇迹：文本生成 (Inference) ---
print("\n=== Generating Text ===")

def generate(model, start_text, max_new_tokens=20):
    """
    Args:
        model: 训练好的模型
        start_text: 提示词 (Prompt)
        max_new_tokens: 生成多少个新词
    """
    # 1. 编码: String -> List[int]
    ids = tokenizer.encode(start_text)

    print(f"Prompt: '{start_text}'")
    print("Generated: ", end='', flush=True)

    for _ in range(max_new_tokens):
        # 2. 截断上下文 (Critical!)
        # 因为我们的 PositionalEncoding 和 Attention 只能处理 block_size 这么长
        # 如果现在的 ids 超过了 8 个，必须要把前面太旧的切掉，只保留最后 8 个
        cond_ids = ids[-BLOCK_SIZE:]

        # 3. 准备输入 Tensor
        # shape: (1, len)
        x_in = Tensor(np.array([cond_ids]), label="Infer_In")

        # 4. Forward
        # logits shape: (1, len, vocab_size)
        logits = model(x_in)

        # 5. 取最后一个时间步的预测
        # 我们只关心它对"当下"的预测，也就是最后一个 token 的 logits
        # shape: (vocab_size,)
        last_logits = logits.data[0, -1, :]

        # 6. 贪婪采样 (Greedy Search)
        # 直接选概率最大的那个词 (最稳健，适合调试)
        # 以后我们可以加 Softmax + Random Sampling 让它更有创造力
        pred_id = int(np.argmax(last_logits))

        # 7. 解码并打印
        pred_word = tokenizer.decode([pred_id])
        print(pred_word, end='', flush=True)

        # 8. 把生成的词加回 ids，作为下一次的输入
        ids.append(pred_id)

    print("\n-----------------------")

# 测试几个例子
generate(model, "First", max_new_tokens=10)
generate(model, "Before", max_new_tokens=10)
generate(model, "The", max_new_tokens=10)
