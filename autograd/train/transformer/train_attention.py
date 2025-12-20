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

#train_text = text[:10000] # 调试用小数据
train_text = text 
tokenizer = Tokenizer(train_text)
full_ids = tokenizer.encode(train_text)
vocab_size = len(tokenizer.stoi)

print(f"Vocab Size: {vocab_size}")

# --- 2. Batch 处理 (保持不变) ---
BLOCK_SIZE = 32  # 上下文窗口
BATCH_SIZE = 128

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
EMB_DIM = 64
HEAD_SIZE = 64 # 单头注意力，通常设为和 Embedding 一样大

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
optimizer = Adam(model.parameters(), lr=0.001) 

print(f"\nStart Training Attention Model (Dummy Mode)...")

for step in range(1000): # 跑几步测试数据流
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
# 注意：这必须是训练集中出现过的词或短语，效果才最好
# 如果你训练的是莎士比亚，试试这些：
generate(model, "First", max_new_tokens=10)
generate(model, "Before", max_new_tokens=10)
generate(model, "The", max_new_tokens=10)
# 如果你真的想测 "The cat sat on the"，你需要确保这句话在 input.txt 里
# 否则它可能会根据莎士比亚风格生成 "The king..." 或者 "The lord..."
