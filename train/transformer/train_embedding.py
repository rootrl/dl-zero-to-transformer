import os
import numpy as np
from mtorch.tokenizer import Tokenizer 
from mtorch.tensor import Tensor
from mtorch.nn import Sequential, Linear, Embedding
from mtorch.celoss import CrossEntropyLoss
from mtorch.optimizer import Adam

# --- 1. 数据加载与 Tokenizer ---
# 确保你有 data/input.txt，如果没有，请先运行之前的下载脚本
file_path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please download tiny shakespeare first.")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 为了演示速度，只取前 10000 个字符训练
# (你可以随时注释掉这行跑全量，但 CPU 会比较慢)
# train_text = text[:10000] 
train_text = text 

print("Initializing Tokenizer...")
tokenizer = Tokenizer(train_text)
vocab_size = len(tokenizer.stoi)
print(f"Vocab Size: {vocab_size}")

# 把所有文本转成 ID 列表
full_ids = tokenizer.encode(train_text)
data_len = len(full_ids)
print(f"Total Tokens in training set: {data_len}")

# --- 2. 辅助函数 ---

def one_hot(indices, vocab_size):
    """
    将整数索引转为 One-Hot 矩阵。
    Input: (Batch,) -> [1, 5, 2]
    Output: (Batch, Vocab_Size)
    """
    batch_size = len(indices)
    res = np.zeros((batch_size, vocab_size), dtype=np.float32)
    res[np.arange(batch_size), indices] = 1
    return res

def get_batch(ids, batch_size=32):
    """
    随机采样 Batch。
    X: 当前词
    Y: 下一个词
    """
    # 随机选择 batch_size 个起始点
    # 注意：不能选到最后一个，因为最后一个没有 next
    ix = np.random.randint(0, len(ids) - 1, (batch_size,))
    
    x_batch_ids = [ids[i] for i in ix]
    y_batch_ids = [ids[i+1] for i in ix]
    
    # X 不需要 One-Hot，因为 Embedding 层吃整数
    # Y 需要 One-Hot，因为 CrossEntropy 吃 One-Hot
    x_tensor = Tensor(x_batch_ids, label="Input_Ids")
    y_onehot = one_hot(y_batch_ids, vocab_size)
    y_tensor = Tensor(y_onehot, label="Target_OneHot")
    
    return x_tensor, y_tensor

# --- 3. 模型定义 ---
emb_dim = 64 # 嵌入维度：把每个词压缩成 N 个浮点数

# 架构：Embedding -> Linear -> Logits
# 没有激活函数，这是一个纯线性的 Embedding 学习模型
model = Sequential([
    Embedding(vocab_size, emb_dim),
    Linear(emb_dim, vocab_size) # 映射回词表大小，预测概率
])

# --- 4. 训练配置 ---
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01) # 学习率稍微大一点

print("\nStart Training Embedding Model...")

# 训练参数
max_steps = 5000 # batch
batch_size = 512

for step in range(max_steps):
    # 1. Get Batch
    X_batch, Y_batch = get_batch(full_ids, batch_size)
    
    # 2. Forward
    logits = model(X_batch)
    
    # 3. Loss
    loss = criterion(logits, Y_batch)
    
    # 4. Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.data:.4f}")

print("Training Complete.")

# --- 5. 验证 (Inference) ---
print("\n=== Inference Test ===")

def predict_next(word):
    if word not in tokenizer.stoi:
        return f"<UNK> ({word} not in vocab)"
    
    # 1. Encode
    word_id = tokenizer.encode(word) # return list like [3]
    x_in = Tensor(word_id)
    
    # 2. Forward
    logits = model(x_in) # shape (1, vocab_size)
    
    # 3. Get Prediction
    # 简单起见，直接取概率最大的 (Greedy Search)
    # 实际 GPT 会用概率采样
    probs = logits.data
    pred_id = np.argmax(probs)
    
    # 4. Decode
    pred_word = tokenizer.decode([pred_id])
    return pred_word

# 测试几个常用词
test_words = ["First", "Before", "speak", "the", "of"]
for w in test_words:
    next_w = predict_next(w)
    print(f"'{w}' -> '{next_w}'")

# --- 6. 向量几何分析 (Cosine Similarity) ---
print("\n=== Geometric Analysis: Nearest Neighbors ===")

def get_nearest_neighbors(word, k=5):
    if word not in tokenizer.stoi:
        print(f"'{word}' not in vocab.")
        return

    # 1. 拿到查询词的向量 (Query Vector)
    word_id = tokenizer.stoi[word]
    # shape: (1, emb_dim)
    q_vec = model.layers[0].w.data[word_id] 

    # 2. 拿到所有词的向量 (All Vectors)
    # shape: (vocab_size, emb_dim)
    all_vecs = model.layers[0].w.data 

    # 3. 计算余弦相似度 (Cosine Similarity)
    # sim = (A . B) / (|A| * |B|)
    
    # 计算模长 (Norm)
    q_norm = np.linalg.norm(q_vec)
    all_norms = np.linalg.norm(all_vecs, axis=1)
    
    # 点积 (Dot Product)
    dot_products = np.dot(all_vecs, q_vec)
    
    # 相似度
    similarities = dot_products / (all_norms * q_norm + 1e-8)
    
    # 4. 排序 (argsort 返回的是从小到大的索引，所以要取反)
    sorted_ids = np.argsort(-similarities)
    
    print(f"\nNearest neighbors to '{word}':")
    for i in range(1, k+1): # 跳过第0个，因为第0个是自己 (sim=1.0)
        neighbor_id = sorted_ids[i]
        neighbor_word = tokenizer.itos[neighbor_id]
        sim_score = similarities[neighbor_id]
        print(f"  {neighbor_word}: {sim_score:.4f}")

# 试试看这几个词
get_nearest_neighbors("King")
get_nearest_neighbors("the")
get_nearest_neighbors("speak")
