from mtorch.tensor import Tensor
import numpy as np
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

# 线性层
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # 1. 初始化权重 W (in, out)
        # k = 1/sqrt(in)
        k = np.sqrt(1.0 / in_features)
        
        # W 随机初始化
        w_data = np.random.uniform(-k, k, (in_features, out_features))
        self.w = Tensor(w_data, label="W")
        
        # b 初始化为 0
        if bias:
            self.b = Tensor(np.zeros((1, out_features)), label="b")
        else:
            self.b = None

    def __call__(self, x):
        # 允许像函数一样调用 layer(x)
        return self.forward(x)

    def forward(self, x):
        # Y = X @ W + b
        out = x @ self.w
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self):
        return [self.w, self.b] if self.b is not None else [self.w]

class Embedding(Module):
    def __init__(self, vocab_size, emb_dim):
        #初始化w
        self.w = Tensor(np.random.standard_normal([vocab_size, emb_dim]), label="Embed_W")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        assert isinstance(x, Tensor), "x must be type: Tensor"
        idx = x.data.astype(int)
        out = self.w.data[idx]
        out = Tensor(out, _children=(self.w,), _op='lookup')
        def _backward():
            np.add.at(self.w.grad, idx, out.grad) 
        out._backward = _backward
        return out

    def parameters(self):
        return [self.w]

class PositionalEncoding(Module):
    def __init__(self, emb_dim, max_len=5000):
        """
        emb_dim: d_model, 也就是 Embedding 的维度 (比如 64, 512)
        max_len: 预计算的最大长度 (足够覆盖训练中最长的句子即可)
        """
        self.emb_dim = emb_dim

        # 1. 初始化一个矩阵 (Max_Len, Emb_Dim) 全是 0
        pe = np.zeros((max_len, emb_dim))

        # 2. 生成位置索引 (列向量)
        # shape: (Max_Len, 1) -> [[0], [1], [2], ...]
        position = np.arange(0, max_len).reshape(-1, 1)

        # 3. 计算分母 (频率项 div_term)
        # 公式是 1 / 10000^(2i/d)。
        # 为了数值稳定性，我们在对数空间计算：exp( -2i * log(10000) / d )
        # 我们只需要计算一半的维度 (0, 2, 4...)，因为 sin 和 cos 共享频率
        div_term = np.exp(np.arange(0, emb_dim, 2) * -(np.log(10000.0) / emb_dim))

        # 4. 填充矩阵
        # 偶数列 (0, 2, 4...) 用 sin
        # position * div_term 利用了广播机制: (Max_Len, 1) * (Emb_Dim/2,) -> (Max_Len, Emb_Dim/2)
        pe[:, 0::2] = np.sin(position * div_term)

        # 奇数列 (1, 3, 5...) 用 cos
        pe[:, 1::2] = np.cos(position * div_term)

        # 5. 保存为 Tensor
        # 关键点：requires_grad=False (或者这里只是不放入 parameters())
        # 因为这是固定的数学公式，不需要反向传播更新它
        self.pe = Tensor(pe, label="Sinusoidal_PE")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        x: (Batch_Size, Seq_Len, Emb_Dim)
        """
        # 1. 获取当前输入的实际长度 T
        # x.data 的 shape 是 (B, T, C)
        seq_len = x.data.shape[1]

        # 2. 切片 (Slicing)
        # pe_slice shape: (T, C)
        pe_slice_data = self.pe.data[:seq_len, :]

        # 3. 封装成 Tensor
        pe_slice = Tensor(pe_slice_data, label="PE_Slice")

        # 4. 加法 (Addition) & 广播 (Broadcasting)
        # (B, T, C) + (T, C)
        return x + pe_slice

    def parameters(self):
        return [] # 没有可学习参数

# 激活函数
class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # 递归收集所有子层的参数
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()

class ReLU(Module):
    def __call__(self, x):
        return x.relu()

