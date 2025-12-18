import numpy as np
from mtorch.tensor import Tensor

# Loss 
class CrossEntropyLoss:
    def __call__(self, logits, targets):
        """
        Args:
            logits: Tensor (N, C) - 未经过 Softmax 的原始分数
            targets: Tensor (N, C) - One-Hot 编码的目标概率 (或者只是索引)
                     这里我们假设是 One-Hot (或者概率分布)，形状与 logits 一致
        """
        # 1. 基础数据准备
        # 为了数值稳定，我们需要减去最大值 (Max Trick)
        # e^x / sum(e^x) = e^(x-c) / sum(e^(x-c))

        # 注意：这里我们为了简单，直接在 numpy 层操作数据计算 Loss
        # 然后手动构建一个 Tensor 作为结果，并手动定义它的 _backward
        # 这种 "手动 fused kernel" 的写法在底层库中很常见

        N = logits.data.shape[0]

        # --- Numerical Stability Trick ---
        # shift_logits = logits - max(logits)
        shift_logits = logits.data - np.max(logits.data, axis=1, keepdims=True)
        Z = np.sum(np.exp(shift_logits), axis=1, keepdims=True)
        log_probs = shift_logits - np.log(Z)
        probs = np.exp(log_probs)

        # 2. 计算 Forward Loss (NLL)
        # loss = -sum(y * log(p)) / N
        # 加上 1e-9 防止 log(0)
        loss_val = -np.sum(targets.data * log_probs) / N

        out = Tensor(loss_val, _children=(logits, targets), _op='CrossEntropy')

        # 3. 定义 Backward (The Magic)
        def _backward():
            # 梯度公式：dL/dZ = (P - Y) / N
            grad = (probs - targets.data) / N

            # 链式法则：out.grad 通常是 1
            grad = grad * out.grad

            # 回传给 logits
            if logits.grad is None:
                logits.grad = grad
            else:
                logits.grad += grad

            # targets 通常不需要梯度，略过

        out._backward = _backward
        return out

class BCEWithLogitsLoss:
    def __call__(self, logits, targets):
        """
        Args:
            logits: (N, 1) - 未经过 Sigmoid 的原始分数
            targets: (N, 1) - 0 或 1 的标签
        """
        # 1. Forward (使用数值稳定公式)
        # L = max(x, 0) - x*y + log(1 + exp(-abs(x)))
        x = logits.data
        y = targets.data

        # stable computation using numpy
        loss_per_sample = np.maximum(x, 0) - x * y + np.log(1 + np.exp(-np.abs(x)))
        loss_val = np.mean(loss_per_sample)

        out = Tensor(loss_val, _children=(logits, targets), _op='BCEWithLogits')

        # 2. Backward (可以直接利用 Sigmoid 的性质)
        def _backward():
            # 计算 Sigmoid 概率: 1 / (1 + e^-x)
            probs = 1 / (1 + np.exp(-x))

            # 梯度公式: (probs - y) / N
            N = x.shape[0]
            grad = (probs - y) / N

            # 链式法则
            grad = grad * out.grad

            if logits.grad is None:
                logits.grad = grad
            else:
                logits.grad += grad

        out._backward = _backward
        return out
