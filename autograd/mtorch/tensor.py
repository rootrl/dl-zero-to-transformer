import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        
        self.data = np.array(data, dtype=np.float32)
        
        self.grad = np.zeros_like(self.data)
        
        self._backward = lambda: None
        
        self._prev = set(_children)

        self._op = _op

        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            grad = out.grad

            # 1. 如果 self 维度比 grad 少 (例如 (8, 32) vs (32, 8, 32))，说明发生了广播，要先把前面的维度 sum 掉
            grad_self = grad
            while grad_self.ndim > self.data.ndim:
                grad_self = np.sum(grad_self, axis=0)

            # 2. 如果维度一样，但 self 某些维度是 1 (例如 (1, 32) vs (32, 32))，要在这些维度上 sum
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)

            self.grad += grad_self

            # --- 处理 other (同理) ---
            grad_other = grad
            # 1. 处理维度差异 (3D -> 2D) 
            while grad_other.ndim > other.data.ndim:
                grad_other = np.sum(grad_other, axis=0)

            # 2. 处理广播维度 (N, 1) -> (N, M)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)

            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            grad = out.grad

            # --- 1. 对 self 求导 (公式: grad * other) ---
            grad_self = grad * other.data

            # Unbroadcast: 处理广播导致的维度差异
            # Step A: 如果 grad 维度比 self 多，先把前面的维度 sum 掉
            while grad_self.ndim > self.data.ndim:
                grad_self = np.sum(grad_self, axis=0)
            # Step B: 如果 self 某些维度是 1，把这些维度 sum 掉保持形状
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)

            self.grad += grad_self

            # --- 2. 对 other 求导 (公式: grad * self) ---
            grad_other = grad * self.data

            # Unbroadcast: 同上
            while grad_other.ndim > other.data.ndim:
                grad_other = np.sum(grad_other, axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)

            other.grad += grad_other

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            grad = out.grad

            # --- 1. 对 self (输入 X) 求导 ---
            # 公式: dL/dX = dL/dY @ W.T
            # 技巧: 用 swapaxes(-1, -2) 代替 .T，只转置最后两个维度，保护 Batch 维
            if other.data.ndim > 1:
                other_T = other.data.swapaxes(-1, -2)
            else:
                other_T = other.data

            grad_self = grad @ other_T

            # Unbroadcast: 处理可能发生的广播 (虽然 Linear 输入通常不广播，但为了鲁棒性)
            # 逻辑同 __add__
            while grad_self.ndim > self.data.ndim:
                grad_self = np.sum(grad_self, axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)

            self.grad += grad_self

            # --- 2. 对 other (权重 W) 求导 ---
            # 公式: dL/dW = X.T @ dL/dY
            # 技巧: 同样只转置 X 的最后两维
            if self.data.ndim > 1:
                self_T = self.data.swapaxes(-1, -2)
            else:
                self_T = self.data

            grad_other = self_T @ grad

            # Unbroadcast (关键步骤!)
            # 此时 grad_other 可能是 (Batch, In, Out)，但权重 other 是 (In, Out)
            # 必须把 Batch 维度 sum 掉
            while grad_other.ndim > other.data.ndim:
                grad_other = np.sum(grad_other, axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)

            other.grad += grad_other

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1) 

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "MicroTorch only supports float/int power for now"

        out = Tensor(self.data ** other, _children=(self,), _op=f'**')
        def _backward():
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward
        return out

    def sum(self):
        # 1. Forward
        out = Tensor(np.sum(self.data), _children=(self,), _op='sum')

        # 2. Backward
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """
        支持两种调用方式:
        x.reshape(256, 65)
        x.reshape((256, 65))
        """
        # 1. 参数处理 (兼容 *args 和 tuple)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        # 2. Forward
        out_data = self.data.reshape(shape)
        out = Tensor(out_data, _children=(self,), _op='reshape')

        # 3. Backward
        def _backward():
            # 关键：self.data.shape 是原始形状 (例如 32, 8, 65)
            # out.grad 是传回来的形状 (例如 256, 65)
            # 我们只需要把 grad 捏回原始形状即可
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out


    def transpose(self, dim0, dim1):
        """
        交换两个维度。
        Example: x.transpose(1, 2)
        """
        # 1. Forward
        # np.swapaxes 返回的是视图(View)，非常高效
        out_data = np.swapaxes(self.data, dim0, dim1)
        out = Tensor(out_data, _children=(self,), _op='transpose')
        
        # 2. Backward
        def _backward():
            # 梯度的反向传播很简单：把维度再换回来即可
            if self.grad is None:
                self.grad = np.swapaxes(out.grad, dim0, dim1)
            else:
                self.grad += np.swapaxes(out.grad, dim0, dim1)
                
        out._backward = _backward
        return out


    def sigmoid(self):
        #z = 1/(1+np.exp(-Z))
        z = 1/(1+np.exp(-self.data))

        #forward
        out = Tensor(z, _children=(self,), _op='sigmoid')

        #backward
        def _backward():
               self.grad += out.grad * (z * (1-z))

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def softmax(self, dim=-1):
        """
        Softmax 函数。
        用于 Attention 机制内部的权重归一化。
        注意：如果是计算 CrossEntropyLoss，请直接使用 logits，不要先调这个函数。
        """
        # --- 1. Forward ---
        x = self.data

        # Numerical Stability: max trick
        # 减去最大值不改变 Softmax 结果，但能防止 exp 溢出
        c = np.max(x, axis=dim, keepdims=True)
        e_x = np.exp(x - c)

        # 计算分母
        sum_e_x = np.sum(e_x, axis=dim, keepdims=True)

        out_data = e_x / sum_e_x

        out = Tensor(out_data, _children=(self,), _op='softmax')

        # --- 2. Backward ---
        def _backward():
            # Softmax Gradient:
            # dL/dx_i = y_i * (dL/dy_i - sum_k(dL/dy_k * y_k))
            # 这里的推导非常漂亮，利用了 y 自身的性质

            y = out_data
            grad = out.grad

            # 计算 dot term: sum(grad * y)
            # 保持维度以便广播
            dot = np.sum(grad * y, axis=dim, keepdims=True)

            # 最终梯度
            self_grad = (grad - dot) * y

            if self.grad is None:
                self.grad = self_grad
            else:
                self.grad += self_grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward() 
        

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

