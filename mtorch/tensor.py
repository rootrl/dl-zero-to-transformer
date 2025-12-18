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

    # only for forward.
    def softmax(self):
        out = np.exp(self.data)
        probs = out / np.sum(out, axis=1, keepdims=True)
        return Tensor(probs, _children=(self,), _op='softmax')

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

# 优化器
class Optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters) 
    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError("子类必须实现 step 方法")

class SGD(Optimizer):
    def __init__(self, parameters, lr=1.0):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0 # 时间步 t
        
        self.m = {} # 一阶矩 (Momentum)
        self.v = {} # 二阶矩 (Velocity)

    def step(self):
        self.t += 1
        
        for p in self.parameters:
            # 如果参数没有梯度（没参与计算），跳过
            if p.grad is None:
                continue
                
            # 1. 初始化状态 (第一次运行时)
            if p not in self.m:
                # 必须和 p.data 形状一样，初始为 0
                self.m[p] = np.zeros_like(p.data)
                self.v[p] = np.zeros_like(p.data)
            
            # 取出当前参数对应的状态
            m = self.m[p]
            v = self.v[p]
            grad = p.grad

            # 2. 更新 m (动量)
            # m = beta1 * m + (1 - beta1) * g
            self.m[p] = self.beta1 * m + (1 - self.beta1) * grad
            
            # 3. 更新 v (能量)
            # v = beta2 * v + (1 - beta2) * g^2
            self.v[p] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # 4. 偏差修正 (Bias Correction)
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            # 5. 参数更新
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

