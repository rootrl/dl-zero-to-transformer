import numpy as np
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

