from mtorch.tensor import Tensor
import numpy as np
if __name__ == "__main__":
    # 模拟一个简单的线性层: Z = X @ W
    # Batch Size = 2, Input Dim = 3, Output Dim = 2
    
    # 输入 X (2x3)
    X = Tensor([[1.0, 2.0, 3.0], 
                [4.0, 5.0, 6.0]], label="X")
    
    # 权重 W (3x2) - 也就是我们要更新的参数
    W = Tensor([[1.0, 0.0], 
                [0.0, 1.0], 
                [1.0, 1.0]], label="W")
    
    # Forward
    Z = X @ W
    
    # 模拟 Loss 回传：假设 Loss 对 Z 的梯度全为 1
    Z.grad = np.ones_like(Z.data)
    
    # Backward
    Z._backward()
    
    print("X Shape:", X.data.shape)
    print("W Shape:", W.data.shape)
    print("Z Shape:", Z.data.shape)
    
    print("\nCalculated W.grad:\n", W.grad)
    
    # 你的直觉验证：
    # W.grad 应该是 X.T @ Z.grad
    # X.T 是 (3,2), Z.grad 是 (2,2) -> 结果 (3,2)
    # 算一下第一行第一列：1*1 + 4*1 = 5.0
    print("\nExpected W.grad[0,0] should be 5.0")
