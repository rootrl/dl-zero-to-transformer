from mtorch.tensor import Tensor
# 测试脚本
x = Tensor(2.0, label='x')
y = Tensor(3.0, label='y')

# 1. Forward
z = x * y    # z.data = 6.0
l = z + x    # l.data = 8.0 (这里 x 被复用了！既在 z 里，又在 l 里)

# 2. Manually trigger Backward (我们还没写自动引擎，先手动逆序调用)
# 假设 l 是 Loss，所以 l.grad 初始为 1.0
l.grad = 1.0

# 倒序执行 _backward
l._backward() # 这会把梯度传给 z 和 x(作为加数)
z._backward() # 这会把梯度传给 x(作为乘数) 和 y

# 3. 验证结果
# 数学推导：
# l = x * y + x
# dl/dx = y + 1 = 3 + 1 = 4
# dl/dy = x = 2

print(f"x.grad (Expect 4.0): {x.grad}")
print(f"y.grad (Expect 2.0): {y.grad}")
