import numpy as np
import matplotlib.pyplot as plt

# 1. 生成 x
x_data = np.linspace(-5, 5, 100) # 比如从 -5 到 5

# 2. 生成 y (标签)
# 逻辑：真实的 w=1, b=0 (分界线在 x=0 处)
# z = 1*x + 0
# 加上一点噪声，让分界线不那么生硬
noise = np.random.normal(0, 0.5, x_data.shape)
z_true = 1 * x_data + 0 + noise

# 核心：把连续的 z 变成 0 或 1
# 如果 z > 0 则为 1，否则为 0
# this is train data
y_data = (z_true > 0).astype(int)

# 现在你去画图，x_data是横轴，y_data是纵轴，你会看到两级台阶
fig, ax = plt.subplots()
ax.plot(x_data, y_data, "ro")
#plt.show()
'''
# 定义函数
z = wx+b
y' = 1/(1+np.exp(-z))

# bce
L = -(y * np.log(y') + (1-y) * np.log(1-y'))

dL/w = dL/y' * dy'/z * dz/w

dL/w = (y'-y) * x
dL/b = (y' -y)

'''

w = 0
b = 0
ln = 0.0001
epochs = 50000

for i in range(epochs):
    z = w * x_data + b
    y_hat = 1/(1+np.exp(-z))

    loss = -(y_data * np.log(y_hat) + (1-y_data) * np.log(1-y_hat))
    loss = np.mean(loss)

    dw = np.mean((y_hat - y_data) * x_data)
    db = np.mean(y_hat - y_data)
    #print(dw)
    #exit()

    w = w - ln * dw
    b = b - ln * db

    if i % 500 == 0:
        print(loss)

#print(loss)
print("\n w:\n", w, "\n b:\n", b)

# print(len(y_data))
