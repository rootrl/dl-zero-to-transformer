import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(42) # 固定随机种子，保证结果可复现

def function(x,w,b):
    return w*x+b

x_data = np.linspace(0, 100, 50)
#x_data = [2,3,4,5,6,7,8,18,13,10]

w = 0.9
b = 125
e = np.random.normal(0, 150 * 0.04, x_data.shape)

y_data_true = [function(x,w,b) for x in x_data]
y_data_pred = w * x_data + b + e 
#y_data_true = x_data * function(x,w,b) 
# how to use function?
# y_data_true = x_data * w + b + e 

# plot
fig, ax = plt.subplots()
fig.suptitle("main title")
ax.plot(x_data, y_data_pred, 'ro')
ax.plot(x_data, y_data_true, 'b')
ax.set_title("y = wx + b")
ax.set_xlabel("x")
ax.set_ylabel("y")

#plt.show()


# figue out w,b
'''
loss = 1/n * np.sum((y' - y)**2)


y'= wx + b

u = y' - y

2u * 1 * x

dw = 2(y' - y) * x
dy = 2(y' - y)

loss 最小

找出 w, b

'''

dw = 0
db = 0
ln = 0.0001
n = len(x_data)

# 梯度下降
for i in range(50000):
    y_pred = dw * x_data + db
    loss = 1/n * np.sum((y_pred - y_data_pred)**2)
    dw = dw - ln * (2/n * np.sum((y_pred - y_data_pred) * x_data))
    db = db - ln * (2/n * np.sum((y_pred - y_data_pred)))
    if i % 500 == 0:
        print("loss:", loss)
        print("acurate:", np.sum(y_data_true - dw * x_data + db) / n)

print(dw, db)
y_train = dw * x_data + b
plt.plot(x_data, y_train, 'g')
residuals = y_data_pred - (dw * x_data + db) # 注意这里用你训练出的参数
plt.hist(residuals, bins=5)
plt.title("Residuals Histogram")
#plt.show()
