import numpy as np
import requests
import gzip
import os

def load_mnist():
    """
    下载并解析 MNIST 数据集，返回 (X_train, Y_train), (X_test, Y_test)
    没有任何黑盒魔法，只有纯粹的文件解压和字节读取。
    """
    #base_url = "http://yann.lecun.com/exdb/mnist/"
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    # 下载数据
    if not os.path.exists("data"):
        os.makedirs("data")
        
    for file in files:
        file_path = os.path.join("data", file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            url = base_url + file
            r = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(r.content)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as f:
            # 读取魔数和维度信息 (大端序)
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            # 读取像素数据
            buffer = f.read()
            data = np.frombuffer(buffer, dtype=np.uint8)
            data = data.reshape(num_images, rows * cols) # 展平: (N, 28*28)
            return data

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_items = int.from_bytes(f.read(4), 'big')
            buffer = f.read()
            data = np.frombuffer(buffer, dtype=np.uint8)
            return data.reshape(-1, 1) # (N, 1)

    print("Parsing training data...")
    X_train = parse_images(os.path.join("data", "train-images-idx3-ubyte.gz"))
    Y_train = parse_labels(os.path.join("data", "train-labels-idx1-ubyte.gz"))
    
    print("Parsing test data...")
    X_test = parse_images(os.path.join("data", "t10k-images-idx3-ubyte.gz"))
    Y_test = parse_labels(os.path.join("data", "t10k-labels-idx1-ubyte.gz"))

    # 归一化 (Normalization): 将像素值 0-255 缩放到 0-1 之间
    # 这是一个关键的“稳定器”步骤！
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    return (X_train, Y_train), (X_test, Y_test)

'''
Usage
# 执行加载
(X_train, Y_train), (X_test, Y_test) = load_mnist()

print("\nData Loaded Success!")
print(f"X_train shape: {X_train.shape}") # 期待 (60000, 784)
print(f"Y_train shape: {Y_train.shape}") # 期待 (60000, 1)
'''
