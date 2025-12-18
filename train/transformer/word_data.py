import os
import requests

# 确保 data 目录存在
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, 'input.txt')

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not os.path.exists(file_path):
    print(f"Downloading {url} to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(url).text)
    print("Download finished.")
else:
    print("input.txt already exists.")

# 打印前 200 个字符看看长啥样
with open(file_path, 'r', encoding='utf-8') as f:
    print("\nData Sample:\n" + "-"*20)
    print(f.read(200))
    print("-" * 20)
