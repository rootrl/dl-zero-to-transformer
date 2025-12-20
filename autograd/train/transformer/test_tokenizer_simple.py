from mtorch.tokenizer import Tokenizer

# 这是一个很好的单元测试用例
debug_corpus = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
"""

tokenizer = Tokenizer(debug_corpus)
test_sentence = "speak further code"
ids = tokenizer.encode(test_sentence)
print(ids)


decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")

tokenizer = Tokenizer("The cat sat.")

# 2. 测试一个完全陌生的词 "dog"
ids = tokenizer.encode("The dog sat.")

# 3. 验证
print(tokenizer.stoi) # 看看词表长啥样
print(ids)            # 期望: [id_The, 0, id_sat, id_.]  (0 是 <UNK>)
print(tokenizer.decode(ids)) # 期望: "The <UNK> sat ."
