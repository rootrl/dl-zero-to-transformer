import re

class Tokenizer:
    def __init__(self, text, unk_token="<UNK>"):
        self.unk_token = unk_token
        self.pattern = r"\w+|[^\w\s]\s+"
        # 1. 预处理：正则分词
        tokens = re.findall(self.pattern, text)
        # 2. 构建词表：去重
        unique_tokens = sorted(list(set(tokens)))
        # 3. 增加特殊 token: <UNK>
        unique_tokens.insert(0, self.unk_token)
        # 4.构建 stoi, itos
        self.stoi = {token: i for i, token in enumerate(unique_tokens)}
        self.itos = {i: token for i, token in enumerate(unique_tokens)}
        
    def encode(self, text):
        # 1. 正则分词
        tokens = re.findall(self.pattern, text)
        # 2. 查表 (如果词不存在，返回 <UNK> 的 index)
        ids = [self.stoi.get(token, 0) for token in tokens]
        # return List[int]
        return ids

    def decode(self, ids):
        # List[int] -> List[str] -> join with space? 
        tokens = [self.itos.get(i, self.unk_token) for i in ids]
        return "".join(tokens)
