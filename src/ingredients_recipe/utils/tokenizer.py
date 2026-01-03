import re
from collections import Counter
from typing import List

class SimpleTokenizer:
    def __init__(self, min_freq: int = 1, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def _preprocess(self, text: str):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9,.!?'\s]", " ", text)
        tokens = text.split()
        return tokens

    def build_vocab(self, texts: List[str], max_vocab: int = 20000):
        counter = Counter()
        for t in texts:
            counter.update(self._preprocess(t))
        words = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        most_common = [w for w, c in counter.most_common(max_vocab) if c >= self.min_freq]
        words.extend(most_common)
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str, max_length: int = 128):
        tokens = self._preprocess(text)
        ids = [self.word2idx.get(self.bos_token)]
        for t in tokens:
            ids.append(self.word2idx.get(t, self.word2idx.get(self.unk_token)))
            if len(ids) >= max_length - 1:
                break
        ids.append(self.word2idx.get(self.eos_token))
        if len(ids) < max_length:
            ids += [self.word2idx.get(self.pad_token)] * (max_length - len(ids))
        return ids[:max_length]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), self.unk_token)
            if w in (self.bos_token, self.eos_token, self.pad_token):
                continue
            words.append(w)
        return " ".join(words)
