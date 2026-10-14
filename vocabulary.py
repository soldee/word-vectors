import pickle
from pathlib import Path

class Vocabulary:
    def __init__(self, unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.token2idx = {}
        self.idx2token = []
        self.token2counter = {}
        self._len = 0

        self.unk_idx = self._add_token_own(unk_token)
        self.pad_idx = self._add_token_own(pad_token)

    def _add_token_own(self, token):
        self.idx2token.append(token)
        idx = self._len
        self.token2idx[token] = idx
        self._len += 1
        return idx

    def add_token(self, token: str, threshold: int):
        if token not in self.token2idx:
            count = self.token2counter[token] = self.token2counter.get(token, 0) + 1
            if count > threshold:
                self.idx2token.append(token)
                self.token2idx[token] = self._len
                self._len += 1

    def get_index(self, token: str) -> str:
        return self.token2idx.get(token, self.unk_idx)

    def get_token(self, idx: int):
        return self.idx2token[idx]

    def __len__(self):
        return self._len

    def save(self, file: Path):
        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, file: Path):
        with open(file, 'rb') as f:
            self.__dict__.update(pickle.load(f))
