import numpy as np
from vocabulary import Vocabulary
import os
from pathlib import Path
from config import cfg

if __name__ == '__main__':
    data_dir = Path(cfg.DATA_DIR)
    vocab_dir = data_dir / 'vocab.pickle'

    x_valid = data_dir / (cfg.VALID_FILE + '.x.np')
    y_valid = data_dir / (cfg.VALID_FILE + '.y.np')

    vocabulary = Vocabulary()
    vocabulary.load(vocab_dir)
    print(f"len(vocabulary)={len(vocabulary)}")

    x = np.load(x_valid, allow_pickle=True)
    y = np.load(y_valid, allow_pickle=True)

    print(f'x={x[0]}, y={y[0]}')
    print(f'x={[vocabulary.get_token(token_idx) for token_idx in x[0]]}, y={vocabulary.get_token(y[0])}')

    print(f'x={x[1]}, y={y[1]}')
    print(f'x={[vocabulary.get_token(token_idx) for token_idx in x[1]]}, y={vocabulary.get_token(y[1])}')
