from typing import List
from pathlib import Path
import re
from vocabulary import Vocabulary
from config import cfg
import pickle

class Punctuation:
    def __init__(self):
        p_geminada = r'(?P<geminada>l\s*·\s*l)'
        p_html = r'&apos;|&quot;'
        p_punct = r'[^\w\s·]|_'
        p_spaces = r'\s+'

        self.pattern = re.compile(
            f"(?:{p_geminada})|(?:{p_html})|(?:{p_punct})|(?:{p_spaces})"
        )

    def clean(self, s: str):
        return self.pattern.sub(
            # we transform all forms of geminada to l·l
            lambda m: 'l·l' if m.group('geminada') else ' ',
            s
        ).strip()

def build_vocabulary(train_file: Path, vocab_file: Path) -> Vocabulary:
    vocab = Vocabulary()
    punctuation = Punctuation()

    counter = 0
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip().rstrip()
            if line:
                processed_line = punctuation.clean(line)
                for token in processed_line.split():
                    counter += 1
                    vocab.add_token(token)

    print(f"Number of tokens (with punctuation processed) = {counter}")
    vocab.save(vocab_file)

    return vocab

def build_dataset(files: List[Path], vocab: Vocabulary, window_size: int):
    punctuation = Punctuation()

    for file in files:
        print(f"Builiding dataset for file '{file}'")

        with open(file, 'r') as f:
            for line in f:
                line = line.strip().rstrip()
                if line:
                    processed_line = punctuation.clean(line)
                    
                    token_line = []
                    for token in processed_line.split():
                        token_line.append(vocab.get_index(token))

                    left_window = window_size // 2
                    right_window = window_size - left_window - 1

                    if len(token_line) < left_window:
                        continue

                    samples = []
                    targets = []

                    aux_line = [vocabulary.pad_idx] * left_window + token_line + [vocabulary.pad_idx] * right_window
                    for i, token_idx in enumerate(token_line):
                        samples.append(aux_line[i:(i+left_window)] + aux_line[(i+left_window+1):(i+window_size)])
                        targets.append(token_idx)

        print(f"Writing targets and samples for file '{file}'")
        with open(file.with_name(file.name + ".x.np"), 'wb') as f:
            pickle.dump(samples, f)

        with open(file.with_name(file.name + ".y.np"), 'wb') as f:
            pickle.dump(targets, f)


if __name__ == '__main__':
    FILES = [cfg.VALID_FILE, cfg.TEST_FILE, cfg.TRAIN_FILE]
    data_dir = Path(cfg.DATA_DIR)
    vocab_file = data_dir / 'vocab.pickle'

    if vocab_file.exists() and vocab_file.is_file():
        print(f"Loading Vocabulary from {vocab_file}")
        vocabulary = Vocabulary()
        vocabulary.load(vocab_file)
    else:
        print("Vocabulary file does not exist, building...")
        vocabulary = build_vocabulary(
            train_file=data_dir / cfg.TRAIN_FILE, 
            vocab_file=vocab_file
        )

    print(f"Vocabulary length = {len(vocabulary)}")

    build_dataset(
        files=[data_dir / x for x in FILES], 
        vocab=vocabulary, 
        window_size=cfg.WINDOW_SIZE
    )
