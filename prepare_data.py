"""Downloads TinyShakespeare (~1.1M characters) and writes train/val token files."""
import os
import urllib.request

import numpy as np

from tokenizer import CharTokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    text = open(path, encoding="utf-8").read()

    tok = CharTokenizer.from_text(text)
    tok.save(os.path.join(DATA_DIR, "vocab.json"))
    ids = np.array(tok.encode(text), dtype=np.uint16)

    split = int(0.9 * len(ids))   # 90/10 train/val split
    ids[:split].tofile(os.path.join(DATA_DIR, "train.bin"))
    ids[split:].tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"{len(text):,} chars | vocab {tok.vocab_size} | train {split:,} | val {len(ids) - split:,}")


if __name__ == "__main__":
    main()
