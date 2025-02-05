import os

import numpy as np
import requests

from .tokenizing import encode


DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets", "shakespeare")

def get_shakespeare_data(tokenizer: str):
    """Inspired from https://github.com/karpathy/nanoGPT/"""
    raw_path = os.path.join(DATA_PATH, "raw.txt")
    train_path = os.path.join(DATA_PATH, f"train_{tokenizer}.npy")
    test_path = os.path.join(DATA_PATH, f"test_{tokenizer}.npy")

    # if path is not even there, download all data
    if not os.path.exists(DATA_PATH):
        print("Downloading raw Shakespeare texts")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        os.makedirs(DATA_PATH, exist_ok=True)
        text = requests.get(url, timeout=60).text
        with open(raw_path, "w+", encoding="utf8") as f:
            f.write(text)

    # attempt to find cached version for current tokenizer
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Tokenizing Shakespeare texts")
        # load text
        with open(raw_path, encoding="utf8") as f:
            text = "".join(f.readlines())
        i = int(0.8*len(text))
        # encode text
        x = np.array(encode(text[:i], tokenizer), dtype=np.uint16)
        x_test = np.array(encode(text[i:], tokenizer), dtype=np.uint16)
        # map memory
        mem = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=x.shape)
        mem[:] = x
        mem = np.memmap(test_path, dtype=np.uint16, mode="w+", shape=x_test.shape)
        mem[:] = x_test

    # at this point we know that the binfile was properly created so we load it
    return {"train": np.memmap(train_path, dtype=np.uint16, mode="r"),
            "val": np.memmap(test_path, dtype=np.uint16, mode="r")}
