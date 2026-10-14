"""Generate text from a trained checkpoint.

  python sample.py --prompt "ROMEO:" --tokens 500 --temperature 0.8 --top-k 40
"""
import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoints/best.pt")
p.add_argument("--data-dir", default="data")
p.add_argument("--prompt", default="\n")
p.add_argument("--tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

torch.manual_seed(args.seed)
tok = CharTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
ckpt = torch.load(args.ckpt, map_location=args.device)
model = GPT(GPTConfig(**ckpt["config"])).to(args.device).eval()
model.load_state_dict(ckpt["model"])

unknown = sorted(set(args.prompt) - set(tok.chars))
if unknown or not args.prompt:
    raise SystemExit(f"Prompt must be non-empty and only use characters from the training text; "
                     f"unknown: {unknown}")
idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print(tok.decode(out[0].tolist()))
