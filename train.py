"""Train the character-level GPT on TinyShakespeare.

  python prepare_data.py
  python train.py                 # GPU recommended (~15-30 min on a T4)
  python train.py --device cpu --max-iters 500 --eval-interval 100   # quick CPU check
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    # model
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    # optimisation
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=200)
    return p.parse_args()


def lr_at(it, args):
    """Linear warm-up, then cosine decay from lr down to min_lr."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = CharTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
    data = {s: np.fromfile(os.path.join(args.data_dir, f"{s}.bin"), dtype=np.uint16) for s in ("train", "val")}

    def get_batch(split):
        d = data[split]
        ix = np.random.randint(0, len(d) - args.block_size - 1, args.batch_size)
        x = torch.from_numpy(np.stack([d[i:i + args.block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + 1 + args.block_size] for i in ix]).astype(np.int64))
        return x.to(args.device), y.to(args.device)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(args.device)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M | device {args.device}")

    # AdamW: weight-decay only the 2D weight matrices, not biases / LayerNorm / embeddings' 1D params.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    history, best_val = [], float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss()
            history.append({"iter": it, **losses, "lr": lr})
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {time.time() - t0:.0f}s")
            with open(os.path.join(args.out_dir, "history.json"), "w") as f:   # survives Ctrl+C
                json.dump(history, f, indent=2)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__, "iter": it,
                            "val_loss": best_val}, os.path.join(args.out_dir, "best.pt"))
        if it == args.max_iters:
            break

        _, loss = model(*get_batch("train"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # gradient clipping
        opt.step()

    print(f"Best val loss: {best_val:.4f}  (checkpoint: {args.out_dir}/best.pt)")


if __name__ == "__main__":
    main()
