"""Plot train/val loss curves from checkpoints/history.json -> loss_curve.png"""
import json

import matplotlib.pyplot as plt

h = json.load(open("checkpoints/history.json"))
it = [r["iter"] for r in h]
plt.plot(it, [r["train"] for r in h], label="train")
plt.plot(it, [r["val"] for r in h], label="val")
plt.xlabel("iteration"); plt.ylabel("cross-entropy loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Char-level GPT on TinyShakespeare")
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
print("saved loss_curve.png")
