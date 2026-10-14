# GPT-Style Transformer from Scratch

A ~10.8M-parameter decoder-only transformer, written from scratch in PyTorch and trained on TinyShakespeare at the character level.

## What's implemented ([`model.py`](model.py))

- Multi-head **causal self-attention**: fused QKV projection, scaled dot-product, lower-triangular mask
- Learned **token + positional embeddings** (with weight tying between the embedding and output layer)
- **Pre-LayerNorm** blocks with **residual** connections
- **MLP** (4× expansion, GELU)
- **Character-level tokenizer** ([`tokenizer.py`](tokenizer.py))
- **Temperature + top-k sampling** (`GPT.generate`)

Training ([`train.py`](train.py)):
- **AdamW** with weight decay applied only to 2D weight matrices
- **Linear warm-up + cosine decay** learning-rate schedule
- **Gradient clipping** (max norm 1.0)
- Periodic train/val evaluation, best-checkpoint saving, loss history to JSON

Default config: 6 layers, 6 heads, 384-dim embeddings, 256-char context, dropout 0.2.

## Run

```bash
pip install -r requirements.txt
python prepare_data.py          # downloads TinyShakespeare, 90/10 split
python train.py                 # 5000 iters; ~15-30 min on a T4 GPU
python sample.py --prompt "ROMEO:" --temperature 0.8 --top-k 40
python plot_loss.py             # saves loss_curve.png
```

No GPU? Colab's free T4 works. On CPU, try `--max-iters 500` to check everything runs (it will be slow and the text will be rough).

## Results

> Fill in from **your** run.

| | Value |
|---|---|
| Parameters | 10.76M |
| Hardware | |
| Training time | |
| Best validation loss | |

<!-- After running plot_loss.py, commit loss_curve.png and uncomment:
![loss curve](loss_curve.png)
-->

### Sample (temperature 0.8, top-k 40)

```
paste a sample here
```

## What I learned

_Fill in: what happened when you removed warm-up or gradient clipping, how temperature and top-k change the output, where it starts to overfit._

## Experiments to try

- Swap learned positional embeddings for RoPE
- Use `F.scaled_dot_product_attention` and compare speed
- Byte-pair encoding instead of characters
- Scale the model up and plot val loss against parameter count
