# Vision Transformer (ViT) — PyTorch

A clean PyTorch implementation of the **Vision Transformer (ViT)** for image classification.
Based on ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020).

## Architecture

```
Input Image
  → Patch Embedding (Conv2D stride=patch_size)
  → [CLS] token + Positional Embedding
  → Transformer Encoder (depth × [MHSA + FFN + LayerNorm])
  → CLS token → MLP Head → Class logits
```

## Variants

| Model | Image | Patch | Embed | Depth | Heads | Params |
|---|---|---|---|---|---|---|
| `vit_tiny` | 32×32 | 4×4 | 192 | 12 | 3 | ~5M |
| `vit_base` | 224×224 | 16×16 | 768 | 12 | 12 | ~86M |

## Setup

```bash
pip install -r requirements.txt
```

## Train on CIFAR-10

```bash
python main.py
python main.py --epochs 100 --batch-size 64 --lr 1e-4
```

## Run Tests

```bash
pytest
```

## Reference

- Paper: https://arxiv.org/abs/2010.11929
- Original JAX code: https://github.com/google-research/vision_transformer

## License

MIT
