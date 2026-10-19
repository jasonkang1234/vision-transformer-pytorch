# LLM Wiki — Vision Transformer (ViT)

## Table of Contents
1. [From ViT to Multimodal LLMs](#1-from-vit-to-multimodal-llms)
2. [Vision Transformer Architecture Deep Dive](#2-vision-transformer-architecture-deep-dive)
3. [ViT Variants & Evolution](#3-vit-variants--evolution)
4. [CLIP: Connecting Vision and Language](#4-clip-connecting-vision-and-language)
5. [Multimodal LLMs (MLLMs)](#5-multimodal-llms-mllms)
6. [Vision-Language Models in Practice](#6-vision-language-models-in-practice)
7. [Self-Supervised Vision Learning](#7-self-supervised-vision-learning)
8. [Key Concepts Glossary](#8-key-concepts-glossary)
9. [Further Reading](#9-further-reading)

---

## 1. From ViT to Multimodal LLMs

The Vision Transformer (ViT) you are training in this repo is the **foundational building block** of all modern multimodal AI systems — models that understand both images and text.

```
ViT (2020)
  ↓  Connect image patches to text tokens
CLIP (2021)  ──  Contrastive image-text pretraining
  ↓  Use ViT as visual encoder in an LLM
Flamingo (2022)  ──  Few-shot vision-language model
  ↓  Scale up + instruction tuning
GPT-4V (2023)  ──  Vision-capable GPT-4
Gemini (2023)  ──  Native multimodal (image, audio, video, text)
Claude 3 (2024)  ──  Vision + document understanding
```

---

## 2. Vision Transformer Architecture Deep Dive

### Patch Embedding
ViT treats an image as a **sequence of patches**, analogous to how an LLM treats text as a sequence of tokens.

```
Image: 224×224×3
  → Split into 14×14 = 196 patches of size 16×16
  → Each patch: 16×16×3 = 768 values → Linear projection → 768-dim vector
  → Prepend [CLS] token → sequence of 197 vectors
  → Add learned positional embedding
  → Feed into Transformer encoder
```

### Why [CLS] Token?
Borrowed from BERT. The `[CLS]` token has no positional meaning — it attends to all patches and aggregates global image information. Its final representation is fed to the classification head.

### Self-Attention in ViT

```
For each patch i:
  Query Qᵢ = Wq · xᵢ
  Key   Kⱼ = Wk · xⱼ   (for all patches j)
  Value Vⱼ = Wv · xⱼ

  Attention(Qᵢ, K, V) = softmax(QᵢKᵀ / √d) · V
```

This allows patch 42 (top-right corner) to attend to patch 150 (bottom-left corner) — **global context with no inductive bias**, unlike CNN's local receptive fields.

---

## 3. ViT Variants & Evolution

| Model | Year | Key Innovation |
|---|---|---|
| **ViT** | 2020 | Original: image patches as tokens |
| **DeiT** | 2021 | Data-efficient ViT; knowledge distillation from CNN teacher |
| **Swin Transformer** | 2021 | Hierarchical ViT with shifted windows; efficient for dense prediction |
| **BEiT** | 2021 | BERT-style masked image modelling pretraining |
| **MAE** | 2021 | Masked Autoencoder; reconstruct 75% masked patches |
| **ViT-22B** | 2023 | Google; 22 billion parameter ViT |
| **DINOv2** | 2023 | Self-supervised ViT; strong universal visual features |

### ViT Sizes

| Variant | Layers | Hidden | Heads | Params |
|---|:---:|:---:|:---:|:---:|
| ViT-Tiny | 12 | 192 | 3 | 5M |
| ViT-Small | 12 | 384 | 6 | 22M |
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

---

## 4. CLIP: Connecting Vision and Language

**CLIP** (Contrastive Language–Image Pre-Training, OpenAI 2021) is the bridge between ViT and LLMs.

### Architecture
```
Image ──► ViT encoder ──► image embedding
                                  │
                    Contrastive   │  (maximize cosine similarity
                    loss          │   for matched pairs)
                                  │
Text  ──► Text Transformer ──► text embedding
```

### Training Signal
Trained on 400M (image, text) pairs from the internet. For a batch of N pairs:
- Maximise similarity of the N matching (image, text) pairs
- Minimise similarity of the N² − N non-matching pairs

### Zero-Shot Classification with CLIP
```python
from PIL import Image
import torch
import clip

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
text = clip.tokenize(["a cat", "a dog", "a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    probs = (image_features @ text_features.T).softmax(dim=-1)
# No training needed — zero-shot!
```

---

## 5. Multimodal LLMs (MLLMs)

Modern multimodal LLMs combine a **ViT visual encoder** with a **language model decoder**.

### General Architecture
```
Image
  → ViT encoder  →  image tokens
                          │
                     [Projection]
                          │
                          ▼
Text tokens  →  [Concat]  →  LLM decoder  →  Response
```

### Major Multimodal LLMs

| Model | Visual Encoder | Language Model | Capabilities |
|---|---|---|---|
| **GPT-4V** | ViT-based | GPT-4 | Image QA, OCR, charts |
| **Gemini** | Native multimodal | Gemini | Image, audio, video, text |
| **Claude 3** | ViT-based | Claude | Documents, images, charts |
| **LLaVA** | CLIP ViT-L | Vicuna/Llama | Open source, image QA |
| **Flamingo** | NFNet + cross-attention | Chinchilla | Few-shot vision-language |
| **BLIP-2** | ViT + Q-Former | OPT / FlanT5 | Image captioning, VQA |
| **InternVL** | InternViT-6B | InternLM | State-of-art open source |

---

## 6. Vision-Language Models in Practice

### Using Claude for Image Understanding
```python
import anthropic
import base64

client = anthropic.Anthropic()

with open("chart.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode()

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
            {"type": "text", "text": "Describe what you see in this image."}
        ]
    }]
)
print(response.content[0].text)
```

### Using LLaVA Locally (Open Source)
```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

image = Image.open("image.jpg")
inputs = processor("What is in this image?", image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 7. Self-Supervised Vision Learning

Modern ViT models are pretrained **without labels** using self-supervised objectives:

| Method | Objective | Notes |
|---|---|---|
| **MAE** | Reconstruct 75% masked patches | Simple, scalable, strong features |
| **DINO / DINOv2** | Self-distillation with no labels | Excellent dense features for segmentation |
| **BEiT** | Predict discrete visual tokens (like BERT) | Strong fine-tuning performance |
| **CLIP** | Match image to text description | Zero-shot transfer, open vocabulary |

```
# MAE pretraining concept
Original image patches: [P1][P2][P3][P4][P5][P6][P7][P8]
Masked (75%):           [P1][  ][  ][P4][  ][  ][  ][P8]
ViT encodes visible,
Decoder reconstructs:   [P1][P2][P3][P4][P5][P6][P7][P8]
```

---

## 8. Key Concepts Glossary

| Term | Definition |
|---|---|
| **Patch** | Fixed-size non-overlapping tile of an image (e.g. 16×16 pixels) |
| **Token** | A single unit in a sequence — a patch in ViT, a word-piece in LLMs |
| **CLS token** | Special learnable token prepended to the sequence; aggregates global info |
| **Positional embedding** | Learned vector added to each patch to encode its spatial position |
| **Contrastive learning** | Training objective that pulls matching pairs together, pushes non-matches apart |
| **Zero-shot** | Model generalises to new classes never seen during training |
| **VQA** | Visual Question Answering — answering natural language questions about images |
| **OCR** | Optical Character Recognition — reading text from images |
| **Multimodal** | A model that processes more than one type of input (text + image, text + audio) |
| **Cross-attention** | Attention where queries come from one modality and keys/values from another |

---

## 9. Further Reading

- [An Image is Worth 16x16 Words: ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [CLIP: Learning Transferable Visual Models (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [Masked Autoencoders Are Scalable Vision Learners (He et al., 2021)](https://arxiv.org/abs/2111.06377)
- [DINOv2: Learning Robust Visual Features (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [LLaVA: Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [The Illustrated ViT (Jay Alammar)](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Hugging Face Transformers — ViT](https://huggingface.co/docs/transformers/model_doc/vit)
