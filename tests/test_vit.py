import torch
from vit_model import ViT, vit_tiny, PatchEmbedding


def test_patch_embedding_shape() -> None:
    embed = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=192)
    x = torch.randn(2, 3, 32, 32)
    out = embed(x)
    assert out.shape == (2, 64, 192)  # (B, num_patches, embed_dim)


def test_vit_tiny_output_shape() -> None:
    model = vit_tiny(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_vit_tiny_eval_deterministic() -> None:
    model = vit_tiny(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
