from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F


class DinoViT(VisionTransformer):
    def __init__(self, device="cuda"):

        super().__init__(
            img_size=28,
            patch_size=4,
            in_chans=1,
            num_classes=0,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            device=device,
        )


class DinoHead(nn.Module):
    def __init__(self, n_embed, n_out):
        super().__init__()
        self.embed_dim = n_embed
        self.out_dim = n_out
        self.layers = nn.Sequential(
            nn.Linear(n_embed, n_out),
            # nn.GELU(),
            # nn.Linear(4 * n_embed, n_out),
        )

    def forward(self, x):
        return self.layers(x)
        # return F.normalize(self.layers(x), dim=-1)
