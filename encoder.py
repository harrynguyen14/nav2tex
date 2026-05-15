import torch
import torch.nn as nn
import timm
from timm.data import resolve_model_data_config, create_transform


MODEL_NAME = "naflexvit_base_patch16_gap.e300_s576_in1k"


class NaFlexViTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        self.hidden_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.proj = (
            nn.Linear(self.hidden_dim, output_dim)
            if output_dim is not None and output_dim != self.hidden_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, max_patches: int | None = None) -> torch.Tensor:
        features = self.backbone.forward_features(x)  # (B, S, C)
        features = self.proj(features)
        if max_patches is None:
            return features
        B, S, C = features.shape
        if S > max_patches:
            features = features[:, :max_patches, :]
        elif S < max_patches:
            pad = features.new_zeros(B, max_patches - S, C)
            features = torch.cat([features, pad], dim=1)
        return features


def build_transform(model_name: str = MODEL_NAME, is_training: bool = False):
    cfg = resolve_model_data_config(timm.create_model(model_name, pretrained=False))
    mean, std = cfg["mean"], cfg["std"]
    # No spatial resize — _patch_aware_resize in dataset controls image size.
    # Only convert to tensor and normalize.
    from torchvision import transforms
    ops = []
    if is_training:
        ops += [transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)
