from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        downsample: nn.Module | None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return torch.relu(out + identity)


class SmallResNet(nn.Module):
    """Lightweight ResNet for 32x32 inputs."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
        )

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride, downsample),
            ResidualBlock(out_ch, out_ch, 1, None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class CifarResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 inputs (CIFAR-style stem, no ImageNet weights).

    Matches the SCAN / SimCLR-CIFAR backbone convention: 3x3 stride-1 stem, no maxpool.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 512) -> None:
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(net.fc.in_features, embed_dim)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CifarResNet50(nn.Module):
    """ResNet-50 adapted for 32x32 inputs (CIFAR-style stem, no ImageNet weights)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        net = resnet50(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(net.fc.in_features, embed_dim)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, bias=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """Compact ViT for 32x32 inputs (patch 4 -> 8x8 tokens + CLS).

    Self-attention over patches lets masked / augmented views aggregate global context,
    which CNN avg-pool cannot do as explicitly.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        image_size: int = 32,
        patch_size: int = 4,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )
        self.blocks = nn.ModuleList(
            ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._reset_linear()

    def _reset_linear(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, 0])


def build_encoder(
    backbone: str,
    *,
    in_channels: int = 3,
    embed_dim: int = 256,
    image_size: int = 32,
    vit_patch_size: int = 4,
    vit_depth: int = 6,
    vit_heads: int = 4,
    vit_mlp_dim: int = 512,
) -> nn.Module:
    name = backbone.lower()
    if name in {"resnet", "small_resnet", "cnn"}:
        return SmallResNet(in_channels, embed_dim)
    if name in {"resnet18", "resnet_18"}:
        return CifarResNet18(in_channels, embed_dim)
    if name in {"resnet50", "resnet_50"}:
        return CifarResNet50(in_channels, embed_dim)
    if name in {"vit", "small_vit", "vit_tiny", "deit_tiny"}:
        # vit_tiny / deit_tiny: paper DeiT-Ti defaults if caller left custom small-vit sizes.
        if name in {"vit_tiny", "deit_tiny"}:
            embed_dim = 192
            vit_depth = 12
            vit_heads = 3
            vit_mlp_dim = 768
        return SmallViT(
            in_channels=in_channels,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=vit_patch_size,
            depth=vit_depth,
            num_heads=embed_dim // 64 if vit_heads <= 0 else vit_heads,
            mlp_dim=vit_mlp_dim,
        )
    raise ValueError(
        f"Unknown backbone={backbone!r}; expected resnet, resnet18, resnet50, vit, or vit_tiny"
    )


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PrototypeBank(nn.Module):
    """Learnable unit-sphere prototypes for SwAV-style unsupervised clustering."""

    def __init__(self, embed_dim: int, num_prototypes: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        nn.init.normal_(self.prototypes, std=0.02)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)


class JEPA(nn.Module):
    """Encoder (+ optional predictor) with EMA target or single-encoder stop-grad."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        ema_momentum: float = 0.996,
        use_ema_target: bool = True,
        num_prototypes: int = 0,
        backbone: str = "resnet",
        image_size: int = 32,
        vit_patch_size: int = 4,
        vit_depth: int = 6,
        vit_heads: int = 4,
        vit_mlp_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = build_encoder(
            backbone,
            in_channels=in_channels,
            embed_dim=embed_dim,
            image_size=image_size,
            vit_patch_size=vit_patch_size,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            vit_mlp_dim=vit_mlp_dim,
        )
        self.predictor = Predictor(embed_dim)
        self.use_ema_target = use_ema_target
        self.ema_momentum = ema_momentum
        self.prototype_bank = (
            PrototypeBank(embed_dim, num_prototypes) if num_prototypes > 0 else None
        )
        if use_ema_target:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = None

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if not self.use_ema_target or self.target_encoder is None:
            return
        m = self.ema_momentum
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Positive/negative path: EMA encoder or same encoder (stop-grad applied outside)."""
        if self.use_ema_target:
            assert self.target_encoder is not None
            return self.target_encoder(x)
        return self.encoder(x)

    def anchor_embedding(
        self, corrupt: torch.Tensor, clean: torch.Tensor, anchor_mode: str
    ) -> torch.Tensor:
        if anchor_mode == "predictor_corrupt":
            return self.predictor(self.encoder(corrupt))
        if anchor_mode == "encoder_corrupt":
            return self.encoder(corrupt)
        if anchor_mode == "encoder_clean":
            return self.encoder(clean)
        raise ValueError(
            f"Unknown anchor_mode={anchor_mode!r}; "
            "expected predictor_corrupt, encoder_corrupt, or encoder_clean"
        )

    def forward(
        self,
        corrupt: torch.Tensor,
        clean: torch.Tensor,
        anchor_mode: str = "predictor_corrupt",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_anchor = self.anchor_embedding(corrupt, clean, anchor_mode)
        z_positive = self.encode_target(clean)
        return z_anchor, z_positive

    def param_count(self, anchor_mode: str = "predictor_corrupt") -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        pred = sum(p.numel() for p in self.predictor.parameters())
        proto = (
            sum(p.numel() for p in self.prototype_bank.parameters())
            if self.prototype_bank is not None
            else 0
        )
        trainable = enc + (pred if anchor_mode == "predictor_corrupt" else 0) + proto
        return {
            "backbone": self.backbone,
            "encoder": enc,
            "predictor": pred,
            "prototypes": proto,
            "total_trainable": trainable,
            "use_ema_target": self.use_ema_target,
        }
