from __future__ import annotations

import copy

import torch
import torch.nn as nn


class SmallResNet(nn.Module):
    """Lightweight ResNet for 32x32 inputs (~400K params at embed_dim=256)."""

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


class TripletJEPA(nn.Module):
    """Online encoder + predictor + EMA target encoder (I-JEPA style)."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.encoder = SmallResNet(in_channels, embed_dim)
        self.predictor = Predictor(embed_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        m = self.ema_momentum
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(m).add_(online.data, alpha=1.0 - m)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self, corrupt: torch.Tensor, clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_anchor = self.predictor(self.encoder(corrupt))
        z_positive = self.target_encoder(clean)
        return z_anchor, z_positive

    def param_count(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        pred = sum(p.numel() for p in self.predictor.parameters())
        return {"encoder": enc, "predictor": pred, "total_trainable": enc + pred}
