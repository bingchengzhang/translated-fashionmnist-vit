"""Classification models for translated 64 x 64 FashionMNIST images."""

from __future__ import annotations

import torch
from torch import nn


class ConvPatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.projection(images)
        return patches.flatten(2).transpose(1, 2)


class LinearPatchEmbedding(nn.Module):
    """Flatten every patch and project it with a linear layer."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = images.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.flatten(1, 2).flatten(2)
        return self.projection(patches)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        patch_embedding: str = "conv",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        embedding_classes = {
            "conv": ConvPatchEmbedding,
            "linear": LinearPatchEmbedding,
        }
        if patch_embedding not in embedding_classes:
            raise ValueError("patch_embedding must be 'conv' or 'linear'.")
        self.patch_embed = embedding_classes[patch_embedding](
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


class MLPClassifier(nn.Module):
    """Parameter-scale-matched MLP for 64 x 64 grayscale inputs."""

    def __init__(self, image_size: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        input_dim = image_size * image_size
        hidden_dim = 128
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class CNNClassifier(nn.Module):
    """Compact CNN with local receptive fields and shared spatial filters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
