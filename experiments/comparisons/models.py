"""MLP, CNN and ViT models used in the comparison study."""

from __future__ import annotations

from torch import nn

from models import VisionTransformer

from .configs import ExperimentDefinition


class MLPClassifier(nn.Module):
    """Parameter-scale-matched MLP for 64x64 grayscale inputs."""

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

    def forward(self, images):
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

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(
    definition: ExperimentDefinition,
    image_size: int = 64,
    num_classes: int = 10,
) -> nn.Module:
    if definition.model_type == "mlp":
        return MLPClassifier(image_size=image_size, num_classes=num_classes)
    if definition.model_type == "cnn":
        return CNNClassifier(num_classes=num_classes)
    if definition.model_type == "vit":
        if definition.patch_size is None or definition.patch_embedding is None:
            raise ValueError("ViT definitions require patch size and embedding type.")
        return VisionTransformer(
            image_size=image_size,
            patch_size=definition.patch_size,
            in_channels=1,
            num_classes=num_classes,
            embed_dim=128,
            depth=4,
            num_heads=4,
            mlp_dim=512,
            dropout=0.1,
            patch_embedding=definition.patch_embedding,
        )
    raise ValueError(f"Unsupported model type: {definition.model_type}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
