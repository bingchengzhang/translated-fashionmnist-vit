"""Experiment definitions shared by the runners and visualizations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentDefinition:
    config_id: str
    display_name: str
    model_type: str
    patch_size: int | None = None
    patch_embedding: str | None = None


EXPERIMENTS: dict[str, ExperimentDefinition] = {
    "mlp": ExperimentDefinition(
        config_id="mlp",
        display_name="MLP",
        model_type="mlp",
    ),
    "cnn": ExperimentDefinition(
        config_id="cnn",
        display_name="CNN",
        model_type="cnn",
    ),
    "vit_p16_conv": ExperimentDefinition(
        config_id="vit_p16_conv",
        display_name="ViT (patch 16)",
        model_type="vit",
        patch_size=16,
        patch_embedding="conv",
    ),
    "vit_p8_conv": ExperimentDefinition(
        config_id="vit_p8_conv",
        display_name="ViT (patch 8)",
        model_type="vit",
        patch_size=8,
        patch_embedding="conv",
    ),
    "vit_p4_conv": ExperimentDefinition(
        config_id="vit_p4_conv",
        display_name="ViT (patch 4)",
        model_type="vit",
        patch_size=4,
        patch_embedding="conv",
    ),
    "vit_p16_linear": ExperimentDefinition(
        config_id="vit_p16_linear",
        display_name="ViT (Flatten+Linear)",
        model_type="vit",
        patch_size=16,
        patch_embedding="linear",
    ),
}


GROUPS: dict[str, tuple[str, ...]] = {
    "model": ("mlp", "cnn", "vit_p16_conv"),
    "patch_size": ("vit_p4_conv", "vit_p8_conv", "vit_p16_conv"),
    "patch_embedding": ("vit_p16_conv", "vit_p16_linear"),
}


SETTING_ORDER: tuple[tuple[int, str, str], ...] = (
    (1, "A", "A"),
    (2, "B", "B"),
    (3, "A", "B"),
    (4, "B", "A"),
)


def configuration_ids_for_groups(groups: list[str]) -> list[str]:
    selected: set[str] = set()
    if "all" in groups:
        groups = list(GROUPS)
    for group in groups:
        selected.update(GROUPS[group])
    return [config_id for config_id in EXPERIMENTS if config_id in selected]


def groups_for_configuration(config_id: str) -> list[str]:
    return [group for group, members in GROUPS.items() if config_id in members]
