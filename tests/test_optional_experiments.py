from __future__ import annotations

import unittest

import torch

from models.vit import ConvPatchEmbedding, LinearPatchEmbedding
from optional_experiments_vs_teammate.configurations import (
    EXPERIMENTS,
    GROUPS,
    SETTING_ORDER,
    configuration_ids_for_groups,
)
from optional_experiments_vs_teammate.models import (
    build_model,
    count_trainable_parameters,
)


class OptionalExperimentTests(unittest.TestCase):
    def test_every_model_produces_ten_logits(self):
        images = torch.rand(2, 1, 64, 64)
        for definition in EXPERIMENTS.values():
            with self.subTest(config_id=definition.config_id):
                model = build_model(definition)
                logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertGreater(count_trainable_parameters(model), 0)

    def test_group_selection_is_unique_and_ordered(self):
        selected = configuration_ids_for_groups(["all"])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(selected, list(EXPERIMENTS))
        for members in GROUPS.values():
            self.assertTrue(set(members).issubset(EXPERIMENTS))

    def test_setting_order_matches_assignment(self):
        self.assertEqual(
            SETTING_ORDER,
            (
                (1, "A", "A"),
                (2, "B", "B"),
                (3, "A", "B"),
                (4, "B", "A"),
            ),
        )

    def test_conv_and_linear_patch_projection_are_equivalent(self):
        torch.manual_seed(7)
        conv = ConvPatchEmbedding(64, 16, 1, 128)
        torch.manual_seed(7)
        linear = LinearPatchEmbedding(64, 16, 1, 128)
        images = torch.rand(2, 1, 64, 64)
        self.assertTrue(
            torch.allclose(conv(images), linear(images), atol=1e-6, rtol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
