from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from translated_fashionmnist.experiments.config import (
    EXPERIMENTS,
    GROUPS,
    SETTING_ORDER,
    configuration_ids_for_groups,
)
from translated_fashionmnist.experiments.run import (
    consolidate_training_history,
    create_parser,
)
from translated_fashionmnist.experiments.protocol import (
    ProtocolConfig,
    _can_resume,
    _expected_metadata,
    build_model,
)
from translated_fashionmnist.models import (
    ConvPatchEmbedding,
    LinearPatchEmbedding,
    count_trainable_parameters,
)
from translated_fashionmnist.utils import write_csv


class ExperimentTests(unittest.TestCase):
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
        self.assertEqual(GROUPS["model"], ("mlp", "cnn", "vit_p8_conv"))

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

    def test_invalid_protocol_values_are_rejected(self):
        with self.assertRaises(ValueError):
            ProtocolConfig(epochs=0)
        with self.assertRaises(ValueError):
            ProtocolConfig(val_fraction=1.0)
        with self.assertRaises(ValueError):
            ProtocolConfig(limit_test_samples=-1)

    def test_resume_requires_complete_result_record(self):
        config = ProtocolConfig()
        expected = _expected_metadata(EXPERIMENTS["mlp"], config, "A")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metadata.json").write_text(
                json.dumps({"experiment": expected}),
                encoding="utf-8",
            )
            torch.save({"model_state": {}}, run_dir / "best.pt")
            self.assertFalse(_can_resume(run_dir, expected))

            (run_dir / "training_result.json").write_text(
                json.dumps(
                    {
                        "best_epoch": 1,
                        "best_val_accuracy": 0.5,
                        "elapsed_seconds": 1.0,
                        "parameter_count": 10,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(_can_resume(run_dir, expected))

            write_csv(
                [
                    {
                        "epoch": epoch,
                        "learning_rate": 1e-3,
                        "train_loss": 1.0,
                        "train_accuracy": 0.5,
                        "val_loss": 0.9,
                        "val_accuracy": 0.6,
                    }
                    for epoch in range(1, config.epochs + 1)
                ],
                run_dir / "history.csv",
            )
            self.assertTrue(_can_resume(run_dir, expected))

    def test_invalid_model_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            ConvPatchEmbedding(64, 0, 1, 128)
        with self.assertRaises(ValueError):
            LinearPatchEmbedding(64, 7, 1, 128)

    def test_runner_defaults_to_ignored_output_directory(self):
        arguments = create_parser().parse_args([])
        self.assertEqual(arguments.output_dir, "outputs/comparison")
        self.assertEqual(ProtocolConfig().output_dir, arguments.output_dir)

    def test_runner_consolidates_per_fit_histories(self):
        fields = {
            "learning_rate": 1e-3,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 0.9,
            "val_accuracy": 0.6,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("A", "B"):
                write_csv(
                    [
                        {"epoch": epoch, **fields}
                        for epoch in (1, 2)
                    ],
                    root / "runs" / "mlp" / f"train_{mode}" / "history.csv",
                )
            rows = consolidate_training_history(root, ["mlp"], epochs=2)
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {(row["config_id"], row["train_mode"]) for row in rows},
                {("mlp", "A"), ("mlp", "B")},
            )
            self.assertTrue((root / "training_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
