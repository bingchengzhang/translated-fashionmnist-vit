from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from translated_fashionmnist import TranslatedFashionMNIST, VisionTransformer


class DummyFashionMNIST(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        image = torch.zeros(1, 28, 28)
        image[:, 4:24, 6:22] = (index + 1) / 8
        return image, index % 10


class DatasetTests(unittest.TestCase):
    def test_random_mode_is_deterministic(self):
        first = TranslatedFashionMNIST(
            DummyFashionMNIST(),
            canvas_size=64,
            mode="A",
            seed=7,
            return_position=True,
        )
        second = TranslatedFashionMNIST(
            DummyFashionMNIST(),
            canvas_size=64,
            mode="A",
            seed=7,
            return_position=True,
        )
        image_a, label_a, position_a = first[3]
        image_b, label_b, position_b = second[3]
        self.assertEqual(tuple(image_a.shape), (1, 64, 64))
        self.assertTrue(torch.equal(image_a, image_b))
        self.assertTrue(torch.equal(position_a, position_b))
        self.assertEqual(int(label_a), int(label_b))

    def test_center_mode_position(self):
        dataset = TranslatedFashionMNIST(
            DummyFashionMNIST(),
            canvas_size=64,
            mode="B",
            return_position=True,
        )
        _, _, position = dataset[0]
        self.assertEqual(position.tolist(), [18, 18])


class ModelTests(unittest.TestCase):
    def test_forward_and_backward(self):
        model = VisionTransformer(
            image_size=64,
            patch_size=8,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_dim=64,
            dropout=0.0,
        )
        images = torch.rand(4, 1, 64, 64)
        labels = torch.tensor([0, 1, 2, 3])
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (4, 10))
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_linear_patch_embedding(self):
        model = VisionTransformer(
            image_size=64,
            patch_size=8,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_dim=64,
            patch_embedding="linear",
        )
        logits = model(torch.rand(2, 1, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
