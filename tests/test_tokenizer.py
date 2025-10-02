import shutil
import tempfile
import unittest
from pathlib import Path

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmf_ai.dataset import build_layout_dataset
from vmf_ai.model import GeneratorConfig, VMFBrushGenerator
from vmf_ai.tokenizer import MaterialVocabulary
from vmf_tools import VMFBuilder, Vector3


class TestMaterialVocabulary(unittest.TestCase):
    def test_encoding_and_decoding(self) -> None:
        vocab = MaterialVocabulary(["DEV/A", "DEV/B", "DEV/A"])
        self.assertGreaterEqual(len(vocab), 3)
        encoded = vocab.encode("DEV/A")
        self.assertNotEqual(encoded, vocab.pad_id)
        self.assertEqual(vocab.decode(encoded), "DEV/A")
        self.assertEqual(vocab.decode(vocab.encode("missing")), vocab.unk_token)


class TestLayoutDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        builder = VMFBuilder()
        builder.add_axis_aligned_block(Vector3(-64, -64, 0), Vector3(64, 64, 128), material="DEV/A")
        builder.add_axis_aligned_block(Vector3(128, -64, 0), Vector3(256, 64, 128), material="DEV/B")
        builder.save(str(self.tmpdir / "sample.vmf"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_dataset_shapes_and_normalisation(self) -> None:
        dataset, vocab = build_layout_dataset(self.tmpdir)
        self.assertEqual(dataset.feature_dim, 6)
        self.assertEqual(dataset.max_brushes, 2)
        sample = dataset[0]
        self.assertEqual(sample["features"].shape, (2, 6))
        self.assertTrue(sample["mask"].all())
        self.assertEqual(sample["materials"].shape, (2,))
        self.assertNotEqual(sample["materials"][0].item(), vocab.pad_id)

        # Denormalisation should invert the stored features.
        restored = dataset.denormalise(sample["features"]) * sample["mask"].unsqueeze(-1)
        self.assertFalse(torch.isnan(restored).any())

    def test_model_forward_shapes(self) -> None:
        dataset, vocab = build_layout_dataset(self.tmpdir)
        sample = dataset[0]
        config = GeneratorConfig(
            max_brushes=dataset.max_brushes,
            feature_dim=dataset.feature_dim,
            material_vocab_size=len(vocab),
        )
        model = VMFBrushGenerator(config)
        batch = {
            "features": sample["features"].unsqueeze(0),
            "materials": sample["materials"].unsqueeze(0),
            "mask": sample["mask"].unsqueeze(0),
        }
        outputs = model(**batch)
        self.assertEqual(outputs["feature_pred"].shape, (1, dataset.max_brushes, dataset.feature_dim))
        self.assertEqual(outputs["material_logits"].shape, (1, dataset.max_brushes, len(vocab)))
        self.assertEqual(outputs["presence_logits"].shape, (1, dataset.max_brushes))


if __name__ == "__main__":
    unittest.main()
