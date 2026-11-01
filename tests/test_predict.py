from src import runtime

runtime.configure()
import unittest, tempfile
from pathlib import Path
import numpy as np
import torch
from src.model import SmallFCN
from src.predict import Predictor
from src.utils import save_checkpoint, load_checkpoint, flatten_checkpoint


class PredictionTests(unittest.TestCase):
    def test_packed_checkpoint_and_compiled_configuration(self):
        model = SmallFCN(width_mult=0.5)
        state = model.state_dict()
        name = "e1.0.weight"
        weight = state.pop(name)
        scale = weight.abs().amax((1, 2, 3), keepdim=True) / 127
        values = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        saved = dict(
            state_dict=state,
            architecture=dict(width_mult=0.5),
            quantized={name: dict(values=values, scale=scale)},
            inference=dict(input_size=128, flip=True, logit_bias=0.2, torchscript=True),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "packed.pt"
            torch.save(saved, path)
            loaded = load_checkpoint(SmallFCN(), path, torch.device("cpu"))
            self.assertTrue(
                torch.equal(loaded.state_dict()[name], values.float() * scale)
            )
            self.assertEqual(loaded.input_size, 128)
            flat_path = Path(folder) / "flat.pt"
            torch.save(flatten_checkpoint(saved), flat_path)
            flat_model = load_checkpoint(SmallFCN(), flat_path, torch.device("cpu"))
            for key, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, flat_model.state_dict()[key]), key)
            predictor = Predictor(flat_path)
            self.assertTrue(predictor.flip)
            self.assertEqual(predictor.logit_bias, 0.2)
            mask = predictor(np.zeros((256, 256, 3), np.uint8))
            self.assertEqual(mask.shape, (256, 256))
            self.assertEqual(mask.dtype, np.uint8)

    def test_saved_model_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            save_checkpoint(SmallFCN(), path)
            predictor = Predictor(path)
            mask = predictor(np.zeros((256, 256, 3), np.uint8))
            self.assertEqual(mask.shape, (256, 256))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)) <= {0, 1})
            with self.assertRaises(ValueError):
                predictor(np.zeros((128, 128, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
