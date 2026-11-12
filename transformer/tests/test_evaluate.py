"""Exercise the Python API without requiring checkpoints or physics dependencies."""
import contextlib
import importlib.util
import io
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / "data/data_multiple"
        data.mkdir(parents=True)
        pairs = [(2, n) for n in (120, 128, 132, 144, 272)]
        pairs += [(3, n) for n in (99, 107, 115, 119, 243)]
        pairs += [(4, n) for n in (98, 118, 122, 130, 142)]
        pairs += [(5, n) for n in (89, 149, 153, 165, 213)]
        with (data / "geometry_log_degree_10.p").open("wb") as f:
            pickle.dump({f"{a} {b}": {"model_ID": 0} for a, b in pairs}, f)
        for h12 in (2, 3, 4, 5):
            np.save(data / f"train_data_raw_test_{h12}.npy", np.zeros((1, 2)))
        self.samples = [[np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))] for _ in pairs]
        self.validator = Mock(side_effect=lambda f, m, t, idx, model, out, *args:
                              (f[:1], m[:1] + 2j, t[:1] + 3j, idx[:1],
                               dict(out, valid=1, valid_unique=1)))
        utilities = SimpleNamespace(
            load_config=Mock(return_value={"train_data_flux": "train_data_test.npy", "base": 10, "model": "decoder"}),
            load_model_from_config=Mock(return_value=(None, SimpleNamespace(max_len_out=80), None)),
            process_samples=Mock(return_value=self.samples),
            convert_samples_to_data=Mock(side_effect=lambda parts, *args: tuple(parts)),
            validate_vacua=self.validator, to_old_a_matrix=lambda m: m,
        )
        jax = SimpleNamespace(numpy=np, config=SimpleNamespace(update=Mock()),
                              clear_caches=Mock(), block_until_ready=Mock())
        modules = {
            "jax": jax, "jax.numpy": np,
            "jaxvacua": SimpleNamespace(FluxVacuaFinder=Mock()),
            "tqdm.auto": SimpleNamespace(tqdm=lambda xs: xs),
            "jax_transformer.utils": SimpleNamespace(sample=Mock()),
            "evaluation_scripts.evaluation_utils": utilities,
            "evaluation_scripts.output_utils": SimpleNamespace(print_console_output=Mock()),
            "data_scripts.tokenizer_utils": SimpleNamespace(int_to_token=lambda prompt, *a, **kw: (prompt, None)),
        }
        path = Path(__file__).resolve().parents[1] / "evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluate_under_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch("argparse.ArgumentParser.parse_args", side_effect=AssertionError("Parsed on import")):
            spec.loader.exec_module(self.module)
        utilities.load_config.assert_not_called()

    def evaluate(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.module.evaluate(38, "checkpoint_1", cwd=self.root, **kwargs)

    def test_sample_without_file(self):
        self.assertIs(self.evaluate(only_sample=True), self.samples)
        self.validator.assert_not_called()
        self.assertFalse((self.root / "data/samples").exists())

    def test_save_and_evaluate(self):
        self.evaluate(only_sample=True, samples_file="saved")
        result = self.evaluate(only_evaluate=True, samples_file="saved", single=True)
        self.assertEqual(len(result), 20)
        with self.assertRaises(FileExistsError):
            self.evaluate(only_sample=True, samples_file="saved")

    def test_validated_results_and_conditions(self):
        for condition in ({"single": True}, {"nflux": 0}, {"w0": 1.1}, {"gs_inv": 4.0}, {}):
            with self.subTest(condition=condition):
                result = self.evaluate(**condition)
                self.assertEqual(len(result), 20 if condition else 100)
                self.assertEqual(result[0]["geometry"], (2, 120))
                np.testing.assert_array_equal(result[0]["moduli"], self.samples[0][1][:1] + 2j)
                np.testing.assert_array_equal(result[0]["sample_indices"], [0])
                self.assertEqual(result[-1]["prompt"][:2], (5, 213))

    def test_no_valid_samples(self):
        self.validator.side_effect = lambda *args: (None, None, None, None, {"valid": 0, "valid_unique": 0})
        result = self.evaluate(single=True)
        self.assertTrue(all(row["fluxes"] is None for row in result))

    def test_incompatible_modes(self):
        with self.assertRaises(ValueError):
            self.evaluate(only_sample=True, only_evaluate=True, samples_file="saved")

    def test_missing_processed_samples(self):
        self.samples[0] = None
        result = self.evaluate(single=True)
        self.assertIsNone(result[0]["fluxes"])
        self.assertIsNone(result[0]["moduli"])
        self.assertIsNone(result[0]["sample_indices"])


if __name__ == "__main__":
    unittest.main()
