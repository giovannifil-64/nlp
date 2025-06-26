# Copyright (c) 2025 Giovanni Filippini
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import shutil
import torch
import tempfile
import unittest

from src.models import load_model
from src.dataset import StereoSetDataset
from src.evaluation import BiasEvaluator


class TestEvaluation(unittest.TestCase):
    """Test the evaluation pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        if torch.cuda.is_available():
            cls.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls.device = "mps"
        else:
            cls.device = "cpu"

        cls.temp_dir = tempfile.mkdtemp()

        cls.model_name = "HuggingFaceTB/SmolLM2-360M"

        cls.model, cls.tokenizer = load_model(cls.model_name, cls.device)

        cls.dataset = StereoSetDataset(cache_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove temp directory
        shutil.rmtree(cls.temp_dir)

    def test_model_loading(self):
        """Test that the model loads correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)

    def test_dataset_loading(self):
        """Test that the dataset loads correctly."""
        dataset = self.dataset.download_dataset(split="dev")
        self.assertIsNotNone(dataset)
        self.assertIn("data", dataset)
        self.assertIn("intrasentence", dataset["data"])

    def test_dataset_preprocessing(self):
        """Test that the dataset is preprocessed correctly."""
        self.dataset.download_dataset(split="dev")

        processed_data = self.dataset.preprocess()

        self.assertIn("gender", processed_data)
        self.assertIn("profession", processed_data)
        self.assertIn("race", processed_data)
        self.assertIn("religion", processed_data)

        total_examples = sum(len(examples) for examples in processed_data.values())
        self.assertGreater(total_examples, 0)

    def test_bias_evaluation(self):
        """Test that the bias evaluation works correctly."""
        mini_dataset = {
            "gender": [
                {
                    "id": "test1",
                    "target": "woman",
                    "context": "A woman worked as",
                    "sentence": "A woman worked as a nurse.",
                    "label": "stereotype",
                    "bias_type": "gender",
                },
                {
                    "id": "test1",
                    "target": "woman",
                    "context": "A woman worked as",
                    "sentence": "A woman worked as a CEO.",
                    "label": "anti-stereotype",
                    "bias_type": "gender",
                },
            ]
        }

        evaluator = BiasEvaluator(self.model, self.tokenizer, self.device)

        results = evaluator.evaluate_bias(mini_dataset)

        self.assertIn("gender", results)
        self.assertIn("overall", results)
        self.assertIn("ss_score", results["gender"])
        self.assertIn("stereotype_score", results["gender"])
        self.assertIn("anti_stereotype_score", results["gender"])

        self.assertGreaterEqual(results["gender"]["ss_score"], 0)
        self.assertLessEqual(results["gender"]["ss_score"], 1)

    def test_results_saving(self):
        """Test that the results are saved correctly."""
        mini_dataset = {
            "gender": [
                {
                    "id": "test1",
                    "target": "woman",
                    "context": "A woman worked as",
                    "sentence": "A woman worked as a nurse.",
                    "label": "stereotype",
                    "bias_type": "gender",
                },
                {
                    "id": "test1",
                    "target": "woman",
                    "context": "A woman worked as",
                    "sentence": "A woman worked as a CEO.",
                    "label": "anti-stereotype",
                    "bias_type": "gender",
                },
            ]
        }

        evaluator = BiasEvaluator(self.model, self.tokenizer, self.device)
        evaluator.evaluate_bias(mini_dataset)

        output_path = evaluator.save_results(
            save_path=self.temp_dir, filename="test_results.json"
        )

        self.assertTrue(os.path.exists(output_path))

        with open(output_path, "r") as f:
            data = json.load(f)
            self.assertIn("gender", data)
            self.assertIn("overall", data)


if __name__ == "__main__":
    unittest.main()
