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

import os
import shutil
import torch
import tempfile
import unittest

from src.evaluate_models import (
    calculate_additional_metrics,
    generate_bias_report,
    compare_models,
)


class TestComprehensiveEvaluation(unittest.TestCase):
    """Test the comprehensive evaluation functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment with minimal test data."""
        cls.temp_dir = tempfile.mkdtemp()

        if torch.cuda.is_available():
            cls.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls.device = "mps"
        else:
            cls.device = "cpu"

        cls.test_results = {
            "gender": {
                "ss_score": 0.6,
                "stereotype_score": 0.8,
                "anti_stereotype_score": 0.5,
                "count": 10,
            },
            "profession": {
                "ss_score": 0.55,
                "stereotype_score": 0.7,
                "anti_stereotype_score": 0.6,
                "count": 15,
            },
            "race": {
                "ss_score": 0.45,
                "stereotype_score": 0.6,
                "anti_stereotype_score": 0.7,
                "count": 20,
            },
            "religion": {
                "ss_score": 0.52,
                "stereotype_score": 0.65,
                "anti_stereotype_score": 0.6,
                "count": 12,
            },
            "overall": {
                "ss_score": 0.53,
                "stereotype_score": 0.69,
                "anti_stereotype_score": 0.61,
                "count": 57,
            },
        }

        cls.model_evaluation_results = [
            {"model_name": "distilbert-base-uncased", "results": cls.test_results},
            {
                "model_name": "roberta-base",
                "results": {
                    "gender": {
                        "ss_score": 0.58,
                        "stereotype_score": 0.75,
                        "anti_stereotype_score": 0.55,
                        "count": 10,
                    },
                    "profession": {
                        "ss_score": 0.52,
                        "stereotype_score": 0.65,
                        "anti_stereotype_score": 0.58,
                        "count": 15,
                    },
                    "race": {
                        "ss_score": 0.48,
                        "stereotype_score": 0.62,
                        "anti_stereotype_score": 0.68,
                        "count": 20,
                    },
                    "religion": {
                        "ss_score": 0.51,
                        "stereotype_score": 0.63,
                        "anti_stereotype_score": 0.62,
                        "count": 12,
                    },
                    "overall": {
                        "ss_score": 0.51,
                        "stereotype_score": 0.66,
                        "anti_stereotype_score": 0.63,
                        "count": 57,
                    },
                },
            },
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        shutil.rmtree(cls.temp_dir)

    def test_calculate_additional_metrics(self):
        """Test that additional metrics are calculated correctly."""
        results = calculate_additional_metrics(self.test_results.copy())

        for category in results:
            self.assertIn("bias_difference", results[category])
            self.assertIn("bias_ratio", results[category])
            self.assertIn("bias_severity", results[category])
            self.assertIn("bias_direction", results[category])

        # Check specific values for one category
        self.assertAlmostEqual(results["gender"]["bias_difference"], 0.3)
        self.assertAlmostEqual(results["gender"]["bias_ratio"], 1.6)
        self.assertAlmostEqual(results["gender"]["bias_severity"], 0.1)
        self.assertEqual(results["gender"]["bias_direction"], 1)  # Stereotype bias

        # Check race has anti-stereotype bias (direction -1)
        self.assertEqual(results["race"]["bias_direction"], -1)

    def test_generate_bias_report(self):
        """Test that bias report is generated correctly."""
        results = calculate_additional_metrics(self.test_results.copy())

        report_file = generate_bias_report(
            results, "test-model", save_path=self.temp_dir
        )

        self.assertTrue(os.path.exists(report_file))

        with open(report_file, "r", encoding="utf-8") as f:
            report_content = f.read()

            # Check that it contains expected sections
            self.assertIn("# Bias Evaluation Report: test-model", report_content)
            self.assertIn("## Overall Summary", report_content)
            self.assertIn("## Category Breakdown", report_content)
            self.assertIn("## Detailed Metrics", report_content)
            self.assertIn("## Interpretation", report_content)

            # Check that it contains specific metrics
            self.assertIn("SS Score", report_content)
            self.assertIn("Bias Severity", report_content)
            self.assertIn("Bias Direction", report_content)

    def test_compare_models(self):
        """Test that model comparison works correctly."""
        for i in range(len(self.model_evaluation_results)):
            self.model_evaluation_results[i]["results"] = calculate_additional_metrics(
                self.model_evaluation_results[i]["results"].copy()
            )

        comparison_result = compare_models(
            self.model_evaluation_results, output_dir=self.temp_dir, show_plots=False
        )

        self.assertIsInstance(comparison_result, dict)
        self.assertIn("report_file", comparison_result)
        self.assertIn("visualizations", comparison_result)
        
        report_file = comparison_result["report_file"]
        self.assertTrue(os.path.exists(report_file))

        with open(report_file, "r", encoding="utf-8") as f:
            comparison_content = f.read()

            self.assertIn("# Model Comparison: Bias Evaluation", comparison_content)
            self.assertIn("## Overall Comparison", comparison_content)
            self.assertIn("## Key Findings", comparison_content)

            self.assertIn("distilbert-base-uncased", comparison_content)
            self.assertIn("roberta-base", comparison_content)
            
        self.assertIsInstance(comparison_result["visualizations"], dict)
        for viz_name, viz_path in comparison_result["visualizations"].items():
            self.assertTrue(os.path.exists(viz_path), f"Visualization file {viz_name} not found at {viz_path}")


if __name__ == "__main__":
    unittest.main()
