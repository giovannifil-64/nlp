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
        # Use temporary directory for test outputs
        cls.temp_dir = tempfile.mkdtemp()

        # Determine device for testing
        if torch.cuda.is_available():
            cls.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls.device = "mps"
        else:
            cls.device = "cpu"

        # Create minimal test results
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

        # Create dummy evaluation results for two models
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
        # Remove temp directory
        shutil.rmtree(cls.temp_dir)

    def test_calculate_additional_metrics(self):
        """Test that additional metrics are calculated correctly."""
        # Calculate additional metrics
        results = calculate_additional_metrics(self.test_results.copy())

        # Check that all categories have the new metrics
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
        # Add additional metrics first
        results = calculate_additional_metrics(self.test_results.copy())

        # Generate report
        report_file = generate_bias_report(
            results, "test-model", save_path=self.temp_dir
        )

        # Check that report file exists
        self.assertTrue(os.path.exists(report_file))

        # Check contents of report
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
        # First add additional metrics to both models' results
        for i in range(len(self.model_evaluation_results)):
            self.model_evaluation_results[i]["results"] = calculate_additional_metrics(
                self.model_evaluation_results[i]["results"].copy()
            )

        # Generate comparison
        comparison_file = compare_models(
            self.model_evaluation_results, output_dir=self.temp_dir, show_plots=False
        )

        # Check that comparison file exists
        self.assertTrue(os.path.exists(comparison_file))

        # Check contents of comparison
        with open(comparison_file, "r", encoding="utf-8") as f:
            comparison_content = f.read()

            # Check that it contains expected sections
            self.assertIn("# Model Comparison: Bias Evaluation", comparison_content)
            self.assertIn("## Overall Comparison", comparison_content)
            self.assertIn("## Key Findings", comparison_content)

            # Check that it contains both model names
            self.assertIn("distilbert-base-uncased", comparison_content)
            self.assertIn("roberta-base", comparison_content)


if __name__ == "__main__":
    unittest.main()
