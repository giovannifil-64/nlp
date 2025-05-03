import json
import os
import responses
import tempfile
import unittest

from src.dataset import StereoSetDataset


class TestStereoSetDataset(unittest.TestCase):
    """Tests for the StereoSetDataset class."""

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory for testing."""
        cls.temp_dir = tempfile.mkdtemp()

    def setUp(self):
        """Set up a clean dataset instance for each test."""
        self.dataset = StereoSetDataset(cache_dir=self.temp_dir)

    @responses.activate
    def test_download_dev_dataset(self):
        """Test downloading the dev dataset."""
        mock_data = {"version": "1.0", "data": {"intrasentence": []}}
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json",
            json=mock_data,
            status=200,
        )

        result = self.dataset.download_dataset(split="dev")
        self.assertEqual(result, mock_data)

        # Check if file was cached
        cache_path = os.path.join(self.temp_dir, "stereoset_dev.json")
        self.assertTrue(os.path.exists(cache_path))

        # Check cache content
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        self.assertEqual(cached_data, mock_data)

    @responses.activate
    def test_download_test_dataset(self):
        """Test downloading the test dataset."""
        mock_data = {"version": "1.0-test", "data": {"intersentence": []}}
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/HUFS-NLP/CL_Polarizer/refs/heads/main/Benchmarking/benchmark/intrinsic/stereoset/test.json",
            json=mock_data,
            status=200,
        )

        result = self.dataset.download_dataset(split="test")
        self.assertEqual(result, mock_data)

        # Check if file was cached
        cache_path = os.path.join(self.temp_dir, "cl_polarizer_test.json")
        self.assertTrue(os.path.exists(cache_path))

        # Check cache content
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        self.assertEqual(cached_data, mock_data)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after tests."""
        import shutil

        shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    unittest.main()
