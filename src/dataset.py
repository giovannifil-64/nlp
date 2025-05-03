import json
import os
import requests

from tqdm import tqdm


class StereoSetDataset:
    """Class for loading and preprocessing the StereoSet dataset."""

    def __init__(self, cache_dir="data"):
        """
        Initialize the StereoSet dataset loader.

        Args:
            cache_dir (str): Directory to cache the dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def download_dataset(self, split="dev"):
        """
        Download the StereoSet dataset if not already cached.

        Args:
            split (str): Dataset split to download.

        Returns:
            dict: The loaded dataset
        """
        file_name = f"stereoset_{split}.json"
        file_path = os.path.join(self.cache_dir, file_name)

        if os.path.exists(file_path):
            print(f"Loading cached dataset from {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        else:
            urls = {
                "dev": "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json",
            }

            if split not in urls:
                raise ValueError(f"Invalid split: {split}. Must be one of {list(urls.keys())}")

            print(f"Downloading StereoSet {split} dataset...")
            response = requests.get(urls[split])
            if response.status_code == 200:
                self.dataset = response.json()

                # Save to cache
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.dataset, f)
                print(f"Dataset saved to {file_path}")
            else:
                raise Exception(f"Failed to download dataset: {response.status_code}")

        return self.dataset

    def preprocess(self):
        """
        Preprocess the dataset for evaluation.

        Returns:
            dict: Processed dataset with categories as keys
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")

        processed_data = {"gender": [], "profession": [], "race": [], "religion": []}

        print("Preprocessing dataset...")

        for item in tqdm(self.dataset["data"]["intrasentence"]):
            bias_type = item["bias_type"]

            if bias_type in processed_data:
                target = item["target"]
                context = item["context"]

                # Process each stereotype, anti-stereotype, and unrelated option
                for option in item["sentences"]:
                    processed_data[bias_type].append(
                        {
                            "id": item["id"],
                            "target": target,
                            "context": context,
                            "sentence": option["sentence"],
                            "label": option["gold_label"],
                            "bias_type": bias_type,
                        }
                    )

        print(f"Processed dataset with {sum(len(items) for items in processed_data.values())} examples")
        return processed_data
