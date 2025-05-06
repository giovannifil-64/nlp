import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

from functools import lru_cache
from tqdm import tqdm


class BiasEvaluator:
    """Class for evaluating social bias in language models using the StereoSet benchmark."""

    def __init__(self, model, tokenizer, device="mps", batch_size=8, random_seed=42):
        """
        Initialize the bias evaluator.

        Parameters
        ----------
        model : obj
            The pre-trained transformer model
        tokenizer : obj
            Tokenizer for the model
        device : str, optional
            Device to run inference on, by default "mps"
        batch_size : int, optional
            Batch size for processing multiple sentences at once, by default 8
        random_seed : int, optional
            Random seed for reproducibility, by default 42
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}
        self.batch_size = batch_size
        self.random_seed = random_seed

        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    @lru_cache(maxsize=1024)
    def _cached_sentence_score(self, sentence):
        """
        Cached version of sentence scoring for avoiding redundant computations.

        Parameters
        ----------
        sentence : str
            The sentence to score

        Returns
        -------
        float
            Log probability score
        """
        tokens = self.tokenizer(sentence, return_tensors="pt").to(self.device)

        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        labels = input_ids.clone()

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs.loss.item()
            score = -loss
            return score
        
        except Exception as e:
            print(f"Error evaluating sentence: {e}")
            return 0.0

    def evaluate_sentence_score(self, sentence):
        """
        Get the likelihood score for a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to score

        Returns
        -------
        float
            Log probability score
        """
        return self._cached_sentence_score(sentence)

    def evaluate_batch(self, sentences):
        """
        Process multiple sentences in a batch for efficiency.

        Parameters
        ----------
        sentences : list of str
            List of sentences to score

        Returns
        -------
        list of float
            List of scores corresponding to the input sentences
        """
        scores = []
        
        for sentence in sentences:
            scores.append(self.evaluate_sentence_score(sentence))
        
        return scores

    def evaluate_bias(self, dataset):
        """
        Evaluate bias across the dataset.

        Parameters
        ----------
        dataset : dict
            Preprocessed dataset with categories

        Returns
        -------
        dict
            Evaluation results
        """
        start_time = time.time()

        results = {
            "overall": {
                "ss_score": 0.0,
                "stereotype_score": 0.0,
                "anti_stereotype_score": 0.0,
                "count": 0,
            }
        }

        for category in dataset.keys():
            results[category] = {
                "ss_score": 0.0,
                "stereotype_score": 0.0,
                "anti_stereotype_score": 0.0,
                "count": 0,
            }

        print("Evaluating bias...")
        total_examples = sum(len(examples) for examples in dataset.values())
        
        print(f"Total examples to evaluate: {total_examples}")

        for category, examples in dataset.items():
            print(f"Evaluating {category} bias ({len(examples)} examples)...")

            grouped_examples = {}

            for example in examples:
                if example["id"] not in grouped_examples:
                    grouped_examples[example["id"]] = []
                grouped_examples[example["id"]].append(example)

            batch_stereotype = []
            batch_anti_stereotype = []
            batch_example_ids = []
            batch_categories = []

            with tqdm(
                total=len(grouped_examples), desc=f"Processing {category}"
            ) as pbar:
                for example_id, example_set in grouped_examples.items():
                    if len(example_set) < 2:
                        pbar.update(1)
                        continue

                    stereotype_ex = None
                    anti_stereotype_ex = None

                    for ex in example_set:
                        if ex["label"] == "stereotype":
                            stereotype_ex = ex
                        elif ex["label"] == "anti-stereotype":
                            anti_stereotype_ex = ex

                    if not stereotype_ex or not anti_stereotype_ex:
                        pbar.update(1)
                        continue

                    batch_stereotype.append(stereotype_ex["sentence"])
                    batch_anti_stereotype.append(anti_stereotype_ex["sentence"])
                    batch_example_ids.append(example_id)
                    batch_categories.append(category)

                    if len(batch_stereotype) >= self.batch_size:
                        self._process_batch(
                            batch_stereotype,
                            batch_anti_stereotype,
                            batch_categories,
                            results,
                        )

                        batch_stereotype = []
                        batch_anti_stereotype = []
                        batch_example_ids = []
                        batch_categories = []

                        pbar.update(self.batch_size)

                        if "cuda" in self.device:
                            torch.cuda.empty_cache()

                        if "mps" in self.device:
                            torch.mps.empty_cache()

                if batch_stereotype:
                    self._process_batch(
                        batch_stereotype,
                        batch_anti_stereotype,
                        batch_categories,
                        results,
                    )
                    pbar.update(len(batch_stereotype))

        for category in results:
            if results[category]["count"] > 0:
                results[category]["stereotype_score"] /= results[category]["count"]
                results[category]["anti_stereotype_score"] /= results[category]["count"]
                results[category]["ss_score"] /= results[category]["count"]

        elapsed_time = time.time() - start_time
        
        print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
        print(f"Processed {results['overall']['count']} example pairs")

        self.results = results
        
        return results

    def _process_batch(
        self, batch_stereotype, batch_anti_stereotype, batch_categories, results
    ):
        """
        Process a batch of examples.

        Parameters
        ----------
        batch_stereotype : list of str
            List of stereotype sentences
        batch_anti_stereotype : list of str
            List of anti-stereotype sentences
        batch_categories : list of str
            List of categories for each example
        results : dict
            Results dictionary to update
        """
        try:
            stereotype_scores = self.evaluate_batch(batch_stereotype)

            anti_stereotype_scores = self.evaluate_batch(batch_anti_stereotype)

            for i in range(len(batch_stereotype)):
                category = batch_categories[i]
                stereotype_score = stereotype_scores[i]
                anti_stereotype_score = anti_stereotype_scores[i]

                try:
                    # Calculate SS Score: P(stereotype) / (P(stereotype) + P(anti-stereotype))
                    # Handle division by zero and other numerical issues
                    if stereotype_score + anti_stereotype_score == 0:
                        ss_score = 0.5  # Neutral if both scores are zero
                    else:
                        ss_score = stereotype_score / (stereotype_score + anti_stereotype_score)

                    ss_score = max(0.0, min(1.0, ss_score))

                    results[category]["stereotype_score"] += stereotype_score
                    results[category]["anti_stereotype_score"] += anti_stereotype_score
                    results[category]["ss_score"] += ss_score
                    results[category]["count"] += 1

                    results["overall"]["stereotype_score"] += stereotype_score
                    results["overall"]["anti_stereotype_score"] += anti_stereotype_score
                    results["overall"]["ss_score"] += ss_score
                    results["overall"]["count"] += 1

                except Exception as e:
                    print(f"Error processing results: {e}")
                    continue

        except Exception as e:
            print(f"Error in batch processing: {e}")

    def save_results(self, save_path="results", filename=None):
        """
        Save evaluation results to a file.

        Parameters
        ----------
        save_path : str, optional
            Directory to save results, by default "results"
        filename : str, optional
            Optional filename, by default None

        Returns
        -------
        str
            Path to saved file

        Raises
        ------
        ValueError
            If no results are available
        """
        if not self.results:
            raise ValueError("No results to save. Run evaluate_bias() first.")

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if filename is None:
                model_name = self.model.__class__.__name__
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{model_name}_bias_evaluation_{timestamp}.json"

            file_path = os.path.join(save_path, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)

            print(f"Results saved to {file_path}")
            return file_path

        except Exception as e:
            print(f"Error saving results: {e}")
            return None

    def visualize_results(self, save_path="results", show=True):
        """
        Visualize the bias evaluation results.

        Parameters
        ----------
        save_path : str, optional
            Directory to save visualization, by default "results"
        show : bool, optional
            Whether to display the visualization, by default True

        Returns
        -------
        str
            Path to saved visualization

        Raises
        ------
        ValueError
            If no results are available
        """
        if not self.results:
            raise ValueError("No results to visualize. Run evaluate_bias() first.")

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            categories = [cat for cat in self.results.keys() if cat != "overall"]
            ss_scores = [self.results[cat]["ss_score"] for cat in categories]
            stereotype_scores = [self.results[cat]["stereotype_score"] for cat in categories]
            anti_stereotype_scores = [self.results[cat]["anti_stereotype_score"] for cat in categories]

            categories.append("overall")
            ss_scores.append(self.results["overall"]["ss_score"])
            stereotype_scores.append(self.results["overall"]["stereotype_score"])
            anti_stereotype_scores.append(self.results["overall"]["anti_stereotype_score"])

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

            bars = ax1.bar(categories, ss_scores, color="skyblue")
            ax1.set_title("Stereotype Score (SS)")
            ax1.set_xlabel("Bias Category")
            ax1.set_ylabel("SS Score")
            ax1.axhline(y=0.5, color="r", linestyle="--", label="Neutral (0.5)")
            ax1.legend()
            ax1.grid(axis="y", linestyle="--", alpha=0.7)

            for i, bar in enumerate(bars):
                count = self.results[categories[i]]["count"]
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            x = np.arange(len(categories))
            width = 0.35
            ax2.bar(x - width / 2, stereotype_scores, width, label="Stereotype")
            ax2.bar(x + width / 2, anti_stereotype_scores, width, label="Anti-Stereotype")
            ax2.set_title("Raw Scores")
            ax2.set_xlabel("Bias Category")
            ax2.set_ylabel("Score")
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(axis="y", linestyle="--", alpha=0.7)

            bias_severity = [abs(score - 0.5) for score in ss_scores]
            bars = ax3.bar(categories, bias_severity, color="salmon")
            ax3.set_title("Bias Severity (|SS - 0.5|)")
            ax3.set_xlabel("Bias Category")
            ax3.set_ylabel("Severity")
            ax3.grid(axis="y", linestyle="--", alpha=0.7)

            for i, bar in enumerate(bars):
                direction = "↑" if ss_scores[i] > 0.5 else "↓"
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    direction,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="green" if direction == "↓" else "red",
                )

            fig.tight_layout()

            model_name = self.model.__class__.__name__
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(save_path, f"{model_name}_bias_visualization_{timestamp}.png")
            fig.savefig(file_path, dpi=300)

            if show:
                fig.show()
            else:
                plt.close(fig)

            print(f"Visualization saved to {file_path}")
            return file_path
 
        except Exception as e:
            print(f"Error visualizing results: {e}")
            return None
