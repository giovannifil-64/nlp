import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from datetime import datetime

from src.models import load_model
from src.dataset import StereoSetDataset
from src.evaluation import BiasEvaluator


def evaluate_model_bias(
    model_name, device, split="test", output_dir="results", show_plots=False
):
    """
    Evaluate a model for social bias using the StereoSet dataset.

    Args:
        model_name (str): Name of the model to evaluate
        device (str): Device to run on (cuda, cpu, mps)
        split (str): Dataset split to use (dev, test)
        output_dir (str): Directory to save results
        show_plots (bool): Whether to display plots

    Returns:
        dict: Evaluation results
    """
    print(f"\n========== Evaluating {model_name} ==========")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(
        output_dir, f"{model_name.replace('/', '_')}_{timestamp}"
    )
    os.makedirs(model_output_dir, exist_ok=True)

    model, tokenizer = load_model(model_name, device)

    print(f"Loading StereoSet dataset ({split} split)...")
    
    dataset = StereoSetDataset()
    dataset.download_dataset(split=split)
    processed_data = dataset.preprocess()

    evaluator = BiasEvaluator(model, tokenizer, device)
    evaluation_results = evaluator.evaluate_bias(processed_data)
    results = calculate_additional_metrics(evaluation_results)

    results_file = evaluator.save_results(save_path=model_output_dir, filename=f"bias_evaluation.json")

    vis_file = evaluator.visualize_results(save_path=model_output_dir, show=show_plots)
    report_file = generate_bias_report(results, model_name, save_path=model_output_dir)

    # Print summary
    print_results_summary(results, model_name)

    return {
        "model_name": model_name,
        "results": results,
        "results_file": results_file,
        "visualization_file": vis_file,
        "report_file": report_file,
    }


def calculate_additional_metrics(results):
    """
    Calculate additional bias metrics from the evaluation results.

    Args:
        results (dict): Evaluation results from BiasEvaluator

    Returns:
        dict: Results with additional metrics
    """
    # Add additional metrics for each category
    for category in results:
        # Skip if empty category
        if results[category]["count"] == 0:
            continue

        # Bias difference: absolute difference between stereotype and anti-stereotype scores
        results[category]["bias_difference"] = abs(
            results[category]["stereotype_score"]
            - results[category]["anti_stereotype_score"]
        )

        # Bias ratio: ratio of stereotype to anti-stereotype scores
        if results[category]["anti_stereotype_score"] > 0:
            results[category]["bias_ratio"] = (
                results[category]["stereotype_score"]
                / results[category]["anti_stereotype_score"]
            )
        else:
            results[category]["bias_ratio"] = float("inf")

        # Bias severity: how far ss_score is from 0.5 (neutral)
        results[category]["bias_severity"] = abs(results[category]["ss_score"] - 0.5)

        # Bias direction: positive for stereotype bias, negative for anti-stereotype bias
        results[category]["bias_direction"] = (
            1 if results[category]["ss_score"] > 0.5 else -1
        )

    return results


def generate_bias_report(results, model_name, save_path="results"):
    """
    Generate a report of bias evaluation results.

    Args:
        results (dict): Evaluation results
        model_name (str): Name of the model
        save_path (str): Directory to save the report

    Returns:
        str: Path to the report file
    """
    report_file = os.path.join(save_path, "bias_report.md")

    with open(report_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Bias Evaluation Report: {model_name}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Write overall summary
        f.write("## Overall Summary\n\n")
        f.write(f"- **SS Score**: {results['overall']['ss_score']:.4f}\n")
        f.write(f"- **Bias Severity**: {results['overall']['bias_severity']:.4f}\n")
        f.write(
            f"- **Bias Direction**: {'Stereotype' if results['overall']['bias_direction'] > 0 else 'Anti-Stereotype'}\n"
        )
        f.write(f"- **Total Examples**: {results['overall']['count']}\n\n")

        # Write category breakdowns
        f.write("## Category Breakdown\n\n")
        f.write(
            "| Category | SS Score | Bias Severity | Bias Direction | Stereotype Score | Anti-Stereotype Score | Examples |\n"
        )
        f.write(
            "| -------- | -------- | ------------- | -------------- | ---------------- | --------------------- | -------- |\n"
        )

        for category in sorted(results.keys()):
            if category != "overall" and results[category]["count"] > 0:
                f.write(
                    f"| {category.capitalize()} | {results[category]['ss_score']:.4f} | "
                )
                f.write(f"{results[category]['bias_severity']:.4f} | ")
                f.write(
                    f"{'Stereotype' if results[category]['bias_direction'] > 0 else 'Anti-Stereotype'} | "
                )
                f.write(f"{results[category]['stereotype_score']:.4f} | ")
                f.write(f"{results[category]['anti_stereotype_score']:.4f} | ")
                f.write(f"{results[category]['count']} |\n")

        f.write("\n## Detailed Metrics\n\n")
        f.write("| Category | Bias Difference | Bias Ratio |\n")
        f.write("| -------- | --------------- | ---------- |\n")

        for category in sorted(results.keys()):
            if category != "overall" and results[category]["count"] > 0:
                f.write(
                    f"| {category.capitalize()} | {results[category]['bias_difference']:.4f} | "
                )
                if results[category]["bias_ratio"] == float("inf"):
                    f.write("∞ |\n")
                else:
                    f.write(f"{results[category]['bias_ratio']:.4f} |\n")

        f.write("\n## Interpretation\n\n")

        # Write interpretation
        f.write("### SS Score Interpretation\n\n")
        f.write(
            "- **SS Score = 0.5**: No bias (equal preference for stereotypes and anti-stereotypes)\n"
        )
        f.write(
            "- **SS Score > 0.5**: Stereotype bias (model prefers stereotypical associations)\n"
        )
        f.write(
            "- **SS Score < 0.5**: Anti-stereotype bias (model prefers anti-stereotypical associations)\n\n"
        )

        f.write("### Key Findings\n\n")

        # Write some key findings based on the results
        most_biased_category = max(
            [cat for cat in results if cat != "overall" and results[cat]["count"] > 0],
            key=lambda cat: results[cat]["bias_severity"],
        )

        f.write(f"- Most biased category: **{most_biased_category.capitalize()}** ")
        f.write(f"(Severity: {results[most_biased_category]['bias_severity']:.4f})\n")

        if results["overall"]["bias_direction"] > 0:
            f.write("- The model shows an **overall stereotype bias**\n")
        else:
            f.write("- The model shows an **overall anti-stereotype bias**\n")

    print(f"Report saved to {report_file}")
    return report_file


def print_results_summary(results, model_name):
    """
    Print a summary of the evaluation results.

    Args:
        results (dict): Evaluation results
        model_name (str): Name of the model
    """
    print(f"\n===== {model_name} Evaluation Summary =====")
    print(f"Overall SS Score: {results['overall']['ss_score']:.4f}")
    print(f"Overall Bias Severity: {results['overall']['bias_severity']:.4f}")
    print(
        f"Bias Direction: {'Stereotype' if results['overall']['bias_direction'] > 0 else 'Anti-Stereotype'}"
    )
    print("\nCategory Breakdown:")

    for category in sorted(results.keys()):
        if category != "overall" and results[category]["count"] > 0:
            print(
                f"  {category.capitalize():10} | SS Score: {results[category]['ss_score']:.4f} | "
                + f"Severity: {results[category]['bias_severity']:.4f} | "
                + f"Direction: {'Stereotype' if results[category]['bias_direction'] > 0 else 'Anti-Stereotype'}"
            )


def compare_models(evaluation_results, output_dir="results", show_plots=False):
    """
    Compare bias evaluation results from multiple models.

    Args:
        evaluation_results (list): List of evaluation result dictionaries
        output_dir (str): Directory to save comparison results
        show_plots (bool): Whether to display plots

    Returns:
        str: Path to comparison report file
    """
    comparison_dir = os.path.join(
        output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(comparison_dir, exist_ok=True)

    # Extract model names and results
    model_names = [
        result["model_name"].replace("/", "_") for result in evaluation_results
    ]
    model_results = [result["results"] for result in evaluation_results]

    # Generate comparison report
    report_file = os.path.join(comparison_dir, "model_comparison.md")

    with open(report_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("# Model Comparison: Bias Evaluation\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Write overall comparison
        f.write("## Overall Comparison\n\n")
        f.write("| Model | SS Score | Bias Severity | Bias Direction |\n")
        f.write("| ----- | -------- | ------------- | -------------- |\n")

        for i, model_name in enumerate(model_names):
            overall_results = model_results[i]["overall"]
            f.write(f"| {model_name} | {overall_results['ss_score']:.4f} | ")
            f.write(f"{overall_results['bias_severity']:.4f} | ")
            f.write(
                f"{'Stereotype' if overall_results['bias_direction'] > 0 else 'Anti-Stereotype'} |\n"
            )

        # Write category comparisons
        categories = sorted(
            [cat for cat in model_results[0].keys() if cat != "overall"]
        )

        for category in categories:
            f.write(f"\n## {category.capitalize()} Comparison\n\n")
            f.write(
                "| Model | SS Score | Bias Severity | Stereotype Score | Anti-Stereotype Score |\n"
            )
            f.write(
                "| ----- | -------- | ------------- | ---------------- | --------------------- |\n"
            )

            for i, model_name in enumerate(model_names):
                if model_results[i][category]["count"] > 0:
                    cat_results = model_results[i][category]
                    f.write(f"| {model_name} | {cat_results['ss_score']:.4f} | ")
                    f.write(f"{cat_results['bias_severity']:.4f} | ")
                    f.write(f"{cat_results['stereotype_score']:.4f} | ")
                    f.write(f"{cat_results['anti_stereotype_score']:.4f} |\n")

        # Write key findings
        f.write("\n## Key Findings\n\n")

        # Determine least and most biased model overall
        least_biased_idx = min(
            range(len(model_results)),
            key=lambda i: model_results[i]["overall"]["bias_severity"],
        )
        most_biased_idx = max(
            range(len(model_results)),
            key=lambda i: model_results[i]["overall"]["bias_severity"],
        )

        f.write(f"- **Least biased model**: {model_names[least_biased_idx]} ")
        f.write(
            f"(Severity: {model_results[least_biased_idx]['overall']['bias_severity']:.4f})\n"
        )

        f.write(f"- **Most biased model**: {model_names[most_biased_idx]} ")
        f.write(
            f"(Severity: {model_results[most_biased_idx]['overall']['bias_severity']:.4f})\n\n"
        )

        # Category-specific findings
        for category in categories:
            cat_least_biased_idx = min(
                range(len(model_results)),
                key=lambda i: (
                    model_results[i][category]["bias_severity"]
                    if model_results[i][category]["count"] > 0
                    else float("inf")
                ),
            )

            if model_results[cat_least_biased_idx][category]["count"] > 0:
                f.write(
                    f"- For **{category}** bias, {model_names[cat_least_biased_idx]} performs best "
                )
                f.write(
                    f"(Severity: {model_results[cat_least_biased_idx][category]['bias_severity']:.4f})\n"
                )

    # Generate comparison visualizations
    generate_comparison_plots(model_names, model_results, comparison_dir, show_plots)

    print(f"Model comparison report saved to {report_file}")
    return report_file


def generate_comparison_plots(model_names, model_results, output_dir, show_plots=False):
    """
    Generate visualizations comparing multiple models.

    Args:
        model_names (list): List of model names
        model_results (list): List of model results dictionaries
        output_dir (str): Directory to save visualizations
        show_plots (bool): Whether to display plots
    """
    # Extract categories
    categories = sorted([cat for cat in model_results[0].keys() if cat != "overall"])
    categories.append("overall")  # Add overall at the end

    # Create bar chart comparing SS scores
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    x = np.arange(len(categories))
    width = 0.8 / len(model_names)

    for i, model_name in enumerate(model_names):
        ss_scores = [model_results[i][cat]["ss_score"] for cat in categories]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, ss_scores, width, label=model_name)

    # Add a horizontal line at 0.5 to indicate neutral bias
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Neutral (0.5)")

    ax.set_xlabel("Category")
    ax.set_ylabel("SS Score")
    ax.set_title("Stereotype Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save figure
    ss_plot_path = os.path.join(output_dir, "ss_score_comparison.png")
    plt.savefig(ss_plot_path)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # Create bar chart comparing bias severity
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    for i, model_name in enumerate(model_names):
        bias_severity = [model_results[i][cat]["bias_severity"] for cat in categories]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, bias_severity, width, label=model_name)

    ax.set_xlabel("Category")
    ax.set_ylabel("Bias Severity")
    ax.set_title("Bias Severity Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save figure
    severity_plot_path = os.path.join(output_dir, "bias_severity_comparison.png")
    plt.savefig(severity_plot_path)

    if show_plots:
        plt.show()
    else:
        plt.close()


def main():
    """
    Main function to evaluate multiple models for bias.
    """
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # List of models to evaluate
    models = ["distilbert-base-uncased", "roberta-base"]

    # Evaluate each model
    evaluation_results = []
    for model_name in models:
        result = evaluate_model_bias(
            model_name=model_name,
            device=device,
            split="test",
            output_dir=output_dir,
            show_plots=False,
        )
        evaluation_results.append(result)

    # Compare models
    compare_models(
        evaluation_results=evaluation_results, output_dir=output_dir, show_plots=False
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
