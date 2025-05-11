import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from datetime import datetime

from src.dataset import StereoSetDataset
from src.evaluation import BiasEvaluator
from src.models import load_model


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
    model_output_dir = os.path.join(output_dir, f"{model_name.replace('/', '_')}_{timestamp}")
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
    for category in results:
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
        f.write(f"# Bias Evaluation Report: {model_name}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overall Summary\n\n")
        f.write(f"- **SS Score**: {results['overall']['ss_score']:.4f}\n")
        f.write(f"- **Bias Severity**: {results['overall']['bias_severity']:.4f}\n")
        f.write(f"- **Bias Direction**: {'Stereotype' if results['overall']['bias_direction'] > 0 else 'Anti-Stereotype'}\n")
        f.write(f"- **Total Examples**: {results['overall']['count']}\n\n")

        f.write("## Category Breakdown\n\n")
        f.write("| Category | SS Score | Bias Severity | Bias Direction | Stereotype Score | Anti-Stereotype Score | Examples |\n")
        f.write("| -------- | -------- | ------------- | -------------- | ---------------- | --------------------- | -------- |\n")

        for category in sorted(results.keys()):
            if category != "overall" and results[category]["count"] > 0:
                f.write(f"| {category.capitalize()} | {results[category]['ss_score']:.4f} | ")
                f.write(f"{results[category]['bias_severity']:.4f} | ")
                f.write(f"{'Stereotype' if results[category]['bias_direction'] > 0 else 'Anti-Stereotype'} | ")
                f.write(f"{results[category]['stereotype_score']:.4f} | ")
                f.write(f"{results[category]['anti_stereotype_score']:.4f} | ")
                f.write(f"{results[category]['count']} |\n")

        f.write("\n## Detailed Metrics\n\n")
        f.write("| Category | Bias Difference | Bias Ratio |\n")
        f.write("| -------- | --------------- | ---------- |\n")

        for category in sorted(results.keys()):
            if category != "overall" and results[category]["count"] > 0:
                f.write(f"| {category.capitalize()} | {results[category]['bias_difference']:.4f} | ")
                if results[category]["bias_ratio"] == float("inf"):
                    f.write("∞ |\n")
                else:
                    f.write(f"{results[category]['bias_ratio']:.4f} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("### SS Score Interpretation\n\n")
        f.write("- **SS Score = 0.5**: No bias (equal preference for stereotypes and anti-stereotypes)\n")
        f.write("- **SS Score > 0.5**: Stereotype bias (model prefers stereotypical associations)\n")
        f.write("- **SS Score < 0.5**: Anti-stereotype bias (model prefers anti-stereotypical associations)\n\n")

        f.write("### Key Findings\n\n")
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
    print(f"Bias Direction: {'Stereotype' if results['overall']['bias_direction'] > 0 else 'Anti-Stereotype'}")
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
        dict: Dictionary with paths to comparison report file and visualizations
    """
    comparison_dir = os.path.join(output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(comparison_dir, exist_ok=True)

    model_names = [result["model_name"].replace("/", "_") for result in evaluation_results]
    model_results = [result["results"] for result in evaluation_results]

    report_file = os.path.join(comparison_dir, "model_comparison.md")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Model Comparison: Bias Evaluation\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overall Comparison\n\n")
        f.write("| Model | SS Score | Bias Severity | Bias Direction |\n")
        f.write("| ----- | -------- | ------------- | -------------- |\n")

        for i, model_name in enumerate(model_names):
            overall_results = model_results[i]["overall"]
            f.write(f"| {model_name} | {overall_results['ss_score']:.4f} | ")
            f.write(f"{overall_results['bias_severity']:.4f} | ")
            f.write(f"{'Stereotype' if overall_results['bias_direction'] > 0 else 'Anti-Stereotype'} |\n")

        categories = sorted([cat for cat in model_results[0].keys() if cat != "overall"])

        for category in categories:
            f.write(f"\n## {category.capitalize()} Comparison\n\n")
            f.write("| Model | SS Score | Bias Severity | Stereotype Score | Anti-Stereotype Score |\n")
            f.write("| ----- | -------- | ------------- | ---------------- | --------------------- |\n")

            for i, model_name in enumerate(model_names):
                if model_results[i][category]["count"] > 0:
                    cat_results = model_results[i][category]
                    f.write(f"| {model_name} | {cat_results['ss_score']:.4f} | ")
                    f.write(f"{cat_results['bias_severity']:.4f} | ")
                    f.write(f"{cat_results['stereotype_score']:.4f} | ")
                    f.write(f"{cat_results['anti_stereotype_score']:.4f} |\n")

        f.write("\n## Key Findings\n\n")
        least_biased_idx = min(
            range(len(model_results)),
            key=lambda i: model_results[i]["overall"]["bias_severity"],
        )
        most_biased_idx = max(
            range(len(model_results)),
            key=lambda i: model_results[i]["overall"]["bias_severity"],
        )

        f.write(f"- **Least biased model**: {model_names[least_biased_idx]} ")
        f.write(f"(Severity: {model_results[least_biased_idx]['overall']['bias_severity']:.4f})\n")

        f.write(f"- **Most biased model**: {model_names[most_biased_idx]} ")
        f.write(f"(Severity: {model_results[most_biased_idx]['overall']['bias_severity']:.4f})\n\n")

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
                f.write(f"- For **{category}** bias, {model_names[cat_least_biased_idx]} performs best ")
                f.write(f"(Severity: {model_results[cat_least_biased_idx][category]['bias_severity']:.4f})\n")

    visualization_paths = generate_comparison_plots(model_names, model_results, comparison_dir, show_plots)

    print(f"\nModel comparison report saved to {report_file}")
    
    return {
        "report_file": report_file,
        "visualizations": visualization_paths
    }


def generate_comparison_plots(model_names, model_results, output_dir, show_plots=False):
    """
    Generate visualizations comparing multiple models.

    Args:
        model_names (list): List of model names
        model_results (list): List of model results dictionaries
        output_dir (str): Directory to save visualizations
        show_plots (bool): Whether to display plots
        
    Returns:
        dict: Dictionary with paths to visualization files
    """
    categories = sorted([cat for cat in model_results[0].keys() if cat != "overall"])
    categories.append("overall")

    # 1. SS Score comparison bar chart
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    x = np.arange(len(categories))
    width = 0.8 / len(model_names)

    for i, model_name in enumerate(model_names):
        ss_scores = [model_results[i][cat]["ss_score"] for cat in categories]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, ss_scores, width, label=model_name)

    # Horizontal line at 0.5 to indicate neutral bias
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Neutral (0.5)")

    ax.set_xlabel("Category")
    ax.set_ylabel("SS Score")
    ax.set_title("Stereotype Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ss_plot_path = os.path.join(output_dir, "ss_score_comparison.png")
    plt.savefig(ss_plot_path)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2. Bias severity comparison bar chart
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

    severity_plot_path = os.path.join(output_dir, "bias_severity_comparison.png")
    plt.savefig(severity_plot_path)

    if show_plots:
        plt.show()
    else:
        plt.close()
        
    # 3. Heatmap of SS scores across models and categories
    categories_without_overall = sorted([cat for cat in model_results[0].keys() if cat != "overall"])
    
    # Create data array for heatmap
    data = np.zeros((len(model_names), len(categories_without_overall)))
    for i, model in enumerate(model_results):
        for j, category in enumerate(categories_without_overall):
            data[i, j] = model[category]["ss_score"]
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Create heatmap
    im = ax.imshow(data, cmap='YlOrRd')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('SS Score')
    
    # Add labels
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xticks(range(len(categories_without_overall)))
    ax.set_xticklabels([cat.capitalize() for cat in categories_without_overall], rotation=45, ha="right")
    
    # Add value annotations on the heatmap
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black")
    
    ax.set_title('Stereotype Scores Across Models and Categories')
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "ss_score_heatmap.png")
    plt.savefig(heatmap_path)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 4. Radar chart comparing models across categories
    categories_without_overall = sorted([cat for cat in model_results[0].keys() if cat != "overall"])
    
    # Number of variables
    N = len(categories_without_overall)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], [cat.capitalize() for cat in categories_without_overall], size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7], ["0.3", "0.4", "0.5", "0.6", "0.7"], color="grey", size=10)
    plt.ylim(0.3, 0.7)
    
    # Plot data
    for i, model_name in enumerate(model_names):
        model_scores = [model_results[i][cat]["ss_score"] for cat in categories_without_overall]
        model_scores += model_scores[:1]  # Close the loop
        
        # Plot data and solid line connecting them
        ax.plot(angles, model_scores, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, model_scores, alpha=0.1)
    
    # Add neutral line (0.5)
    neutral = [0.5 for _ in range(N)]
    neutral += neutral[:1]
    ax.plot(angles, neutral, linewidth=1, linestyle='--', color='r', alpha=0.7, label='Neutral (0.5)')
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Stereotype Score Comparison (Radar Chart)', size=15, y=1.1)
    
    radar_path = os.path.join(output_dir, "ss_score_radar.png")
    plt.savefig(radar_path)
    
    if show_plots:
        plt.show()
    else:
        plt.close()

    print(f"Visualizations saved to {output_dir}")

    return {
        "ss_score_comparison": ss_plot_path,
        "bias_severity_comparison": severity_plot_path,
        "ss_score_heatmap": heatmap_path,
        "ss_score_radar": radar_path
    }


def main():
    """
    Main function to evaluate multiple models for bias.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    models = ["distilbert-base-uncased", "roberta-base"]

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

    compare_models(evaluation_results=evaluation_results, output_dir=output_dir, show_plots=False)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
