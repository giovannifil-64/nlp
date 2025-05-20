import argparse
import json
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import StereoSetDataset
from src.evaluate_models import evaluate_model_bias, compare_models
from src.evaluation import BiasEvaluator
from src.fine_tuning import fine_tune_model, load_fine_tuned_model
from src.models import load_model


parser = argparse.ArgumentParser(
    prog="Mind the Gap",
    description="Evaluate and fine-tune LLMs to evaluate the presence and extent of social bias.",
    epilog="Project proposal: https://github.com/afflint/nlp/blob/main/2024-25/projects/nlp-projects-2024-25.md#mind-the-gap-p2",
)

parser.add_argument(
    "-e",
    "--evaluate",
    action="store_true",
    help="Evaluate the model. Use with --models to run the evaluation on multiple models.",
)

parser.add_argument(
    "-ft",
    "--fine-tune",
    action="store_true",
    help="Fine-tune the model",
)

parser.add_argument(
    "-e-ft",
    "--evaluate-fine-tuned",
    action="store_true",
    help="Evaluate the fine-tuned model",
)

parser.add_argument(
    "-c",
    "--compare",
    action="store_true",
    help="Compare the results with the original model",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="distilbert-base-uncased",
    help="Specify the model name to use. Available models: distilbert-base-uncased, albert-base-v2, roberta-base.",
)

parser.add_argument(
    "-d",
    "--device",
    type=str,
    default=None,
    help="Device to run the model on (cuda, mps, cpu). If not specified, will use CUDA if available, otherwise MPS if running on Apple Silicon, and fallback to the CPU if neither are available.",
)

parser.add_argument(
    "-s",
    "--split",
    type=str,
    default="test",
    help="Dataset split to use (dev, test)",
)

parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="results",
    help="Directory to save results",
)

parser.add_argument(
    "-sp",
    "--show-plots",
    action="store_true",
    help="Show visualization plots",
)

parser.add_argument(
    "-ms",
    "--models",
    nargs="+",
    default=["distilbert-base-uncased", "roberta-base"],
    help="List of models to evaluate",
)

parser.add_argument(
    "-ep",
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs for fine-tuning (default: 3)",
)

parser.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for fine-tuning (default: 16)",
)

parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=5e-5,
    help="Learning rate for fine-tuning (default: 5e-5)",
)

parser.add_argument(
    "-md",
    "--models-dir",
    type=str,
    default="models",
    help="Directory to save/load fine-tuned models",
)

args = vars(parser.parse_args())

for arg in [
    "evaluate",
    "fine_tune",
    "evaluate_fine_tuned",
    "compare",
    "show_plots",
]:
    if arg not in args:
        args[arg] = False

if args["device"] is None:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = args["device"]

if not os.path.exists(args["output_dir"]):
    os.makedirs(args["output_dir"])

if not os.path.exists(args["models_dir"]):
    os.makedirs(args["models_dir"])


def evaluate_model(model_name, device, split, output_dir, show_plots):
    """Run the evaluation pipeline."""
    print(f"Evaluating model: {model_name} on device: {device}")

    model, tokenizer = load_model(model_name, device)

    dataset = StereoSetDataset()
    dataset.download_dataset(split=split)
    processed_data = dataset.preprocess()

    evaluator = BiasEvaluator(model, tokenizer, device)
    results = evaluator.evaluate_bias(processed_data)

    results_file = evaluator.save_results(
        save_path=output_dir,
        filename=f"{model_name.replace('/', '_')}_bias_evaluation.json",
    )

    vis_file = evaluator.visualize_results(save_path=output_dir, show=show_plots)

    print("\n===== Evaluation Results =====")
    print(f"Overall SS Score: {results['overall']['ss_score']:.4f}")

    for category in sorted(results.keys()):
        if category != "overall":
            print(f"{category.capitalize()} SS Score: {results[category]['ss_score']:.4f}")

    print(f"\nResults saved to: {results_file}")
    print(f"Visualization saved to: {vis_file}")

    return results


def run_evaluation(models, device, split, output_dir, show_plots):
    """Run evaluation on multiple models."""
    print(f"Running evaluation on {len(models)} models...")

    evaluation_results = []
    for model_name in models:
        result = evaluate_model_bias(
            model_name=model_name,
            device=device,
            split=split,
            output_dir=output_dir,
            show_plots=show_plots,
        )
        evaluation_results.append(result)

    compare_models(
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        show_plots=show_plots,
    )

    print("\nEvaluation complete!")


def fine_tune_and_save(model_name, device, split, output_dir, models_dir, epochs, batch_size, learning_rate):
    """Fine-tune a model for bias mitigation and save it."""
    print(f"Loading model {model_name} for fine-tuning...")
    
    # Load original model
    model, tokenizer = load_model(model_name, device)
    
    # Load dataset for fine-tuning
    dataset = StereoSetDataset()
    dataset.download_dataset(split=split)
    processed_data = dataset.preprocess()
    
    print(f"Starting fine-tuning on {split} dataset...")
    
    # Fine-tune the model
    fine_tuned_model, tokenizer, stats = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        dataset=processed_data,
        device=device,
        output_dir=models_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print("\n===== Fine-tuning Complete =====")
    print(f"Training statistics:")
    for epoch, loss in enumerate(stats["train_loss_per_epoch"]):
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    print(f"\nFine-tuned model saved to: {models_dir}/{model_name.replace('/', '_')}_final")
    
    return fine_tuned_model, tokenizer


def evaluate_fine_tuned_model(model_name, device, split, output_dir, models_dir, show_plots):
    """Evaluate a fine-tuned model."""
    print(f"Evaluating fine-tuned model: {model_name}")
    
    # Load fine-tuned model
    fine_tuned_model, tokenizer = load_fine_tuned_model(
        model_name=model_name,
        device=device,
        models_dir=models_dir
    )
    
    dataset = StereoSetDataset()
    dataset.download_dataset(split=split)
    processed_data = dataset.preprocess()
    
    evaluator = BiasEvaluator(fine_tuned_model, tokenizer, device)
    results = evaluator.evaluate_bias(processed_data)
    
    results_file = evaluator.save_results(
        save_path=output_dir,
        filename=f"{model_name.replace('/', '_')}_fine_tuned_bias_evaluation.json",
    )
    
    vis_file = evaluator.visualize_results(
        save_path=output_dir, 
        filename_prefix=f"{model_name.replace('/', '_')}_fine_tuned",
        show=show_plots
    )
    
    print("\n===== Fine-tuned Model Evaluation Results =====")
    print(f"Overall SS Score: {results['overall']['ss_score']:.4f}")
    
    for category in sorted(results.keys()):
        if category != "overall":
            print(f"{category.capitalize()} SS Score: {results[category]['ss_score']:.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualization saved to: {vis_file}")
    
    return results


def compare_original_and_fine_tuned(model_name, device, split, output_dir, models_dir, show_plots):
    """Compare the original and fine-tuned model results."""
    print(f"Comparing original and fine-tuned versions of {model_name}")
    
    # Evaluate original model
    original_results = evaluate_model(
        model_name=model_name,
        device=device,
        split=split,
        output_dir=output_dir,
        show_plots=False
    )
    
    # Evaluate fine-tuned model
    fine_tuned_results = evaluate_fine_tuned_model(
        model_name=model_name,
        device=device,
        split=split,
        output_dir=output_dir,
        models_dir=models_dir,
        show_plots=False
    )
    
    comparison = {
        "original": {
            "name": f"Original {model_name}",
            "results": original_results
        },
        "fine_tuned": {
            "name": f"Fine-tuned {model_name}",
            "results": fine_tuned_results
        }
    }

    formatted_model_name = model_name.replace("/", "_")
    comparison_dir = os.path.join(output_dir, f"{time.strftime('%Y-%m-%d')}-{formatted_model_name}-comparison")

    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    timestamp = time.strftime("%H%M%S")
    comparison_file = os.path.join(
        comparison_dir, 
        f"comparison_{timestamp}.json"
    )
    
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Generate comparison visualizations
    generate_comparison_visualizations(
        original_results=original_results,
        fine_tuned_results=fine_tuned_results,
        model_name=model_name,
        output_dir=comparison_dir,
        show_plots=show_plots
    )
    
    print("\n===== Comparison Results =====")
    print(f"Original model - Overall SS Score: {original_results['overall']['ss_score']:.4f}")
    print(f"Fine-tuned model - Overall SS Score: {fine_tuned_results['overall']['ss_score']:.4f}")
    
    # Calculate improvement
    improvement = fine_tuned_results['overall']['ss_score'] - original_results['overall']['ss_score']
    print(f"Overall improvement: {improvement:.4f} ({improvement/original_results['overall']['ss_score']*100:.2f}%)")

    for category in sorted(original_results.keys()):
        if category != "overall":
            orig_score = original_results[category]["ss_score"]
            ft_score = fine_tuned_results[category]["ss_score"]
            cat_improvement = ft_score - orig_score

            print(f"{category.capitalize()}:")
            print(f"  Original: {orig_score:.4f}, Fine-tuned: {ft_score:.4f}")
            print(f"  Improvement: {cat_improvement:.4f} ({cat_improvement/orig_score*100:.2f}%)")
    
    print(f"\nComparison data saved to: {comparison_file}")
    
    return comparison


def generate_comparison_visualizations(original_results, fine_tuned_results, model_name, output_dir, show_plots=False):
    """
    Generate visualizations comparing original and fine-tuned model results.
    
    Parameters
    ----------
    original_results : dict
        Evaluation results for original model
    fine_tuned_results : dict
        Evaluation results for fine-tuned model
    model_name : str
        Name of the model
    output_dir : str
        Directory to save visualizations
    show_plots : bool, optional
        Whether to display plots, by default False
        
    Returns
    -------
    dict
        Paths to generated visualization files
    """
    
    categories = sorted([cat for cat in original_results.keys() if cat != "overall"])
    categories.append("overall")
    
    # 1. Bar chart comparing SS scores across categories
    plt.figure(figsize=(12, 8))
    x = np.arange(len(categories))
    width = 0.35
    
    original_scores = [original_results[cat]["ss_score"] for cat in categories]
    fine_tuned_scores = [fine_tuned_results[cat]["ss_score"] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, original_scores, width, label='Original', color='skyblue')
    rects2 = ax.bar(x + width/2, fine_tuned_scores, width, label='Fine-tuned', color='lightgreen')
    
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Neutral (0.5)')
    
    ax.set_xlabel('Category')
    ax.set_ylabel('SS Score')
    ax.set_title(f'Stereotype Score Comparison: Original vs Fine-tuned {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend()
    
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    fig.tight_layout()
    ss_chart_path = os.path.join(output_dir, 'ss_score_comparison.png')
    plt.savefig(ss_chart_path, dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Radar chart comparing improvement across categories (excluding 'overall')
    categories_without_overall = sorted([cat for cat in original_results.keys() if cat != "overall"])
    
    improvements = [(fine_tuned_results[cat]["ss_score"] - original_results[cat]["ss_score"]) 
                   for cat in categories_without_overall]
    
    N = len(categories_without_overall)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    improvements += improvements[:1]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, improvements, 'o-', linewidth=2)
    ax.fill(angles, improvements, alpha=0.25)
    
    plt.xticks(angles[:-1], [cat.capitalize() for cat in categories_without_overall], size=12)
    
    for i, improvement in enumerate(improvements[:-1]):
        ax.annotate(f'{improvement:.3f}',
                    xy=(angles[i], improvement),
                    xytext=(angles[i], improvement + 0.02),
                    ha='center',
                    va='center')
    
    plt.title(f'Improvement by Category for {model_name}', size=15, y=1.1)
    
    maxval = max(abs(min(improvements)), abs(max(improvements)))
    step = np.ceil(maxval * 10) / 10 / 4
    levels = np.arange(-4*step, 4*step + step, step)
    
    plt.yticks(levels, [f'{level:.2f}' for level in levels], color='grey', size=10)
    plt.ylim(-maxval*1.1, maxval*1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    radar_chart_path = os.path.join(output_dir, 'improvement_radar_chart.png')
    plt.savefig(radar_chart_path, dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    print(f"Visualizations saved to {output_dir}")
    
    return {
        "ss_chart": ss_chart_path,
        "radar_chart": radar_chart_path
    }


if args["evaluate"]:
    if "--models" in sys.argv or "-ms" in sys.argv:
        run_evaluation(
            models=args["models"],
            device=device,
            split=args["split"],
            output_dir=args["output_dir"],
            show_plots=args["show_plots"],
        )
    else:
        evaluate_model(
            model_name=args["model"],
            device=device,
            split=args["split"],
            output_dir=args["output_dir"],
            show_plots=args["show_plots"],
        )

elif args["fine_tune"]:
    if "--models" in sys.argv or "-ms" in sys.argv:
        for model_name in args["models"]:
            print(f"\nFine-tuning Model: {model_name}")
            fine_tune_and_save(
                model_name=model_name,
                device=device,
                split=args["split"],
                output_dir=args["output_dir"],
                models_dir=args["models_dir"],
                epochs=args["epochs"],
                batch_size=args["batch_size"],
                learning_rate=args["learning_rate"],
            )
    else:
        fine_tune_and_save(
            model_name=args["model"],
            device=device,
            split=args["split"],
            output_dir=args["output_dir"],
            models_dir=args["models_dir"],
            epochs=args["epochs"],
            batch_size=args["batch_size"],
            learning_rate=args["learning_rate"],
        )

elif args["evaluate_fine_tuned"]:
    if "--models" in sys.argv or "-ms" in sys.argv:
        for model_name in args["models"]:
            print(f"\nEvaluating Fine-tuned Model: {model_name}")
            evaluate_fine_tuned_model(
                model_name=model_name,
                device=device,
                split=args["split"],
                output_dir=args["output_dir"],
                models_dir=args["models_dir"],
                show_plots=args["show_plots"],
            )
    else:
        evaluate_fine_tuned_model(
            model_name=args["model"],
            device=device,
            split=args["split"],
            output_dir=args["output_dir"],
            models_dir=args["models_dir"],
            show_plots=args["show_plots"],
        )

elif args["compare"]:
    if "--models" in sys.argv or "-ms" in sys.argv:
        for model_name in args["models"]:
            print(f"\nComparing Original and Fine-tuned: {model_name}")
            compare_original_and_fine_tuned(
                model_name=model_name,
                device=device,
                split=args["split"],
                output_dir=args["output_dir"],
                models_dir=args["models_dir"],
                show_plots=args["show_plots"],
            )
    else:
        compare_original_and_fine_tuned(
            model_name=args["model"],
            device=device,
            split=args["split"],
            output_dir=args["output_dir"],
            models_dir=args["models_dir"],
            show_plots=args["show_plots"],
        )

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
