import argparse
import os
import sys
import torch

from src.dataset import StereoSetDataset
from src.evaluation import BiasEvaluator
from src.evaluate_models import evaluate_model_bias, compare_models
from src.models import load_model


parser = argparse.ArgumentParser(
    prog="Bias Evaluator",
    description="Evaluate and fine-tune LLMs to evaluate the presence and extent of social bias.",
    epilog="Lorem ipsum dolor sit amet.",
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
    help="Device to run the model on (cuda, cpu, mps). If not specified, will use CUDA if available, else CPU.",
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
    "--show-plots",
    action="store_true",
    help="Show visualization plots",
)

parser.add_argument(
    "--models",
    nargs="+",
    default=["distilbert-base-uncased", "roberta-base"],
    help="List of models to evaluate",
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


if args["evaluate"]:
    if "--models" in sys.argv or "-models" in sys.argv:
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
    print("Fine-tuning the model")
    if args["model"]:
        print(f"Using the model: {args['model']}")
    # Fine-tuning implementation will be added in the next phase

elif args["evaluate_fine_tuned"]:
    print("Evaluating the fine-tuned model")
    if args["model"]:
        print(f"Using the model: {args['model']}")
    # Evaluating fine-tuned model implementation will be added in the next phase

elif args["compare"]:
    print("Comparing the results with the original model")
    if args["model"]:
        print(f"Using the model: {args['model']}")
    # Comparison implementation will be added in the next phase

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
