# Natural Language Processing

Repository containing the code of the project for the _"Natural Language Processing"_ course (Academic Year 2024-25) at the University of Milan, as part of the Master Degree in Computer Science.

_This project is intended for educational purposes only_


> [!IMPORTANT]
> **AI Disclosure**
> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Project Proposal
> [!NOTE]
> This description is the same as the one in the `README.md` file in the [course repository](https://github.com/afflint/nlp/blob/main/2024-25/projects/nlp-projects-2024-25.md#mind-the-gap-p2)

### Mind the gap (P2)

This project aims to identify, measure, and mitigate social biases, such as gender, race, or profession-related stereotypes, in lightweight transformer models through hands-on fine-tuning and evaluation on targeted NLP tasks. More specifically, the project should implement a four-step methodology, defined as follows:

1. Choose a lightweight pre-trained transformer model (e.g., DistilBERT, ALBERT, RoBERTa-base) suitable for local fine-tuning and evaluation.
2. Evaluate the presence and extent of social bias (e.g., gender, racial, or occupational stereotypes) using dedicated benchmark datasets. Both quantitative metrics and qualitative outputs should be evaluated.
3. Apply a bias mitigation technique, such as **fine-tuning on curated counter-stereotypical data**, integrating **adapter layers**, or employing **contrastive learning**, while keeping the solution computationally efficient and transparent.
4. Re-assess the model using the same benchmark(s) to measure improvements. Students should compare pre- and post-intervention results, discuss trade-offs (e.g., performance vs. fairness), and visualize the impact of their approach.

#### Dataset

- [StereoSet: Measuring stereotypical bias in pretrained language models](https://github.com/moinnadeem/StereoSet). Nadeem, M., Bethke, A., & Reddy, S. (2020). StereoSet: Measuring stereotypical bias in pretrained language models. *arXiv preprint arXiv:2004.09456*.

#### References

- Zhang, Y., & Zhou, F. (2024). Bias mitigation in fine-tuning pre-trained models for enhanced fairness and efficiency. *arXiv preprint arXiv:2403.00625*.
- Fu, C. L., Chen, Z. C., Lee, Y. R., & Lee, H. Y. (2022). Adapterbias: Parameter-efficient token-dependent representation shift for adapters in nlp tasks. *arXiv preprint arXiv:2205.00305*.
- Park, K., Oh, S., Kim, D., & Kim, J. (2024, June). Contrastive Learning as a Polarizer: Mitigating Gender Bias by Fair and Biased sentences. In *Findings of the Association for Computational Linguistics: NAACL 2024* (pp. 4725-4736).

## Project Description

This project evaluates language models for social bias using the StereoSet benchmark. It supports multiple model architectures including masked language models (BERT-like) and causal language models (GPT-like, LLaMA).

## Features

- Evaluate pre-trained language models for social bias across gender, profession, race, and religion
- Automatically detect model architecture and use appropriate evaluation method
- Generate detailed reports with bias metrics and visualizations
- Compare multiple models side-by-side
- Supports both masked language models and causal language models

## Supported Model Types

The tool automatically detects and handles different model architectures:

- **Masked Language Models**: BERT, RoBERTa, DistilBERT, ALBERT, etc.
- **Causal Language Models**: GPT-2, LLaMA, OPT, BLOOM, etc.
- **Sequence-to-Sequence Models**: T5, BART, PEGASUS, etc.

## Project Structure

```bash
nlp-project/
├── data/                    # Bias datasets
│   ├── stereoset_dev.json   # Smaller dataset used for evaluation before and after fine-tuning
│   └── stereoset_test.json  # Larger dataset used for fine-tuning
├── docs/                    # Documentation produced for the project
│   └── ...
├── models/                  # Folder containing the fine-tuned models
│   └── ...
├── results/                 # Folder with the results of the evaluation 
│   └── ...
├── src/                     # Source code
│   ├── dataset.py           # StereoSet dataset utilities
│   ├── evaluate_models.py   # Evaluate models for bias
│   ├── evaluation.py        # Logic for the evaluation of the bias
│   ├── fine_tuning.py       # Fine-tuning for bias mitigation
│   └── models.py            # Model loading utilities
├── tests/                   # Test suite
│   └── ...
├── .gitignore
├── LICENSE
├── main.py                  # Main script entry point
├── README.md
└── requirements.txt
```

You can run:

```bash
python main.py --help
```

to see the available commands and their description.

## Usage

> [!IMPORTANT]
> The project has been developed with Python 3.11.9.

Is recommended to use a virtual environment to install the dependencies. I recommend using Miniconda, as you can specify the Python version to use.

```bash
conda create -n nlp-project python=3.11.9
conda activate nlp-project
```

Install the dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Evaluation

To run the evaluation of a model, you can use the command:

```bash
python main.py --evaluate --model distilbert-base-uncased
```

It's also possible to specify multiple models in all the commands that require a model by using the `--models` flag.

```bash
python main.py --evaluate --models distilbert-base-uncased roberta-base
```

This command will evaluate the model(s) and save the results in the `results/evaluations` folder.

Two datasets are available:
- `stereoset_dev.json`: Smaller dataset used for evaluation before and after fine-tuning (default for evaluation)
- `stereoset_test.json`: Larger dataset used for fine-tuning (default for training)

### Fine-tuning

To fine tune one, or multiple models, you can use:

```bash
python main.py --fine-tune --model distilbert-base-uncased
```

This will use the test dataset by default. To use a different dataset for fine-tuning:

```bash
python main.py --fine-tune --model distilbert-base-uncased --fine-tune-split dev
```

Additional parameters can be specified to the fine-tuning command, such as the number of epochs, the batch size, the device, etc.

The fine-tuned model(s) will be saved in the `models` folder.

### Evaluation after fine-tuning

To evaluate the fine-tuned model(s), run:

```bash
python main.py --evaluate-fine-tuned --model distilbert-base-uncased
```

This will use the dev dataset by default, the same used for the initial evaluation. To use a different dataset:

```bash
python main.py --evaluate-fine-tuned --model distilbert-base-uncased --split test
```

The results will be saved in the `results/fine_tuned_evaluations` folder.

### Comparing the results

To compare the results of pre- and post-fine-tuning, just run:

```bash
python main.py --compare --model distilbert-base-uncased
```

The results will be saved in the `results/comparisons` folder.

## License

[MIT License](LICENSE)