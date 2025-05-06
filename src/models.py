from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_model(model_name="distilbert-base-uncased", device="mps"):
    """
    Load a pre-trained transformer model and its tokenizer.

    Args:
        model_name (str): Name of the model to load from HuggingFace.
            Options: "distilbert-base-uncased", "albert-base-v2", "roberta-base"
        device (str): Device to load the model on. Options: "cuda", "cpu", "mps"

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Move model to specified device
        model = model.to(device)

        print(f"Model loaded successfully and moved to {device}")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
