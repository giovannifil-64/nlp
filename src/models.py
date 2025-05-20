from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig
)


def load_model(model_name="distilbert-base-uncased", device="mps"):
    """
    Load a pre-trained transformer model and its tokenizer.

    Parameters
    ----------
    model_name : str
        Name of the model to load from HuggingFace.
        Examples: "distilbert-base-uncased", "albert-base-v2", "roberta-base", "HuggingFaceTB/SmolLM2-360M"
    device : str
        Device to load the model on. Options: "cuda", "cpu", "mps"

    Returns
    -------
    tuple
        (model, tokenizer)
        
    Notes
    -----
    This function automatically detects the model architecture and uses the appropriate
    model class for loading:
    - For BERT, RoBERTa, etc.: Uses AutoModelForMaskedLM
    - For GPT, LLaMA, etc.: Uses AutoModelForCausalLM
    - For T5, BART, etc.: Uses AutoModelForSeq2SeqLM
    """
    print(f"Loading model: {model_name}")

    try:
        config = AutoConfig.from_pretrained(model_name)
        model_type = config.model_type
        
        if model_type in ["llama", "gpt2", "gpt_neo", "gptj", "bloom", "opt"]:
            print(f"Loading {model_type} model as a causal language model")
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type in ["t5", "bart", "pegasus"]:
            print(f"Loading {model_type} model as a sequence-to-sequence model")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            print(f"Loading {model_type} model as a masked language model")
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Keep track if we need to resize token embeddings
        special_tokens_added = False
        
        # Set pad_token if it doesn't exist (common in causal LMs like GPT, LLaMA)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
                print(f"Set pad_token to bos_token: {tokenizer.pad_token}")
            elif tokenizer.cls_token is not None:
                tokenizer.pad_token = tokenizer.cls_token
                print(f"Set pad_token to cls_token: {tokenizer.pad_token}")
            else:
                # Add a pad token if none exists
                special_tokens_dict = {'pad_token': '[PAD]'}
                num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
                if num_added_tokens > 0:
                    special_tokens_added = True
                print(f"Added [PAD] token to tokenizer")
        
        # Resize model embeddings if new tokens were added
        if special_tokens_added:
            print(f"Resizing model embeddings to match tokenizer size ({len(tokenizer)})")
            model.resize_token_embeddings(len(tokenizer))

        # Move model to specified device
        model = model.to(device)

        print(f"Model loaded successfully and moved to {device}")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
