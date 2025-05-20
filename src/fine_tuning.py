import json
import os
import torch

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class BiasMitigationDataset(Dataset):
    """
    Dataset class for bias mitigation fine-tuning.
    
    Parameters
    ----------
    data : list
        List of examples for training
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    max_length : int, optional
        Maximum sequence length for tokenization, by default 128
    """
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        sentence = example["sentence"]
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        # Set padding tokens to -100 so they're ignored in the loss calculation
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            # If there's no pad_token_id, use the attention mask to mask padding tokens
            labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ModelFineTuner:
    """
    Class for fine-tuning language models to mitigate bias.
    
    Parameters
    ----------
    model : transformers.PreTrainedModel
        The pre-trained transformer model to fine-tune
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    device : str, optional
        Device to run training on, by default "cuda"
    output_dir : str, optional
        Directory to save the fine-tuned model, by default "models"
    """
    
    def __init__(self, model, tokenizer, device="mps", output_dir="models"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def prepare_counter_stereotypical_data(self, dataset):
        """
        Prepare data for bias mitigation fine-tuning by selecting
        anti-stereotype examples.
        
        Parameters
        ----------
        dataset : dict
            Preprocessed dataset with categories
            
        Returns
        -------
        list
            List of examples for fine-tuning
        """
        training_data = []
        
        for category, examples in dataset.items():
            grouped_examples = {}
            for example in examples:
                if example["id"] not in grouped_examples:
                    grouped_examples[example["id"]] = []
                grouped_examples[example["id"]].append(example)
            
            for example_id, example_set in grouped_examples.items():
                for ex in example_set:
                    if ex["label"] == "anti-stereotype":
                        training_data.append(ex)
        
        print(f"Prepared {len(training_data)} counter-stereotypical examples for fine-tuning")
        return training_data
    
    def fine_tune(self, training_data, epochs=3, batch_size=16, learning_rate=5e-5, weight_decay=0.01, warmup_steps=0, save_every_epoch=True):
        """
        Fine-tune the model on counter-stereotypical data.
        
        Parameters
        ----------
        training_data : list
            List of examples for training
        epochs : int, optional
            Number of training epochs, by default 3
        batch_size : int, optional
            Batch size for training, by default 16
        learning_rate : float, optional
            Learning rate for optimization, by default 5e-5
        weight_decay : float, optional
            Weight decay for regularization, by default 0.01
        warmup_steps : int, optional
            Number of warmup steps for learning rate scheduler, by default 0
        save_every_epoch : bool, optional
            Whether to save the model after each epoch, by default True
            
        Returns
        -------
        dict
            Training statistics and information
        """
        
        dataset = BiasMitigationDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = self.model.to(self.device)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        training_stats = {
            "train_loss_per_epoch": [],
            "model_name": self.model.config._name_or_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_examples": len(training_data)
        }
        
        print(f"Starting fine-tuning for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            training_stats["train_loss_per_epoch"].append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            if save_every_epoch:
                self.save_model(f"epoch_{epoch+1}")
        
        self.save_model("final")
        
        stats_path = os.path.join(self.output_dir, "fine_tuning_stats.json")
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
    
    def save_model(self, suffix="final"):
        """
        Save the fine-tuned model and tokenizer.
        
        Parameters
        ----------
        suffix : str, optional
            Suffix for the saved model directory, by default "final"
            
        Returns
        -------
        str
            Path to the saved model
        """

        model_name = self.model.config._name_or_path.replace("/", "_")
        save_dir = os.path.join(self.output_dir, f"{model_name}_{suffix}")
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Model saved to {save_dir}")
        return save_dir


def fine_tune_model(model, tokenizer, dataset, device="mps", output_dir="models", epochs=3, batch_size=16, learning_rate=5e-5):
    """
    Convenience function to fine-tune a model for bias mitigation.
    
    Parameters
    ----------
    model : transformers.PreTrainedModel
        The pre-trained transformer model to fine-tune
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    dataset : dict
        Preprocessed dataset with categories
    device : str, optional
        Device to run training on, by default "cuda"
    output_dir : str, optional
        Directory to save the fine-tuned model, by default "models"
    epochs : int, optional
        Number of training epochs, by default 3
    batch_size : int, optional
        Batch size for training, by default 16
    learning_rate : float, optional
        Learning rate for optimization, by default 5e-5
        
    Returns
    -------
    tuple
        (fine-tuned model, tokenizer, training_stats)
    """

    fine_tuner = ModelFineTuner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir
    )
    
    training_data = fine_tuner.prepare_counter_stereotypical_data(dataset)

    training_stats = fine_tuner.fine_tune(
        training_data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return fine_tuner.model, tokenizer, training_stats


def load_fine_tuned_model(model_name, device="mps", models_dir="models"):
    """
    Load a fine-tuned model from disk.
    
    Parameters
    ----------
    model_name : str
        Original model name or path
    device : str, optional
        Device to load the model on, by default "cuda"
    models_dir : str, optional
        Directory containing fine-tuned models, by default "models"
        
    Returns
    -------
    tuple
        (fine-tuned model, tokenizer)
    """

    model_path = os.path.join(models_dir, f"{model_name.replace('/', '_')}_final")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Fine-tuned model not found at {model_path}")
    
    print(f"Loading fine-tuned model from {model_path}")
    
    # Check the model configuration to determine the correct model class
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type
    
    # Load the appropriate model type based on the architecture
    if model_type in ["llama", "gpt2", "gpt_neo", "gptj", "bloom", "opt"]:
        print(f"Loading fine-tuned {model_type} model as a causal language model")
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif model_type in ["t5", "bart", "pegasus"]:
        print(f"Loading fine-tuned {model_type} model as a sequence-to-sequence model")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        print(f"Loading fine-tuned {model_type} model as a masked language model")
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)
    
    return model, tokenizer
