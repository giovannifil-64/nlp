import os
import pytest
import shutil
import torch
import tempfile

from transformers import AutoModelForMaskedLM, AutoTokenizer
from unittest.mock import patch, MagicMock

from src.fine_tuning import (
    BiasMitigationDataset,
    ModelFineTuner,
    fine_tune_model,
    load_fine_tuned_model
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return {
        "gender": [
            {
                "id": "test1",
                "target": "person",
                "context": "The person works as a",
                "sentence": "The person works as a nurse.",
                "label": "stereotype",
                "bias_type": "gender",
            },
            {
                "id": "test1",
                "target": "person",
                "context": "The person works as a",
                "sentence": "The person works as a doctor.",
                "label": "anti-stereotype",
                "bias_type": "gender",
            },
            {
                "id": "test2",
                "target": "individual",
                "context": "The individual enjoys",
                "sentence": "The individual enjoys shopping.",
                "label": "stereotype",
                "bias_type": "gender",
            },
            {
                "id": "test2",
                "target": "individual",
                "context": "The individual enjoys",
                "sentence": "The individual enjoys sports.",
                "label": "anti-stereotype",
                "bias_type": "gender",
            }
        ]
    }


@pytest.fixture
def mock_model_tokenizer():
    """Create mock model and tokenizer for testing."""

    model = MagicMock(spec=AutoModelForMaskedLM)
    model.config = MagicMock()
    model.config._name_or_path = "test-model"
    model.to = MagicMock(return_value=model)
    
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token_id = 0
    
    def mock_tokenize(text, max_length=None, padding=None, truncation=None, return_tensors=None):
        result = MagicMock()
        result.input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        result.attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        return result
    
    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize
    
    def mock_getitem(key):
        if key == "input_ids":
            return torch.tensor([1, 2, 3])
        elif key == "attention_mask":
            return torch.tensor([1, 1, 1])
        elif key == "labels":
            return torch.tensor([1, 2, 3])
        return MagicMock()
    
    result = MagicMock()
    result.__getitem__ = mock_getitem
    result.squeeze.return_value = torch.tensor([1, 2, 3])
    result.input_ids = torch.tensor([[1, 2, 3, 0, 0]])
    result.attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    
    tokenizer.return_value = result
    
    outputs = MagicMock()
    outputs.loss = torch.tensor(0.5)
    model.return_value = outputs
    
    return model, tokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for model saving/loading."""

    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)


class TestBiasMitigationDataset:
    def test_dataset_initialization(self, mock_dataset, mock_model_tokenizer):
        """Test that the dataset is initialized correctly."""
        _, tokenizer = mock_model_tokenizer
        data = mock_dataset["gender"]
        
        dataset = BiasMitigationDataset(data, tokenizer)
        
        assert len(dataset) == len(data)
        assert dataset.tokenizer == tokenizer
        assert dataset.max_length == 128
    
    @patch("src.fine_tuning.BiasMitigationDataset.__getitem__")
    def test_dataset_getitem(self, mock_getitem, mock_dataset, mock_model_tokenizer):
        """Test that __getitem__ returns the correct format."""
        _, tokenizer = mock_model_tokenizer
        data = mock_dataset["gender"]
        
        mock_getitem.return_value = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3])
        }
        
        dataset = BiasMitigationDataset(data, tokenizer)
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)


class TestModelFineTuner:
    def test_initialization(self, mock_model_tokenizer, temp_dir):
        """Test that the fine-tuner is initialized correctly."""
        model, tokenizer = mock_model_tokenizer
        
        fine_tuner = ModelFineTuner(model, tokenizer, device="mps", output_dir=temp_dir)
        
        assert fine_tuner.model == model
        assert fine_tuner.tokenizer == tokenizer
        assert fine_tuner.device == "mps"
        assert fine_tuner.output_dir == temp_dir
        assert os.path.exists(temp_dir)
        
    def test_prepare_counter_stereotypical_data(self, mock_dataset, mock_model_tokenizer, temp_dir):
        """Test that counter-stereotypical data is prepared correctly."""
        model, tokenizer = mock_model_tokenizer
        
        fine_tuner = ModelFineTuner(model, tokenizer, device="mps", output_dir=temp_dir)
        training_data = fine_tuner.prepare_counter_stereotypical_data(mock_dataset)
        
        assert len(training_data) == 2
        assert all(ex["label"] == "anti-stereotype" for ex in training_data)
        
    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_model(self, mock_makedirs, mock_save, mock_model_tokenizer, temp_dir):
        """Test that models are saved correctly."""
        model, tokenizer = mock_model_tokenizer
        
        model.save_pretrained = MagicMock()
        tokenizer.save_pretrained = MagicMock()
        
        fine_tuner = ModelFineTuner(model, tokenizer, device="mps", output_dir=temp_dir)
        save_path = fine_tuner.save_model(suffix="test")
        
        assert "test-model_test" in save_path
        assert model.save_pretrained.called
        assert tokenizer.save_pretrained.called
        
    @patch("src.fine_tuning.ModelFineTuner.fine_tune")
    def test_fine_tune(self, mock_fine_tune, mock_model_tokenizer, mock_dataset, temp_dir):
        """Test the fine-tuning process with a mock implementation."""
        model, tokenizer = mock_model_tokenizer
        
        mock_fine_tune.return_value = {
            "train_loss_per_epoch": [0.5],
            "model_name": "test-model",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-5,
            "training_examples": 2
        }
        
        fine_tuner = ModelFineTuner(model, tokenizer, device="mps", output_dir=temp_dir)
        training_data = fine_tuner.prepare_counter_stereotypical_data(mock_dataset)
        
        stats = fine_tuner.fine_tune(
            training_data=training_data,
            epochs=1,
            batch_size=1,
            save_every_epoch=False
        )
        
        assert "train_loss_per_epoch" in stats
        assert "model_name" in stats
        assert "epochs" in stats
        assert mock_fine_tune.called


@patch("src.fine_tuning.ModelFineTuner")
def test_fine_tune_model_function(mock_fine_tuner_class, mock_model_tokenizer, mock_dataset):
    """Test the convenience function for fine-tuning."""
    model, tokenizer = mock_model_tokenizer
    
    mock_fine_tuner = MagicMock()
    mock_fine_tuner.prepare_counter_stereotypical_data.return_value = []
    mock_fine_tuner.fine_tune.return_value = {"train_loss_per_epoch": [0.1]}
    mock_fine_tuner.model = model
    
    mock_fine_tuner_class.return_value = mock_fine_tuner
    
    result_model, result_tokenizer, stats = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        dataset=mock_dataset,
        device="mps",
        output_dir="temp"
    )

    assert result_model == model
    assert result_tokenizer == tokenizer
    assert "train_loss_per_epoch" in stats
    assert mock_fine_tuner.prepare_counter_stereotypical_data.called
    assert mock_fine_tuner.fine_tune.called


@patch("os.path.exists")
def test_load_fine_tuned_model(mock_exists, mock_model_tokenizer):
    """Test loading a fine-tuned model."""
    model, tokenizer = mock_model_tokenizer
    
    mock_exists.return_value = True
    
    with patch("transformers.AutoModelForMaskedLM.from_pretrained", return_value=model):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            loaded_model, loaded_tokenizer = load_fine_tuned_model(
                model_name="test-model",
                device="mps",
                models_dir="models"
            )
    
    assert loaded_model == model
    assert loaded_tokenizer == tokenizer
    assert mock_exists.called
