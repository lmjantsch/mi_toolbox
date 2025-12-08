import pytest
import torch
import os
from unittest.mock import MagicMock, patch, ANY

from mi_toolbox.utils.data_types.transformer_cache import TransformerCache

class TestTransformerCache:

    @pytest.fixture
    def mock_config(self):
        """Creates a fake HuggingFace config object."""
        config = MagicMock()
        config.to_dict.return_value = {"hidden_size": 768, "vocab_size": 1000}
        return config

    @pytest.fixture
    def cache_with_tensors(self):
        """Returns a TransformerCache populated with lists of tensors."""
        tc = TransformerCache()
        # Create a list of 3 tensors, shape (2, 2)
        tensors = [torch.randn(2, 2) for _ in range(3)]
        tc.attach("activations", tensors)
        tc.attach("labels", [1, 0, 1]) # Regular list, should be ignored by stack
        return tc

    # --- Initialization Tests ---

    def test_init_plain(self):
        tc = TransformerCache()
        assert tc.model_config is None
        assert len(tc) == 0

    def test_init_with_config_object(self, mock_config):
        tc = TransformerCache(model_config=mock_config)
        assert tc.model_config == mock_config

    @patch("mi_toolbox.utils.data_types.transformer_cache.AutoConfig")
    def test_init_with_model_id(self, mock_auto_config):
        """Test that passing model_id triggers AutoConfig.from_pretrained."""
        # Setup mock return value
        fake_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = fake_config

        tc = TransformerCache(model_id="gpt2")
        
        # Verify it called the library correctly
        mock_auto_config.from_pretrained.assert_called_once_with("gpt2")
        assert tc.model_config == fake_config

    # --- Stack Tensors Tests ---

    def test_stack_tensors_simple(self, cache_with_tensors):
        """Test standard stacking (padding=False)."""
        # Pre-check: it's a list
        assert isinstance(cache_with_tensors["activations"], list)
        
        cache_with_tensors.stack_tensors(padding=False)
        
        # Post-check: it's a tensor
        stacked = cache_with_tensors["activations"]
        assert isinstance(stacked, torch.Tensor)
        assert stacked.shape == (3, 2, 2) # (Batch, Dim1, Dim2)
        
        # Ensure non-tensor lists are untouched
        assert isinstance(cache_with_tensors["labels"], list)

    @patch("mi_toolbox.utils.data_types.transformer_cache.max_pad_sequence")
    def test_stack_tensors_padding(self, mock_max_pad, cache_with_tensors):
        """
        Test that padding=True delegates to the max_pad_sequence utility.
        We mock max_pad_sequence because we don't have its source code here.
        """
        # Setup mock to return a dummy tensor
        mock_max_pad.return_value = torch.tensor([1, 2, 3])
        
        cache_with_tensors.stack_tensors(padding=True)
        
        # Verify our specific utility function was called
        mock_max_pad.assert_called_once()
        # Verify the data was actually replaced by the mock result
        assert torch.equal(cache_with_tensors["activations"], torch.tensor([1, 2, 3]))

    # --- Save / Load Tests ---

    def test_save_calls_config_save(self, tmp_path, mock_config):
        """Test that saving the object also saves the config."""
        tc = TransformerCache(model_config=mock_config)
        tc.attach("dummy", [1, 2, 3])
        
        save_dir = tmp_path / "test_cache"
        tc.save(str(save_dir))
        
        # 1. Verify DataDict saved the data (inherited behavior check)
        # We assume DataDict._save works, checking if file exists is enough proxy
        assert (save_dir / "dummy.json").exists()
        
        # 2. Verify config.save_pretrained was called
        mock_config.save_pretrained.assert_called_once_with(str(save_dir))

    @patch("mi_toolbox.utils.data_types.transformer_cache.AutoConfig")
    def test_load_restores_config(self, mock_auto_config, tmp_path):
        """Test that loading attempts to reload the config."""
        save_dir = tmp_path / "load_test"
        save_dir.mkdir()
        
        # Create dummy data file so DataDict._load finds something
        with open(save_dir / "test_col.json", "w") as f:
            f.write("[1, 2, 3]")
            
        # Setup mock config to return when loaded
        fake_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = fake_config
        
        # Run Load
        loaded_tc = TransformerCache.load(str(save_dir), keys=["test_col"])
        
        # Verify AutoConfig looked for config in that directory
        mock_auto_config.from_pretrained.assert_called_once_with(str(save_dir))
        
        # Verify object state
        assert loaded_tc.model_config == fake_config
        assert loaded_tc["test_col"] == [1, 2, 3]

    @patch("mi_toolbox.utils.data_types.transformer_cache.AutoConfig")
    def test_load_handles_missing_config(self, mock_auto_config, tmp_path):
        """Test graceful failure if config is missing."""
        save_dir = tmp_path / "missing_conf"
        save_dir.mkdir()
        
        # Simulate OSError (file not found) from transformers
        mock_auto_config.from_pretrained.side_effect = OSError("Config not found")
        
        # Should not crash
        loaded_tc = TransformerCache.load(str(save_dir), keys=[])
        
        assert loaded_tc.model_config is None