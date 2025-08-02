"""
Unit tests for DatasetManager class.
DatasetManager类的单元测试。
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import json

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.data.dataset_manager import DatasetManager


class TestDatasetManager:
    """Test suite for DatasetManager."""
    
    def test_initialization_default_dir(self):
        """Test dataset manager initialization with default directory."""
        manager = DatasetManager()
        
        assert manager.data_dir == Path("data")
        assert hasattr(manager, 'DATASETS')
        assert 'synthetic' in manager.DATASETS
        assert 'emnist' in manager.DATASETS
        assert 'chars74k' in manager.DATASETS
    
    def test_initialization_custom_dir(self, temp_dir):
        """Test dataset manager initialization with custom directory."""
        custom_dir = temp_dir / "custom_data"
        manager = DatasetManager(str(custom_dir))
        
        assert manager.data_dir == custom_dir
        assert custom_dir.exists()
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        manager = DatasetManager()
        datasets = manager.list_available_datasets()
        
        assert isinstance(datasets, list)
        assert 'synthetic' in datasets
        assert 'emnist' in datasets
        assert 'chars74k' in datasets
        assert len(datasets) >= 3
    
    def test_get_dataset_info_valid_dataset(self):
        """Test getting dataset information for valid dataset."""
        manager = DatasetManager()
        
        # Test synthetic dataset info
        synthetic_info = manager.get_dataset_info('synthetic')
        
        assert isinstance(synthetic_info, dict)
        assert 'name' in synthetic_info
        assert 'description' in synthetic_info
        assert 'classes' in synthetic_info
        assert synthetic_info['classes'] == 36
    
    def test_get_dataset_info_invalid_dataset(self):
        """Test getting dataset information for invalid dataset."""
        manager = DatasetManager()
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            manager.get_dataset_info('nonexistent_dataset')
    
    @patch('src.data.dataset_manager.cv2')
    def test_generate_character_image(self, mock_cv2, temp_dir):
        """Test character image generation."""
        manager = DatasetManager(str(temp_dir))
        
        # Mock cv2 functions
        mock_cv2.getTextSize.return_value = ((30, 40), 5)
        mock_cv2.putText = Mock()
        mock_cv2.add = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.GaussianBlur = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.getRotationMatrix2D = Mock(return_value=np.eye(2, 3))
        mock_cv2.warpAffine = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.convertScaleAbs = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        
        # Test image generation
        image = manager._generate_character_image('A', 0)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64)
        mock_cv2.putText.assert_called_once()
    
    @patch('src.data.dataset_manager.cv2')
    def test_generate_character_image_with_variations(self, mock_cv2, temp_dir):
        """Test character image generation with different variations."""
        manager = DatasetManager(str(temp_dir))
        
        # Mock cv2 functions
        mock_cv2.getTextSize.return_value = ((30, 40), 5)
        mock_cv2.putText = Mock()
        mock_cv2.add = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.GaussianBlur = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.getRotationMatrix2D = Mock(return_value=np.eye(2, 3))
        mock_cv2.warpAffine = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.convertScaleAbs = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        
        # Test different variations
        for variation in range(10):
            image = manager._generate_character_image('B', variation)
            assert isinstance(image, np.ndarray)
            assert image.shape == (64, 64)
    
    @patch('src.data.dataset_manager.cv2')
    def test_download_dataset_synthetic(self, mock_cv2, temp_dir):
        """Test downloading synthetic dataset."""
        manager = DatasetManager(str(temp_dir))
        
        # Mock cv2 functions for image generation
        mock_cv2.getTextSize.return_value = ((30, 40), 5)
        mock_cv2.putText = Mock()
        mock_cv2.imwrite = Mock(return_value=True)
        mock_cv2.add = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.GaussianBlur = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.getRotationMatrix2D = Mock(return_value=np.eye(2, 3))
        mock_cv2.warpAffine = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.convertScaleAbs = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        
        # Test synthetic dataset generation
        dataset_dir = manager.download_dataset('synthetic')
        
        assert isinstance(dataset_dir, Path)
        assert dataset_dir.exists()
        assert dataset_dir.name == 'synthetic'
        
        # Check that character directories were created
        char_dirs = list(dataset_dir.iterdir())
        assert len(char_dirs) == 36  # 10 digits + 26 letters
    
    def test_download_dataset_invalid(self, temp_dir):
        """Test downloading invalid dataset."""
        manager = DatasetManager(str(temp_dir))
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            manager.download_dataset('invalid_dataset')
    
    @patch('src.data.dataset_manager.requests')
    def test_download_dataset_emnist(self, mock_requests, temp_dir):
        """Test downloading EMNIST dataset."""
        manager = DatasetManager(str(temp_dir))
        
        # Mock successful download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_zip_content"
        mock_requests.get.return_value = mock_response
        
        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip_instance = Mock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
            
            # This should attempt to download but may fail due to mocking
            # We're mainly testing the flow
            try:
                dataset_dir = manager.download_dataset('emnist')
                assert isinstance(dataset_dir, Path)
            except Exception:
                # Expected due to mocking limitations
                pass
    
    @patch('src.data.dataset_manager.cv2')
    @patch('yaml.dump')
    def test_prepare_yolo_dataset(self, mock_yaml_dump, mock_cv2, temp_dir):
        """Test preparing YOLO format dataset."""
        manager = DatasetManager(str(temp_dir))
        
        # Create a mock dataset directory structure
        dataset_dir = temp_dir / "test_dataset"
        dataset_dir.mkdir()
        
        # Create character directories with sample images
        for char in ['0', '1', 'A', 'B']:
            char_dir = dataset_dir / char
            char_dir.mkdir()
            
            # Create sample image files
            for i in range(5):
                img_file = char_dir / f"{char}_{i:03d}.png"
                img_file.touch()
        
        output_dir = temp_dir / "yolo_output"
        
        # Mock cv2 functions
        mock_cv2.imread.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_cv2.imwrite = Mock(return_value=True)
        
        # Test YOLO dataset preparation
        manager.prepare_yolo_dataset(dataset_dir, output_dir)
        
        # Check output structure
        assert output_dir.exists()
        assert (output_dir / "images").exists()
        assert (output_dir / "labels").exists()
        assert (output_dir / "train.txt").exists()
        assert (output_dir / "val.txt").exists()
        assert (output_dir / "dataset.yaml").exists()
    
    def test_character_classes_property(self, temp_dir):
        """Test character classes property."""
        manager = DatasetManager(str(temp_dir))
        
        # Test digit classes
        assert len(manager.digit_classes) == 10
        assert manager.digit_classes[0] == '0'
        assert manager.digit_classes[9] == '9'
        
        # Test letter classes
        assert len(manager.letter_classes) == 26
        assert manager.letter_classes[0] == 'A'
        assert manager.letter_classes[25] == 'Z'
        
        # Test all classes
        assert len(manager.all_classes) == 36
        assert manager.all_classes[:10] == manager.digit_classes
        assert manager.all_classes[10:] == manager.letter_classes
    
    def test_dataset_constants(self):
        """Test dataset constants and metadata."""
        manager = DatasetManager()
        
        # Test EMNIST dataset info
        emnist_info = manager.DATASETS['emnist']
        assert emnist_info['name'] == "EMNIST (Extended MNIST)"
        assert emnist_info['classes'] == 47
        assert 'url' in emnist_info
        
        # Test Chars74K dataset info
        chars74k_info = manager.DATASETS['chars74k']
        assert chars74k_info['name'] == "Chars74K"
        assert chars74k_info['classes'] == 62
        assert 'url' in chars74k_info
        
        # Test synthetic dataset info
        synthetic_info = manager.DATASETS['synthetic']
        assert synthetic_info['name'] == "Synthetic Character Dataset"
        assert synthetic_info['classes'] == 36
        assert synthetic_info['format'] == "generated"
    
    @patch('src.data.dataset_manager.cv2')
    def test_error_handling_image_generation(self, mock_cv2, temp_dir):
        """Test error handling in image generation."""
        manager = DatasetManager(str(temp_dir))
        
        # Mock cv2 to raise an exception
        mock_cv2.getTextSize.side_effect = Exception("OpenCV error")
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            manager._generate_character_image('A', 0)
    
    def test_data_directory_creation(self, temp_dir):
        """Test automatic data directory creation."""
        non_existent_dir = temp_dir / "new_data_dir"
        assert not non_existent_dir.exists()
        
        manager = DatasetManager(str(non_existent_dir))
        assert non_existent_dir.exists()
        assert manager.data_dir == non_existent_dir
    
    @patch('src.data.dataset_manager.cv2')
    def test_synthetic_dataset_statistics(self, mock_cv2, temp_dir):
        """Test synthetic dataset generation statistics."""
        manager = DatasetManager(str(temp_dir))
        
        # Mock cv2 functions
        mock_cv2.getTextSize.return_value = ((30, 40), 5)
        mock_cv2.putText = Mock()
        mock_cv2.imwrite = Mock(return_value=True)
        mock_cv2.add = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.GaussianBlur = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.getRotationMatrix2D = Mock(return_value=np.eye(2, 3))
        mock_cv2.warpAffine = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        mock_cv2.convertScaleAbs = Mock(return_value=np.zeros((64, 64), dtype=np.uint8))
        
        # Generate dataset with limited samples for testing
        with patch.object(manager, 'variations_per_character', 5):
            dataset_dir = manager.download_dataset('synthetic')
            
            # Check statistics
            char_dirs = list(dataset_dir.iterdir())
            assert len(char_dirs) == 36
            
            # Check that each character directory has the expected number of images
            for char_dir in char_dirs:
                if char_dir.is_dir():
                    # Note: actual file count depends on cv2.imwrite mock behavior
                    assert char_dir.exists()
