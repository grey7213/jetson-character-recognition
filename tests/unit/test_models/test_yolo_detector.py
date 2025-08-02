"""
Unit tests for YOLOCharacterDetector class.
YOLOCharacterDetector类的单元测试。
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import cv2

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector


class TestYOLOCharacterDetector:
    """Test suite for YOLOCharacterDetector."""
    
    def test_initialization_default_config(self, mock_model_config):
        """Test detector initialization with default configuration."""
        with patch('src.models.yolo_character_detector.config_loader') as mock_config:
            mock_config.get_model_config.return_value = mock_model_config
            
            detector = YOLOCharacterDetector()
            
            assert detector.config == mock_model_config
            assert detector.model is None
            assert len(detector.all_classes) == 36
            assert detector.digit_classes == [str(i) for i in range(10)]
            assert len(detector.letter_classes) == 26
    
    def test_initialization_custom_config(self, mock_model_config):
        """Test detector initialization with custom configuration."""
        custom_config = mock_model_config.copy()
        custom_config['model']['confidence_threshold'] = 0.7
        
        detector = YOLOCharacterDetector(custom_config)
        
        assert detector.config == custom_config
        assert detector.config['model']['confidence_threshold'] == 0.7
    
    def test_character_classes_mapping(self):
        """Test character class mapping is correct."""
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            
            # Test digit classes
            for i, digit in enumerate(detector.digit_classes):
                assert digit == str(i)
                assert detector.class_names[i] == str(i)
            
            # Test letter classes
            for i, letter in enumerate(detector.letter_classes):
                expected_letter = chr(ord('A') + i)
                assert letter == expected_letter
                assert detector.class_names[i + 10] == expected_letter
    
    def test_get_device_cuda_available(self):
        """Test device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('src.models.yolo_character_detector.config_loader'):
                detector = YOLOCharacterDetector()
                device = detector._get_device()
                assert device == "cuda"
    
    def test_get_device_cuda_not_available(self):
        """Test device selection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('src.models.yolo_character_detector.config_loader'):
                detector = YOLOCharacterDetector()
                device = detector._get_device()
                assert device == "cpu"
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_load_model_pretrained(self, mock_yolo_class):
        """Test loading pretrained model."""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            mock_yolo_class.assert_called_once_with("yolov8n.pt")
            assert detector.model == mock_model
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_load_model_from_path(self, mock_yolo_class):
        """Test loading model from file path."""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        model_path = "path/to/model.pt"
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(model_path)
            
            mock_yolo_class.assert_called_once_with(model_path)
            assert detector.model == mock_model
    
    def test_load_model_invalid_path(self):
        """Test loading model with invalid path."""
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            
            with pytest.raises(Exception):
                detector.load_model("nonexistent/path.pt")
    
    def test_get_model_info_no_model(self):
        """Test getting model info when no model is loaded."""
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            
            info = detector.get_model_info()
            
            assert "status" in info
            assert "No model loaded" in info["status"]
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_get_model_info_with_model(self, mock_yolo_class):
        """Test getting model info when model is loaded."""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [torch.randn(100), torch.randn(200)]
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            info = detector.get_model_info()
            
            assert "num_classes" in info
            assert "device" in info
            assert "parameters" in info
            assert info["num_classes"] == 36
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_predict_no_model(self, mock_yolo_class):
        """Test prediction without loaded model."""
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            with pytest.raises(ValueError, match="Model not loaded"):
                detector.predict(test_image)
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_predict_with_model(self, mock_yolo_class, sample_character_image):
        """Test prediction with loaded model."""
        # Setup mock model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = torch.tensor([
            [100, 100, 200, 200, 0.9, 10],  # x1, y1, x2, y2, conf, class
            [300, 300, 400, 400, 0.8, 1]
        ])
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # Convert single channel to 3 channel
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            detections = detector.predict(test_image)
            
            assert len(detections) == 2
            assert detections[0]['class_name'] == 'A'  # class 10 = 'A'
            assert detections[1]['class_name'] == '1'  # class 1 = '1'
            assert detections[0]['confidence'] == 0.9
            assert detections[1]['confidence'] == 0.8
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_predict_batch(self, mock_yolo_class, sample_character_image):
        """Test batch prediction."""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # Mock the predict method to return empty results
            detector.predict = Mock(return_value=[])
            
            test_images = [sample_character_image, sample_character_image]
            results = detector.predict_batch(test_images)
            
            assert len(results) == 2
            assert detector.predict.call_count == 2
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_train_method(self, mock_yolo_class):
        """Test model training method."""
        mock_model = Mock()
        mock_train_result = Mock()
        mock_train_result.results_dict = {'fitness': 0.95}
        mock_model.train.return_value = mock_train_result
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            dataset_yaml = "path/to/dataset.yaml"
            results = detector.train(dataset_yaml, epochs=10)
            
            mock_model.train.assert_called_once()
            assert results == mock_train_result
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_export_model(self, mock_yolo_class):
        """Test model export functionality."""
        mock_model = Mock()
        mock_model.export.return_value = "exported_model.onnx"
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            output_path = detector.export_model("model.onnx", format="onnx")
            
            mock_model.export.assert_called_once_with(format="onnx")
            assert output_path == "exported_model.onnx"
    
    def test_export_model_no_model(self):
        """Test model export without loaded model."""
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            
            with pytest.raises(ValueError, match="Model not loaded"):
                detector.export_model("model.onnx")
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_predict_with_custom_thresholds(self, mock_yolo_class, sample_character_image):
        """Test prediction with custom confidence and NMS thresholds."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = torch.tensor([])  # Empty results
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            detector.predict(test_image, confidence=0.7, nms_threshold=0.3)
            
            # Verify that custom thresholds were passed to model.predict
            mock_model.predict.assert_called_once()
            call_args = mock_model.predict.call_args
            assert call_args[1]['conf'] == 0.7
            assert call_args[1]['iou'] == 0.3
    
    def test_class_name_mapping_edge_cases(self):
        """Test edge cases in class name mapping."""
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            
            # Test boundary cases
            assert detector.class_names[0] == '0'
            assert detector.class_names[9] == '9'
            assert detector.class_names[10] == 'A'
            assert detector.class_names[35] == 'Z'
            
            # Test invalid class IDs
            assert detector.class_names.get(36) is None
            assert detector.class_names.get(-1) is None
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_prediction_error_handling(self, mock_yolo_class, sample_character_image):
        """Test error handling during prediction."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            detections = detector.predict(test_image)
            
            # Should return empty list on error
            assert detections == []
    
    @patch('src.models.yolo_character_detector.YOLO')
    def test_training_error_handling(self, mock_yolo_class):
        """Test error handling during training."""
        mock_model = Mock()
        mock_model.train.side_effect = Exception("Training failed")
        mock_yolo_class.return_value = mock_model
        
        with patch('src.models.yolo_character_detector.config_loader'):
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            with pytest.raises(Exception, match="Training failed"):
                detector.train("dataset.yaml", epochs=10)
