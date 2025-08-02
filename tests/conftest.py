"""
Pytest configuration and fixtures for Jetson Character Recognition tests.
Jetson字符识别测试的Pytest配置和固件。
"""

import pytest
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import yaml
import json

# Test markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "jetson: Jetson-specific tests")
    config.addinivalue_line("markers", "gpu: GPU-required tests")


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_images_dir(test_data_dir):
    """Provide path to sample images directory."""
    return test_data_dir / "sample_images"


@pytest.fixture(scope="session")
def temp_dir():
    """Provide temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_character_image():
    """Generate a sample character image for testing."""
    # Create a simple test image with character 'A'
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.putText(img, 'A', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    return img


@pytest.fixture
def sample_multi_character_image():
    """Generate a sample image with multiple characters."""
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img.fill(255)  # White background
    
    # Add multiple characters
    cv2.putText(img, 'ABC123', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    return img


@pytest.fixture
def sample_detection_result():
    """Generate sample detection result for testing."""
    return [
        {
            'class_id': 10,  # 'A'
            'class_name': 'A',
            'confidence': 0.95,
            'bbox': [100, 100, 150, 150],  # x1, y1, x2, y2
            'center': [125, 125],
            'area': 2500
        },
        {
            'class_id': 1,   # '1'
            'class_name': '1',
            'confidence': 0.88,
            'bbox': [200, 100, 240, 150],
            'center': [220, 125],
            'area': 2000
        }
    ]


@pytest.fixture
def mock_model_config():
    """Provide mock model configuration."""
    return {
        'model': {
            'name': 'yolov8n',
            'input_size': [640, 640],
            'num_classes': 36,
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001
        },
        'jetson': {
            'use_tensorrt': True,
            'fp16': True,
            'max_batch_size': 1
        }
    }


@pytest.fixture
def mock_camera_config():
    """Provide mock camera configuration."""
    return {
        'camera': {
            'type': 'usb',
            'device_id': 0
        },
        'capture': {
            'width': 1280,
            'height': 720,
            'fps': 30,
            'buffer_size': 1
        },
        'preprocessing': {
            'resize_width': 640,
            'resize_height': 640,
            'normalize': True
        }
    }


@pytest.fixture
def character_classes():
    """Provide character class definitions."""
    digits = [str(i) for i in range(10)]
    letters = [chr(ord('A') + i) for i in range(26)]
    return digits + letters


@pytest.fixture
def mock_dataset_info():
    """Provide mock dataset information."""
    return {
        'name': 'test_dataset',
        'total_images': 1000,
        'num_classes': 36,
        'train_images': 800,
        'val_images': 200,
        'class_distribution': {str(i): 28 for i in range(36)}
    }


@pytest.fixture
def sample_yolo_annotation():
    """Generate sample YOLO format annotation."""
    return [
        "10 0.5 0.5 0.2 0.3",  # class_id center_x center_y width height
        "1 0.3 0.4 0.15 0.25"
    ]


@pytest.fixture
def mock_training_results():
    """Provide mock training results."""
    return {
        'epochs': 100,
        'final_loss': 0.045,
        'final_accuracy': 0.942,
        'training_time': 3600,  # seconds
        'best_epoch': 85,
        'metrics': {
            'precision': 0.94,
            'recall': 0.93,
            'f1_score': 0.935,
            'map50': 0.945
        }
    }


@pytest.fixture
def mock_performance_metrics():
    """Provide mock performance metrics."""
    return {
        'inference_time': 0.067,  # seconds
        'preprocessing_time': 0.005,
        'postprocessing_time': 0.003,
        'total_time': 0.075,
        'fps': 13.3,
        'memory_usage': 1200,  # MB
        'gpu_utilization': 85,  # %
        'power_consumption': 8.5  # W
    }


@pytest.fixture
def create_test_config_file(temp_dir):
    """Factory to create test configuration files."""
    def _create_config(config_data: Dict[str, Any], filename: str = "test_config.yaml"):
        config_path = temp_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        return config_path
    return _create_config


@pytest.fixture
def create_test_image_file(temp_dir):
    """Factory to create test image files."""
    def _create_image(image_data: np.ndarray, filename: str = "test_image.jpg"):
        image_path = temp_dir / filename
        cv2.imwrite(str(image_path), image_data)
        return image_path
    return _create_image


@pytest.fixture
def create_test_annotation_file(temp_dir):
    """Factory to create test annotation files."""
    def _create_annotation(annotations: List[str], filename: str = "test_annotation.txt"):
        annotation_path = temp_dir / filename
        with open(annotation_path, 'w') as f:
            f.write('\n'.join(annotations))
        return annotation_path
    return _create_annotation


@pytest.fixture
def mock_jetson_info():
    """Provide mock Jetson system information."""
    return {
        'is_jetson': True,
        'jetpack_version': '4.6',
        'cuda_version': '10.2',
        'cpu_count': 4,
        'memory_total': 4096,  # MB
        'memory_available': 2048,
        'gpu_name': 'NVIDIA Tegra X1',
        'gpu_memory': 512,  # MB
        'temperature': 45.5,  # Celsius
        'power_mode': 0  # Max performance
    }


@pytest.fixture
def sample_benchmark_data():
    """Provide sample benchmark data."""
    return {
        'model_name': 'yolov8n_character',
        'test_date': '2024-01-20',
        'hardware': 'Jetson Nano 4GB',
        'results': {
            'accuracy': {
                'precision': 0.942,
                'recall': 0.938,
                'f1_score': 0.940,
                'map50': 0.945
            },
            'speed': {
                'avg_inference_time_ms': 67,
                'fps': 15,
                'throughput_imgs_per_sec': 14.9
            },
            'memory': {
                'peak_usage_mb': 1200,
                'average_usage_mb': 1150,
                'gpu_memory_mb': 450
            }
        }
    }


@pytest.fixture
def mock_camera_frame():
    """Generate mock camera frame."""
    # Create a realistic camera frame with some characters
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Add some text to make it more realistic
    cv2.putText(frame, 'TEST', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, '123', (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    return frame


@pytest.fixture
def create_synthetic_dataset(temp_dir):
    """Factory to create synthetic test dataset."""
    def _create_dataset(num_classes: int = 5, images_per_class: int = 10):
        dataset_dir = temp_dir / "synthetic_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        characters = ['0', '1', '2', 'A', 'B'][:num_classes]
        
        for char in characters:
            char_dir = dataset_dir / char
            char_dir.mkdir(exist_ok=True)
            
            for i in range(images_per_class):
                # Create simple character image
                img = np.zeros((64, 64), dtype=np.uint8)
                cv2.putText(img, char, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
                
                # Add some variation
                if i % 2 == 0:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                
                img_path = char_dir / f"{char}_{i:03d}.png"
                cv2.imwrite(str(img_path), img)
        
        return dataset_dir
    
    return _create_dataset


# Pytest hooks for test collection and execution
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "tensorrt" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark Jetson tests
        if "jetson" in item.name:
            item.add_marker(pytest.mark.jetson)


# Skip conditions for different environments
def pytest_runtest_setup(item):
    """Setup conditions for running tests."""
    # Skip GPU tests if CUDA is not available
    if "gpu" in [mark.name for mark in item.iter_markers()]:
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip Jetson tests if not on Jetson hardware
    if "jetson" in [mark.name for mark in item.iter_markers()]:
        import os
        if not os.path.exists('/etc/nv_tegra_release'):
            pytest.skip("Not running on Jetson hardware")
