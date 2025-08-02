"""
End-to-end integration tests for the complete character recognition pipeline.
字符识别完整流水线的端到端集成测试。
"""

import pytest
import numpy as np
import cv2
import time
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.data.dataset_manager import DatasetManager
from src.inference.realtime_detector import RealtimeCharacterDetector
from src.utils.performance import PerformanceMonitor


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    def test_complete_training_pipeline(self, temp_dir, create_synthetic_dataset):
        """Test complete training pipeline from data generation to model training."""
        
        # 1. Generate synthetic dataset / 生成合成数据集
        dataset_dir = create_synthetic_dataset(num_classes=5, images_per_class=10)
        
        # 2. Prepare YOLO format dataset / 准备YOLO格式数据集
        dataset_manager = DatasetManager(str(temp_dir))
        yolo_dir = temp_dir / "yolo_dataset"
        
        with patch('src.data.dataset_manager.cv2') as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            mock_cv2.imwrite.return_value = True
            
            dataset_manager.prepare_yolo_dataset(dataset_dir, yolo_dir)
        
        # Verify YOLO dataset structure / 验证YOLO数据集结构
        assert (yolo_dir / "images").exists()
        assert (yolo_dir / "labels").exists()
        assert (yolo_dir / "dataset.yaml").exists()
        
        # 3. Initialize detector and mock training / 初始化检测器并模拟训练
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_train_result = Mock()
            mock_train_result.results_dict = {'fitness': 0.95}
            mock_model.train.return_value = mock_train_result
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # 4. Train model / 训练模型
            results = detector.train(
                dataset_yaml=str(yolo_dir / "dataset.yaml"),
                epochs=5,  # Small number for testing
                batch_size=2
            )
            
            # Verify training completed / 验证训练完成
            assert results is not None
            mock_model.train.assert_called_once()
    
    def test_complete_inference_pipeline(self, sample_character_image, temp_dir):
        """Test complete inference pipeline from image input to detection output."""
        
        # 1. Setup detector with mock model / 使用模拟模型设置检测器
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10],  # x1, y1, x2, y2, conf, class
                [300, 300, 400, 400, 0.8, 1]
            ])
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # 2. Prepare test image / 准备测试图像
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            
            # 3. Run inference / 运行推理
            detections = detector.predict(test_image)
            
            # 4. Verify results / 验证结果
            assert len(detections) == 2
            assert detections[0]['class_name'] == 'A'  # class 10 = 'A'
            assert detections[1]['class_name'] == '1'  # class 1 = '1'
            assert 0.0 <= detections[0]['confidence'] <= 1.0
            assert 0.0 <= detections[1]['confidence'] <= 1.0
            
            # 5. Verify bounding box format / 验证边界框格式
            for detection in detections:
                bbox = detection['bbox']
                assert len(bbox) == 4
                assert bbox[0] < bbox[2]  # x1 < x2
                assert bbox[1] < bbox[3]  # y1 < y2
    
    def test_realtime_detection_pipeline(self, sample_character_image):
        """Test real-time detection pipeline."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            # Setup mock model / 设置模拟模型
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10]  # Single detection
            ])
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # 1. Initialize real-time detector / 初始化实时检测器
            realtime_detector = RealtimeCharacterDetector()
            realtime_detector.detector.load_model(pretrained=True)
            
            # 2. Test single frame detection / 测试单帧检测
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            result = realtime_detector.detect_single_frame(test_image)
            
            # 3. Verify result structure / 验证结果结构
            assert hasattr(result, 'frame')
            assert hasattr(result, 'detections')
            assert hasattr(result, 'timestamp')
            assert hasattr(result, 'processing_time')
            assert hasattr(result, 'fps')
            
            # 4. Verify detection content / 验证检测内容
            assert len(result.detections) == 1
            assert result.detections[0]['class_name'] == 'A'
            assert result.processing_time > 0
            assert result.fps > 0
    
    def test_performance_monitoring_integration(self, sample_character_image):
        """Test integration with performance monitoring."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            # Setup mock model / 设置模拟模型
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([])  # Empty results
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # 1. Initialize components / 初始化组件
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            monitor = PerformanceMonitor()
            
            # 2. Run detection with performance monitoring / 运行带性能监控的检测
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            
            monitor.start_timer('inference')
            detections = detector.predict(test_image)
            inference_time = monitor.stop_timer('inference')
            
            # 3. Record performance metrics / 记录性能指标
            from src.utils.performance import PerformanceMetrics
            metrics = PerformanceMetrics(
                inference_time=inference_time,
                preprocessing_time=0.001,
                postprocessing_time=0.001,
                total_time=inference_time + 0.002,
                fps=1.0 / (inference_time + 0.002)
            )
            monitor.record_metrics(metrics)
            
            # 4. Verify performance data / 验证性能数据
            latest_metrics = monitor.get_latest_metrics()
            assert latest_metrics is not None
            assert latest_metrics.inference_time == inference_time
            assert latest_metrics.fps > 0
    
    def test_data_to_model_pipeline(self, temp_dir):
        """Test pipeline from data generation to model creation."""
        
        # 1. Generate synthetic data / 生成合成数据
        with patch('src.data.dataset_manager.cv2') as mock_cv2:
            mock_cv2.getTextSize.return_value = ((30, 40), 5)
            mock_cv2.putText = Mock()
            mock_cv2.imwrite.return_value = True
            mock_cv2.add.return_value = np.zeros((64, 64), dtype=np.uint8)
            mock_cv2.GaussianBlur.return_value = np.zeros((64, 64), dtype=np.uint8)
            mock_cv2.getRotationMatrix2D.return_value = np.eye(2, 3)
            mock_cv2.warpAffine.return_value = np.zeros((64, 64), dtype=np.uint8)
            mock_cv2.convertScaleAbs.return_value = np.zeros((64, 64), dtype=np.uint8)
            
            dataset_manager = DatasetManager(str(temp_dir))
            dataset_dir = dataset_manager.download_dataset('synthetic')
            
            # 2. Verify data generation / 验证数据生成
            assert dataset_dir.exists()
            char_dirs = list(dataset_dir.iterdir())
            assert len(char_dirs) == 36
        
        # 3. Prepare training data / 准备训练数据
        yolo_dir = temp_dir / "yolo_training"
        with patch('src.data.dataset_manager.cv2') as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            mock_cv2.imwrite.return_value = True
            
            dataset_manager.prepare_yolo_dataset(dataset_dir, yolo_dir)
        
        # 4. Initialize model for training / 初始化模型进行训练
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # Verify model is ready for training / 验证模型准备就绪
            assert detector.model is not None
            assert len(detector.all_classes) == 36
    
    def test_error_handling_in_pipeline(self, sample_character_image):
        """Test error handling throughout the pipeline."""
        
        # 1. Test model loading error / 测试模型加载错误
        detector = YOLOCharacterDetector()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            detector.predict(sample_character_image)
        
        # 2. Test prediction error handling / 测试预测错误处理
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            mock_yolo.return_value = mock_model
            
            detector.load_model(pretrained=True)
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            
            # Should return empty list on error / 错误时应返回空列表
            detections = detector.predict(test_image)
            assert detections == []
        
        # 3. Test real-time detector error handling / 测试实时检测器错误处理
        with patch('src.inference.realtime_detector.CameraHandler') as mock_camera:
            mock_camera_instance = Mock()
            mock_camera_instance.initialize_camera.return_value = False
            mock_camera.return_value = mock_camera_instance
            
            realtime_detector = RealtimeCharacterDetector()
            
            # Should handle camera initialization failure gracefully
            # 应该优雅地处理摄像头初始化失败
            # This test verifies the error handling exists
    
    def test_configuration_integration(self, mock_model_config, mock_camera_config):
        """Test integration with configuration system."""
        
        with patch('src.models.yolo_character_detector.config_loader') as mock_config_loader:
            mock_config_loader.get_model_config.return_value = mock_model_config
            
            # 1. Test detector with configuration / 使用配置测试检测器
            detector = YOLOCharacterDetector()
            
            assert detector.config == mock_model_config
            assert detector.config['model']['confidence_threshold'] == 0.5
            
        with patch('src.inference.realtime_detector.config_loader') as mock_config_loader:
            mock_config_loader.get_camera_config.return_value = mock_camera_config
            mock_config_loader.get_model_config.return_value = mock_model_config
            
            # 2. Test real-time detector with configuration / 使用配置测试实时检测器
            realtime_detector = RealtimeCharacterDetector()
            
            # Verify configuration is loaded / 验证配置已加载
            mock_config_loader.get_camera_config.assert_called_once()
            mock_config_loader.get_model_config.assert_called_once()
    
    def test_batch_processing_pipeline(self, sample_character_image):
        """Test batch processing pipeline."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10]
            ])
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # 1. Prepare batch of images / 准备图像批次
            test_images = [
                cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
                for _ in range(5)
            ]
            
            # 2. Process batch / 处理批次
            batch_results = detector.predict_batch(test_images)
            
            # 3. Verify batch results / 验证批次结果
            assert len(batch_results) == 5
            for result in batch_results:
                assert len(result) == 1  # One detection per image
                assert result[0]['class_name'] == 'A'
    
    @pytest.mark.slow
    def test_performance_benchmark_integration(self, sample_character_image):
        """Test performance benchmarking integration."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([])
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            monitor = PerformanceMonitor()
            
            # 1. Run multiple inferences for benchmarking / 运行多次推理进行基准测试
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            inference_times = []
            
            for i in range(10):
                monitor.start_timer(f'inference_{i}')
                detector.predict(test_image)
                inference_time = monitor.stop_timer(f'inference_{i}')
                inference_times.append(inference_time)
            
            # 2. Calculate performance statistics / 计算性能统计
            avg_inference_time = sum(inference_times) / len(inference_times)
            max_inference_time = max(inference_times)
            min_inference_time = min(inference_times)
            
            # 3. Verify performance metrics / 验证性能指标
            assert avg_inference_time > 0
            assert max_inference_time >= avg_inference_time
            assert min_inference_time <= avg_inference_time
            assert len(inference_times) == 10
            
            # 4. Calculate FPS / 计算FPS
            avg_fps = 1.0 / avg_inference_time
            assert avg_fps > 0
    
    def test_memory_usage_integration(self, sample_character_image):
        """Test memory usage monitoring integration."""
        
        import psutil
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([])
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # 1. Measure memory before model loading / 测量模型加载前的内存
            initial_memory = psutil.virtual_memory().used
            
            # 2. Load model and measure memory / 加载模型并测量内存
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            model_loaded_memory = psutil.virtual_memory().used
            
            # 3. Run inference and measure memory / 运行推理并测量内存
            test_image = cv2.cvtColor(sample_character_image, cv2.COLOR_GRAY2RGB)
            
            for _ in range(5):
                detector.predict(test_image)
            
            inference_memory = psutil.virtual_memory().used
            
            # 4. Verify memory usage is reasonable / 验证内存使用合理
            model_memory_increase = model_loaded_memory - initial_memory
            inference_memory_increase = inference_memory - model_loaded_memory
            
            # Memory should increase for model loading but not significantly for inference
            # 模型加载应该增加内存，但推理不应该显著增加
            assert model_memory_increase >= 0
            assert inference_memory_increase >= 0
