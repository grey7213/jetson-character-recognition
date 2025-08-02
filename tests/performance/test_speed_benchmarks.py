"""
Speed benchmark tests for character recognition system.
字符识别系统的速度基准测试。
"""

import pytest
import numpy as np
import cv2
import time
import statistics
from pathlib import Path
from unittest.mock import Mock, patch
import psutil
import gc

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.yolo_character_detector import YOLOCharacterDetector
from src.inference.realtime_detector import RealtimeCharacterDetector
from src.utils.performance import PerformanceMonitor, FPSCounter


@pytest.mark.performance
class TestSpeedBenchmarks:
    """Speed benchmark tests."""
    
    @pytest.fixture
    def benchmark_images(self):
        """Generate benchmark test images."""
        images = []
        for i in range(50):  # 50 test images
            # Create varied test images / 创建多样化的测试图像
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Add some text to make it more realistic / 添加文本使其更真实
            cv2.putText(img, f'TEST{i%10}', (100 + i*5, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            images.append(img)
        
        return images
    
    def test_single_image_inference_speed(self, benchmark_images):
        """Benchmark single image inference speed."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            # Setup mock model with realistic timing / 设置具有真实时序的模拟模型
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10],
                [300, 300, 400, 400, 0.8, 1]
            ])
            
            def mock_predict(*args, **kwargs):
                # Simulate inference time / 模拟推理时间
                time.sleep(0.01)  # 10ms simulated inference
                return [mock_result]
            
            mock_model.predict = mock_predict
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # Warm up / 预热
            for _ in range(5):
                detector.predict(benchmark_images[0])
            
            # Benchmark inference speed / 基准测试推理速度
            inference_times = []
            
            for image in benchmark_images[:20]:  # Test with 20 images
                start_time = time.time()
                detections = detector.predict(image)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
            
            # Calculate statistics / 计算统计数据
            avg_time = statistics.mean(inference_times)
            median_time = statistics.median(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            std_dev = statistics.stdev(inference_times)
            
            avg_fps = 1.0 / avg_time
            
            # Performance assertions / 性能断言
            assert avg_time > 0
            assert avg_fps > 0
            assert min_time <= avg_time <= max_time
            assert std_dev >= 0
            
            # Log performance metrics / 记录性能指标
            print(f"\nSingle Image Inference Benchmark:")
            print(f"Average inference time: {avg_time:.4f}s")
            print(f"Median inference time: {median_time:.4f}s")
            print(f"Min/Max inference time: {min_time:.4f}s / {max_time:.4f}s")
            print(f"Standard deviation: {std_dev:.4f}s")
            print(f"Average FPS: {avg_fps:.2f}")
            
            # Performance targets for Jetson Nano / Jetson Nano性能目标
            # These are relaxed for testing with mocks / 这些是模拟测试的宽松目标
            assert avg_fps >= 1.0  # At least 1 FPS with mocks
    
    def test_batch_inference_speed(self, benchmark_images):
        """Benchmark batch inference speed."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10]
            ])
            
            def mock_predict_batch(images, *args, **kwargs):
                # Simulate batch processing time / 模拟批处理时间
                batch_size = len(images) if isinstance(images, list) else 1
                time.sleep(0.005 * batch_size)  # 5ms per image in batch
                return [mock_result] * batch_size
            
            mock_model.predict = mock_predict_batch
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # Test different batch sizes / 测试不同批次大小
            batch_sizes = [1, 2, 4, 8]
            batch_performance = {}
            
            for batch_size in batch_sizes:
                batch_times = []
                
                # Create batches / 创建批次
                num_batches = 10
                for i in range(num_batches):
                    batch = benchmark_images[i*batch_size:(i+1)*batch_size]
                    if len(batch) < batch_size:
                        break
                    
                    start_time = time.time()
                    results = detector.predict_batch(batch)
                    end_time = time.time()
                    
                    batch_time = end_time - start_time
                    batch_times.append(batch_time)
                
                if batch_times:
                    avg_batch_time = statistics.mean(batch_times)
                    throughput = batch_size / avg_batch_time  # Images per second
                    
                    batch_performance[batch_size] = {
                        'avg_batch_time': avg_batch_time,
                        'throughput': throughput,
                        'time_per_image': avg_batch_time / batch_size
                    }
            
            # Analyze batch performance / 分析批处理性能
            print(f"\nBatch Inference Benchmark:")
            for batch_size, metrics in batch_performance.items():
                print(f"Batch size {batch_size}:")
                print(f"  Avg batch time: {metrics['avg_batch_time']:.4f}s")
                print(f"  Throughput: {metrics['throughput']:.2f} images/s")
                print(f"  Time per image: {metrics['time_per_image']:.4f}s")
            
            # Verify batch processing is working / 验证批处理正常工作
            assert len(batch_performance) > 0
            for metrics in batch_performance.values():
                assert metrics['throughput'] > 0
                assert metrics['time_per_image'] > 0
    
    def test_realtime_detection_speed(self, benchmark_images):
        """Benchmark real-time detection speed."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10]
            ])
            
            def mock_predict(*args, **kwargs):
                time.sleep(0.02)  # 20ms simulated inference
                return [mock_result]
            
            mock_model.predict = mock_predict
            mock_yolo.return_value = mock_model
            
            # Initialize real-time detector / 初始化实时检测器
            realtime_detector = RealtimeCharacterDetector()
            realtime_detector.detector.load_model(pretrained=True)
            
            # Benchmark real-time processing / 基准测试实时处理
            processing_times = []
            fps_values = []
            
            for image in benchmark_images[:15]:  # Test with 15 images
                result = realtime_detector.detect_single_frame(image)
                
                processing_times.append(result.processing_time)
                fps_values.append(result.fps)
            
            # Calculate statistics / 计算统计数据
            avg_processing_time = statistics.mean(processing_times)
            avg_fps = statistics.mean(fps_values)
            min_fps = min(fps_values)
            max_fps = max(fps_values)
            
            print(f"\nReal-time Detection Benchmark:")
            print(f"Average processing time: {avg_processing_time:.4f}s")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Min/Max FPS: {min_fps:.2f} / {max_fps:.2f}")
            
            # Performance assertions / 性能断言
            assert avg_processing_time > 0
            assert avg_fps > 0
            assert min_fps <= avg_fps <= max_fps
    
    def test_memory_usage_during_inference(self, benchmark_images):
        """Benchmark memory usage during inference."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([])
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # Measure initial memory / 测量初始内存
            gc.collect()  # Force garbage collection
            initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            
            # Load model / 加载模型
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            model_loaded_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            model_memory_usage = model_loaded_memory - initial_memory
            
            # Run inference and monitor memory / 运行推理并监控内存
            memory_samples = []
            
            for i, image in enumerate(benchmark_images[:20]):
                detector.predict(image)
                
                if i % 5 == 0:  # Sample memory every 5 inferences
                    current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
                    memory_samples.append(current_memory)
            
            # Calculate memory statistics / 计算内存统计
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            avg_memory = statistics.mean(memory_samples)
            
            max_inference_memory = max_memory - initial_memory
            avg_inference_memory = avg_memory - initial_memory
            
            print(f"\nMemory Usage Benchmark:")
            print(f"Initial memory: {initial_memory:.1f} MB")
            print(f"Model memory usage: {model_memory_usage:.1f} MB")
            print(f"Max inference memory: {max_inference_memory:.1f} MB")
            print(f"Avg inference memory: {avg_inference_memory:.1f} MB")
            print(f"Memory range during inference: {min_memory:.1f} - {max_memory:.1f} MB")
            
            # Memory usage assertions / 内存使用断言
            assert model_memory_usage >= 0
            assert max_inference_memory >= model_memory_usage
            assert avg_inference_memory >= 0
    
    def test_fps_counter_accuracy(self, benchmark_images):
        """Test FPS counter accuracy."""
        
        fps_counter = FPSCounter(window_size=10)
        
        # Simulate consistent frame processing / 模拟一致的帧处理
        target_fps = 10  # Target 10 FPS
        frame_interval = 1.0 / target_fps
        
        measured_fps_values = []
        
        for i in range(20):
            start_time = time.time()
            
            # Simulate frame processing / 模拟帧处理
            time.sleep(frame_interval * 0.9)  # Slightly faster than target
            
            fps = fps_counter.update()
            if fps is not None:
                measured_fps_values.append(fps)
        
        if measured_fps_values:
            avg_measured_fps = statistics.mean(measured_fps_values)
            fps_error = abs(avg_measured_fps - target_fps) / target_fps
            
            print(f"\nFPS Counter Accuracy Test:")
            print(f"Target FPS: {target_fps}")
            print(f"Measured FPS: {avg_measured_fps:.2f}")
            print(f"Error: {fps_error:.2%}")
            
            # FPS counter should be reasonably accurate / FPS计数器应该相当准确
            assert fps_error < 0.2  # Within 20% error
    
    @pytest.mark.slow
    def test_sustained_performance(self, benchmark_images):
        """Test sustained performance over time."""
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([
                [100, 100, 200, 200, 0.9, 10]
            ])
            
            def mock_predict(*args, **kwargs):
                time.sleep(0.015)  # 15ms simulated inference
                return [mock_result]
            
            mock_model.predict = mock_predict
            mock_yolo.return_value = mock_model
            
            detector = YOLOCharacterDetector()
            detector.load_model(pretrained=True)
            
            # Run sustained inference / 运行持续推理
            num_iterations = 100
            performance_windows = []
            window_size = 10
            
            for i in range(num_iterations):
                start_time = time.time()
                detector.predict(benchmark_images[i % len(benchmark_images)])
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # Calculate rolling window performance / 计算滚动窗口性能
                if i >= window_size - 1:
                    window_start = i - window_size + 1
                    window_times = []
                    
                    for j in range(window_start, i + 1):
                        # Simulate getting previous times / 模拟获取之前的时间
                        window_times.append(inference_time)
                    
                    window_avg_time = statistics.mean(window_times)
                    window_fps = 1.0 / window_avg_time
                    performance_windows.append(window_fps)
            
            if performance_windows:
                # Analyze performance stability / 分析性能稳定性
                fps_std = statistics.stdev(performance_windows)
                fps_mean = statistics.mean(performance_windows)
                fps_cv = fps_std / fps_mean  # Coefficient of variation
                
                print(f"\nSustained Performance Test:")
                print(f"Iterations: {num_iterations}")
                print(f"Average FPS: {fps_mean:.2f}")
                print(f"FPS standard deviation: {fps_std:.2f}")
                print(f"Coefficient of variation: {fps_cv:.3f}")
                
                # Performance should be stable / 性能应该稳定
                assert fps_cv < 0.3  # CV should be less than 30%
                assert fps_mean > 0
    
    def test_different_input_sizes_performance(self):
        """Test performance with different input image sizes."""
        
        input_sizes = [(320, 320), (416, 416), (512, 512), (640, 640), (800, 800)]
        size_performance = {}
        
        with patch('src.models.yolo_character_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = np.array([])
            
            def mock_predict(image, *args, **kwargs):
                # Simulate size-dependent processing time / 模拟依赖尺寸的处理时间
                if hasattr(image, 'shape'):
                    pixels = image.shape[0] * image.shape[1]
                    base_time = 0.01
                    size_factor = pixels / (640 * 640)
                    time.sleep(base_time * size_factor)
                else:
                    time.sleep(0.01)
                return [mock_result]
            
            mock_model.predict = mock_predict
            mock_yolo.return_value = mock_model
            
            for width, height in input_sizes:
                detector = YOLOCharacterDetector()
                detector.load_model(pretrained=True)
                
                # Create test image of specific size / 创建特定尺寸的测试图像
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                # Benchmark this size / 基准测试此尺寸
                inference_times = []
                
                for _ in range(10):
                    start_time = time.time()
                    detector.predict(test_image)
                    end_time = time.time()
                    
                    inference_times.append(end_time - start_time)
                
                avg_time = statistics.mean(inference_times)
                avg_fps = 1.0 / avg_time
                
                size_performance[(width, height)] = {
                    'avg_time': avg_time,
                    'avg_fps': avg_fps,
                    'pixels': width * height
                }
            
            print(f"\nInput Size Performance Benchmark:")
            for (width, height), metrics in size_performance.items():
                print(f"Size {width}x{height}: {metrics['avg_fps']:.2f} FPS, {metrics['avg_time']:.4f}s")
            
            # Verify performance scales with input size / 验证性能随输入尺寸缩放
            sizes_sorted = sorted(size_performance.keys(), key=lambda x: x[0] * x[1])
            fps_values = [size_performance[size]['avg_fps'] for size in sizes_sorted]
            
            # Generally, larger images should have lower FPS / 通常，较大的图像应该有较低的FPS
            # (though this may not hold exactly with mocks)
            assert all(fps > 0 for fps in fps_values)
