# Performance Optimization Guide / 性能优化指南

## Overview / 概述

This guide provides comprehensive performance optimization strategies for the Jetson Character Recognition system, focusing on maximizing inference speed, minimizing memory usage, and optimizing power consumption on Jetson Nano hardware.

本指南为Jetson字符识别系统提供全面的性能优化策略，专注于最大化推理速度、最小化内存使用和优化Jetson Nano硬件上的功耗。

## Table of Contents / 目录

1. [Hardware Optimization / 硬件优化](#hardware-optimization)
2. [Model Optimization / 模型优化](#model-optimization)
3. [Inference Optimization / 推理优化](#inference-optimization)
4. [Memory Optimization / 内存优化](#memory-optimization)
5. [Power Management / 功耗管理](#power-management)
6. [Benchmarking / 基准测试](#benchmarking)
7. [Troubleshooting / 故障排除](#troubleshooting)

## Hardware Optimization / 硬件优化

### Jetson Nano Configuration / Jetson Nano配置

#### Power Mode Settings / 功耗模式设置

```bash
# Check current power mode / 检查当前功耗模式
sudo nvpmodel -q

# Set maximum performance mode (10W) / 设置最大性能模式（10W）
sudo nvpmodel -m 0

# Set 5W power mode for battery operation / 设置5W功耗模式用于电池操作
sudo nvpmodel -m 1

# Maximize CPU clocks / 最大化CPU时钟频率
sudo jetson_clocks
```

#### Memory Configuration / 内存配置

```bash
# Add swap file for additional memory / 添加交换文件以获得额外内存
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent / 使交换文件永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optimize memory settings / 优化内存设置
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
```

#### GPU Optimization / GPU优化

```bash
# Check GPU status / 检查GPU状态
nvidia-smi

# Set GPU performance mode / 设置GPU性能模式
sudo nvidia-smi -pm 1

# Set maximum GPU clocks / 设置最大GPU时钟频率
sudo nvidia-smi -ac 2600,1300
```

### Cooling and Thermal Management / 散热和热管理

```python
from src.utils.jetson_utils import jetson_monitor

# Monitor temperature / 监控温度
temp_info = jetson_monitor.get_temperature_info()
print(f"CPU Temperature: {temp_info['cpu']}°C")
print(f"GPU Temperature: {temp_info['gpu']}°C")

# Thermal throttling check / 热节流检查
if temp_info['cpu'] > 80:
    print("Warning: High CPU temperature detected")
    # Implement cooling strategies / 实施散热策略
```

## Model Optimization / 模型优化

### Model Selection / 模型选择

| Model | Size | FPS (Jetson Nano) | Accuracy | Memory | Use Case |
|-------|------|-------------------|----------|---------|----------|
| YOLOv8n | 6MB | 15-20 FPS | 94.2% | 1.2GB | Real-time applications |
| YOLOv8s | 22MB | 8-12 FPS | 96.8% | 1.8GB | High accuracy requirements |
| Custom Nano | 3MB | 20-25 FPS | 92.5% | 0.8GB | Ultra-fast inference |

### TensorRT Optimization / TensorRT优化

```python
from src.models.tensorrt_optimizer import TensorRTOptimizer

# Initialize optimizer / 初始化优化器
optimizer = TensorRTOptimizer()

# Optimize model for FP16 precision / 为FP16精度优化模型
optimized_model = optimizer.optimize_model(
    input_model="models/yolov8n_character.pt",
    output_path="models/optimized/character_fp16.engine",
    precision="fp16",
    max_batch_size=1,
    max_workspace_size=1024*1024*1024  # 1GB
)

# Optimize for INT8 precision (requires calibration) / 为INT8精度优化（需要校准）
int8_model = optimizer.optimize_model(
    input_model="models/yolov8n_character.pt",
    output_path="models/optimized/character_int8.engine",
    precision="int8",
    calibration_dataset="data/calibration"
)
```

### Model Quantization / 模型量化

```python
# Post-training quantization / 训练后量化
from src.models.quantization import ModelQuantizer

quantizer = ModelQuantizer()
quantized_model = quantizer.quantize_model(
    model_path="models/yolov8n_character.pt",
    quantization_type="dynamic",  # or "static"
    output_path="models/quantized/character_quantized.pt"
)

# Performance comparison / 性能比较
original_size = quantizer.get_model_size("models/yolov8n_character.pt")
quantized_size = quantizer.get_model_size("models/quantized/character_quantized.pt")
compression_ratio = original_size / quantized_size

print(f"Compression ratio: {compression_ratio:.2f}x")
print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

### Model Pruning / 模型剪枝

```python
# Structured pruning for faster inference / 结构化剪枝以加快推理
from src.models.pruning import ModelPruner

pruner = ModelPruner()
pruned_model = pruner.prune_model(
    model_path="models/yolov8n_character.pt",
    pruning_ratio=0.3,  # Remove 30% of parameters
    structured=True,
    output_path="models/pruned/character_pruned.pt"
)

# Fine-tune pruned model / 微调剪枝后的模型
detector = YOLOCharacterDetector()
detector.load_model("models/pruned/character_pruned.pt")
detector.train(
    dataset_yaml="data/processed/yolo_format/dataset.yaml",
    epochs=20,  # Fewer epochs for fine-tuning
    learning_rate=0.0001  # Lower learning rate
)
```

## Inference Optimization / 推理优化

### Batch Processing / 批处理

```python
# Optimize batch size for throughput / 为吞吐量优化批次大小
def find_optimal_batch_size(detector, test_images, max_batch_size=8):
    best_throughput = 0
    best_batch_size = 1
    
    for batch_size in range(1, max_batch_size + 1):
        try:
            # Test batch processing / 测试批处理
            batches = [test_images[i:i+batch_size] 
                      for i in range(0, len(test_images), batch_size)]
            
            start_time = time.time()
            for batch in batches:
                detector.predict_batch(batch)
            end_time = time.time()
            
            throughput = len(test_images) / (end_time - start_time)
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
    
    return best_batch_size, best_throughput
```

### Input Resolution Optimization / 输入分辨率优化

```python
# Test different input resolutions / 测试不同输入分辨率
resolutions = [(416, 416), (512, 512), (640, 640), (800, 800)]
performance_results = {}

for width, height in resolutions:
    # Update model configuration / 更新模型配置
    config = {
        'model': {
            'input_size': [width, height],
            'confidence_threshold': 0.5
        }
    }
    
    detector = YOLOCharacterDetector(config)
    detector.load_model("models/optimized/character_fp16.engine")
    
    # Benchmark performance / 基准测试性能
    test_image = cv2.resize(test_image, (width, height))
    
    start_time = time.time()
    for _ in range(100):
        detections = detector.predict(test_image)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    fps = 1.0 / avg_time
    
    performance_results[(width, height)] = {
        'fps': fps,
        'inference_time': avg_time,
        'accuracy': evaluate_accuracy(detections, ground_truth)
    }

# Find optimal resolution / 找到最优分辨率
optimal_resolution = max(performance_results.keys(), 
                        key=lambda x: performance_results[x]['fps'])
print(f"Optimal resolution: {optimal_resolution}")
```

### Pipeline Optimization / 流水线优化

```python
# Asynchronous processing pipeline / 异步处理流水线
import asyncio
import threading
from queue import Queue

class OptimizedDetectionPipeline:
    def __init__(self, model_path, max_queue_size=5):
        self.detector = YOLOCharacterDetector()
        self.detector.load_model(model_path)
        
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.running = False
    
    def start(self):
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()
    
    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1.0)
                detections = self.detector.predict(frame)
                self.output_queue.put((frame, detections))
            except:
                continue
    
    def add_frame(self, frame):
        if not self.input_queue.full():
            self.input_queue.put(frame)
    
    def get_result(self):
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None
```

## Memory Optimization / 内存优化

### Memory Profiling / 内存分析

```python
import psutil
import gc
from src.utils.performance import MemoryProfiler

# Profile memory usage / 分析内存使用
profiler = MemoryProfiler()

# Before model loading / 模型加载前
profiler.start_profiling("model_loading")
initial_memory = psutil.virtual_memory().used

# Load model / 加载模型
detector = YOLOCharacterDetector()
detector.load_model("models/character_model.pt")

# After model loading / 模型加载后
model_memory = psutil.virtual_memory().used
model_size = model_memory - initial_memory

print(f"Model memory usage: {model_size / 1024 / 1024:.1f} MB")

# Profile inference memory / 分析推理内存
profiler.start_profiling("inference")
for i in range(100):
    detections = detector.predict(test_image)
    if i % 10 == 0:
        gc.collect()  # Force garbage collection

profiler.stop_profiling("inference")
memory_stats = profiler.get_memory_stats()
```

### Memory Optimization Techniques / 内存优化技术

```python
# 1. Use memory-mapped files for large datasets / 对大数据集使用内存映射文件
import mmap

def load_dataset_mmap(dataset_path):
    with open(dataset_path, 'rb') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return mmapped_file

# 2. Implement image caching with size limits / 实现有大小限制的图像缓存
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_image_preprocessing(image_path):
    image = cv2.imread(image_path)
    return preprocess_image(image)

# 3. Use generators for batch processing / 使用生成器进行批处理
def batch_generator(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [cv2.imread(path) for path in batch_paths]
        yield batch_images
        # Images are automatically garbage collected after yield

# 4. Optimize OpenCV memory usage / 优化OpenCV内存使用
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Limit threads to reduce memory overhead
```

## Power Management / 功耗管理

### Dynamic Power Scaling / 动态功耗缩放

```python
from src.utils.jetson_utils import PowerManager

class AdaptivePowerManager:
    def __init__(self):
        self.power_manager = PowerManager()
        self.performance_history = []
        self.temperature_threshold = 75  # °C
    
    def adjust_power_mode(self, current_fps, target_fps, temperature):
        # Increase performance if FPS is below target / 如果FPS低于目标则提高性能
        if current_fps < target_fps * 0.9 and temperature < self.temperature_threshold:
            self.power_manager.set_max_performance()
        
        # Reduce power if temperature is high / 如果温度过高则降低功耗
        elif temperature > self.temperature_threshold:
            self.power_manager.set_power_mode(1)  # 5W mode
        
        # Optimize for efficiency if performance is adequate / 如果性能足够则优化效率
        elif current_fps > target_fps * 1.1:
            self.power_manager.set_balanced_mode()

# Usage in real-time detection / 在实时检测中使用
power_manager = AdaptivePowerManager()

def detection_callback(result):
    current_fps = result.fps
    temperature = jetson_monitor.get_temperature()
    
    power_manager.adjust_power_mode(
        current_fps=current_fps,
        target_fps=15.0,
        temperature=temperature
    )
```

### Battery Optimization / 电池优化

```python
# Battery-aware detection settings / 电池感知检测设置
class BatteryOptimizedDetector:
    def __init__(self, model_path):
        self.detector = YOLOCharacterDetector()
        self.battery_level = self.get_battery_level()
        
        # Adjust settings based on battery level / 根据电池电量调整设置
        if self.battery_level > 50:
            # High performance mode / 高性能模式
            self.detector.load_model(model_path)
            self.confidence_threshold = 0.5
            self.processing_interval = 1  # Process every frame
        elif self.battery_level > 20:
            # Balanced mode / 平衡模式
            self.detector.load_model("models/optimized/character_quantized.pt")
            self.confidence_threshold = 0.6
            self.processing_interval = 2  # Process every 2nd frame
        else:
            # Power saving mode / 省电模式
            self.detector.load_model("models/optimized/character_int8.engine")
            self.confidence_threshold = 0.7
            self.processing_interval = 5  # Process every 5th frame
    
    def get_battery_level(self):
        # Implementation depends on hardware / 实现取决于硬件
        try:
            with open('/sys/class/power_supply/BAT0/capacity', 'r') as f:
                return int(f.read().strip())
        except:
            return 100  # Assume AC power if battery info unavailable
```

## Benchmarking / 基准测试

### Performance Benchmarking / 性能基准测试

```python
from src.utils.performance import BenchmarkSuite

# Comprehensive benchmark / 综合基准测试
benchmark = BenchmarkSuite()

# Speed benchmark / 速度基准测试
speed_results = benchmark.benchmark_speed(
    model_path="models/character_model.pt",
    test_images=test_image_list,
    iterations=100
)

print(f"Average FPS: {speed_results['avg_fps']:.1f}")
print(f"Min FPS: {speed_results['min_fps']:.1f}")
print(f"Max FPS: {speed_results['max_fps']:.1f}")
print(f"Inference time: {speed_results['avg_inference_time']:.3f}s")

# Memory benchmark / 内存基准测试
memory_results = benchmark.benchmark_memory(
    model_path="models/character_model.pt",
    test_duration=300  # 5 minutes
)

print(f"Peak memory usage: {memory_results['peak_memory_mb']:.1f} MB")
print(f"Average memory usage: {memory_results['avg_memory_mb']:.1f} MB")

# Accuracy benchmark / 精度基准测试
accuracy_results = benchmark.benchmark_accuracy(
    model_path="models/character_model.pt",
    test_dataset="data/test",
    ground_truth="data/test/annotations.json"
)

print(f"Precision: {accuracy_results['precision']:.3f}")
print(f"Recall: {accuracy_results['recall']:.3f}")
print(f"F1 Score: {accuracy_results['f1_score']:.3f}")
```

### Comparative Analysis / 对比分析

```python
# Compare different optimization strategies / 比较不同优化策略
models_to_compare = [
    ("Original", "models/yolov8n_character.pt"),
    ("TensorRT FP16", "models/optimized/character_fp16.engine"),
    ("TensorRT INT8", "models/optimized/character_int8.engine"),
    ("Quantized", "models/quantized/character_quantized.pt"),
    ("Pruned", "models/pruned/character_pruned.pt")
]

comparison_results = {}

for name, model_path in models_to_compare:
    print(f"Benchmarking {name}...")
    
    # Load model / 加载模型
    detector = YOLOCharacterDetector()
    detector.load_model(model_path)
    
    # Run benchmark / 运行基准测试
    results = benchmark.run_full_benchmark(detector, test_images)
    comparison_results[name] = results

# Generate comparison report / 生成对比报告
benchmark.generate_comparison_report(comparison_results, "reports/optimization_comparison.html")
```
