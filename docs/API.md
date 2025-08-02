# API Documentation / API文档

## Overview / 概述

This document provides comprehensive API documentation for the Jetson Character Recognition system. The API is designed for easy integration and provides both high-level and low-level interfaces for character detection and recognition.

本文档提供Jetson字符识别系统的综合API文档。API设计易于集成，为字符检测和识别提供高级和低级接口。

## Table of Contents / 目录

1. [Core Classes / 核心类](#core-classes)
2. [Model API / 模型API](#model-api)
3. [Data Processing API / 数据处理API](#data-processing-api)
4. [Inference API / 推理API](#inference-api)
5. [Utility API / 工具API](#utility-api)
6. [Configuration API / 配置API](#configuration-api)
7. [Examples / 示例](#examples)

## Core Classes / 核心类

### YOLOCharacterDetector

Main class for character detection using YOLOv8.

使用YOLOv8进行字符检测的主类。

```python
from src.models.yolo_character_detector import YOLOCharacterDetector

class YOLOCharacterDetector:
    def __init__(self, model_config: Dict[str, Any] = None)
    def load_model(self, model_path: str = None, pretrained: bool = False) -> None
    def predict(self, image: np.ndarray, confidence: float = None, nms_threshold: float = None) -> List[Dict[str, Any]]
    def predict_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]
    def train(self, dataset_yaml: str, epochs: int = 100, **kwargs) -> Any
    def export_model(self, output_path: str, format: str = "onnx") -> str
    def get_model_info(self) -> Dict[str, Any]
```

#### Methods / 方法

##### `__init__(model_config: Dict[str, Any] = None)`

Initialize the character detector.

初始化字符检测器。

**Parameters / 参数:**
- `model_config` (Dict, optional): Model configuration dictionary / 模型配置字典

**Example / 示例:**
```python
detector = YOLOCharacterDetector()
# or with custom config / 或使用自定义配置
custom_config = {'model': {'confidence_threshold': 0.7}}
detector = YOLOCharacterDetector(custom_config)
```

##### `load_model(model_path: str = None, pretrained: bool = False) -> None`

Load a trained model for inference.

加载训练好的模型进行推理。

**Parameters / 参数:**
- `model_path` (str, optional): Path to model file / 模型文件路径
- `pretrained` (bool): Use pretrained YOLOv8 model / 使用预训练YOLOv8模型

**Raises / 异常:**
- `FileNotFoundError`: If model file doesn't exist / 如果模型文件不存在
- `ValueError`: If model format is invalid / 如果模型格式无效

**Example / 示例:**
```python
# Load pretrained model / 加载预训练模型
detector.load_model(pretrained=True)

# Load custom model / 加载自定义模型
detector.load_model("models/my_character_model.pt")
```

##### `predict(image: np.ndarray, confidence: float = None, nms_threshold: float = None) -> List[Dict[str, Any]]`

Detect characters in a single image.

在单张图像中检测字符。

**Parameters / 参数:**
- `image` (np.ndarray): Input image (BGR format) / 输入图像（BGR格式）
- `confidence` (float, optional): Confidence threshold / 置信度阈值
- `nms_threshold` (float, optional): NMS threshold / NMS阈值

**Returns / 返回:**
- `List[Dict]`: List of detected characters / 检测到的字符列表

**Detection Result Format / 检测结果格式:**
```python
{
    'class_id': int,        # Class ID (0-35) / 类别ID
    'class_name': str,      # Character ('0'-'9', 'A'-'Z') / 字符
    'confidence': float,    # Detection confidence / 检测置信度
    'bbox': [x1, y1, x2, y2],  # Bounding box / 边界框
    'center': [cx, cy],     # Center point / 中心点
    'area': float          # Bounding box area / 边界框面积
}
```

**Example / 示例:**
```python
import cv2
image = cv2.imread("test_image.jpg")
detections = detector.predict(image, confidence=0.6)

for detection in detections:
    print(f"Detected: {detection['class_name']} (confidence: {detection['confidence']:.2f})")
```

### RealtimeCharacterDetector

Real-time character detection pipeline.

实时字符检测流水线。

```python
from src.inference.realtime_detector import RealtimeCharacterDetector

class RealtimeCharacterDetector:
    def __init__(self, model_path: str = None, camera_config: Dict = None, model_config: Dict = None)
    def start_detection(self, callback: Callable = None) -> None
    def stop_detection(self) -> None
    def detect_single_frame(self, frame: np.ndarray) -> DetectionResult
    def get_performance_stats(self) -> Dict[str, float]
```

#### Methods / 方法

##### `start_detection(callback: Callable = None) -> None`

Start real-time character detection.

开始实时字符检测。

**Parameters / 参数:**
- `callback` (Callable, optional): Callback function for detection results / 检测结果回调函数

**Callback Signature / 回调函数签名:**
```python
def detection_callback(result: DetectionResult) -> None:
    # Process detection result / 处理检测结果
    pass
```

**Example / 示例:**
```python
def on_detection(result):
    print(f"Detected {len(result.detections)} characters")
    print(f"FPS: {result.fps:.1f}")

detector = RealtimeCharacterDetector("models/character_model.pt")
detector.start_detection(callback=on_detection)
```

### DatasetManager

Dataset management and processing.

数据集管理和处理。

```python
from src.data.dataset_manager import DatasetManager

class DatasetManager:
    def __init__(self, data_dir: str = "data")
    def download_dataset(self, dataset_name: str) -> Path
    def prepare_yolo_dataset(self, dataset_dir: Path, output_dir: Path) -> None
    def get_dataset_info(self, dataset_name: str) -> Dict
    def list_available_datasets(self) -> List[str]
```

## Model API / 模型API

### Model Loading / 模型加载

```python
# Load different model types / 加载不同类型的模型
detector = YOLOCharacterDetector()

# Pretrained model / 预训练模型
detector.load_model(pretrained=True)

# Custom trained model / 自定义训练模型
detector.load_model("models/custom_model.pt")

# Model with specific configuration / 特定配置的模型
config = {
    'model': {
        'confidence_threshold': 0.7,
        'nms_threshold': 0.3
    }
}
detector = YOLOCharacterDetector(config)
```

### Model Training / 模型训练

```python
# Prepare dataset / 准备数据集
dataset_manager = DatasetManager()
dataset_dir = dataset_manager.download_dataset("synthetic")
dataset_manager.prepare_yolo_dataset(dataset_dir, "yolo_dataset")

# Train model / 训练模型
detector = YOLOCharacterDetector()
detector.load_model(pretrained=True)
results = detector.train(
    dataset_yaml="yolo_dataset/dataset.yaml",
    epochs=100,
    batch_size=16,
    learning_rate=0.001
)
```

### Model Export / 模型导出

```python
# Export to different formats / 导出为不同格式
detector.load_model("models/trained_model.pt")

# ONNX format / ONNX格式
onnx_path = detector.export_model("models/model.onnx", format="onnx")

# TensorRT format (Jetson) / TensorRT格式（Jetson）
trt_path = detector.export_model("models/model.engine", format="tensorrt")
```

## Data Processing API / 数据处理API

### Dataset Generation / 数据集生成

```python
from data.tools.data_generator import SyntheticDataGenerator

# Generate synthetic dataset / 生成合成数据集
generator = SyntheticDataGenerator("data/synthetic")
stats = generator.generate_dataset(count_per_character=100)

# Generate YOLO annotations / 生成YOLO标注
dataset_yaml = generator.generate_yolo_annotations()
```

### Data Augmentation / 数据增强

```python
from src.data.augmentation import CharacterAugmentation

augmenter = CharacterAugmentation()
augmented_data = augmenter.augment_dataset(
    input_dir="data/raw",
    output_dir="data/augmented",
    augmentation_factor=3
)
```

## Inference API / 推理API

### Single Image Detection / 单图像检测

```python
import cv2

# Load image / 加载图像
image = cv2.imread("test_image.jpg")

# Detect characters / 检测字符
detector = YOLOCharacterDetector()
detector.load_model("models/character_model.pt")
detections = detector.predict(image)

# Process results / 处理结果
for detection in detections:
    x1, y1, x2, y2 = detection['bbox']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, detection['class_name'], (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
```

### Batch Processing / 批处理

```python
# Process multiple images / 处理多张图像
images = [cv2.imread(f"image_{i}.jpg") for i in range(10)]
batch_results = detector.predict_batch(images)

for i, detections in enumerate(batch_results):
    print(f"Image {i}: {len(detections)} characters detected")
```

### Real-time Processing / 实时处理

```python
from src.inference.realtime_detector import RealtimeCharacterDetector

# Setup real-time detector / 设置实时检测器
realtime_detector = RealtimeCharacterDetector(
    model_path="models/character_model.pt"
)

# Define callback / 定义回调函数
def process_detections(result):
    print(f"Frame: {len(result.detections)} characters, FPS: {result.fps:.1f}")
    
    # Save detection results / 保存检测结果
    for detection in result.detections:
        print(f"  {detection['class_name']}: {detection['confidence']:.2f}")

# Start detection / 开始检测
realtime_detector.start_detection(callback=process_detections)

# Stop after some time / 一段时间后停止
import time
time.sleep(30)
realtime_detector.stop_detection()
```

## Utility API / 工具API

### Performance Monitoring / 性能监控

```python
from src.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor()

# Time operations / 计时操作
monitor.start_timer('inference')
detections = detector.predict(image)
inference_time = monitor.stop_timer('inference')

# Record metrics / 记录指标
from src.utils.performance import PerformanceMetrics
metrics = PerformanceMetrics(
    inference_time=inference_time,
    fps=1.0/inference_time,
    memory_usage=monitor.get_memory_usage()
)
monitor.record_metrics(metrics)

# Get statistics / 获取统计信息
stats = monitor.get_performance_summary()
```

### Jetson Utilities / Jetson工具

```python
from src.utils.jetson_utils import jetson_monitor, jetson_optimizer

# Check if running on Jetson / 检查是否在Jetson上运行
if jetson_monitor.is_jetson:
    # Get system info / 获取系统信息
    system_info = jetson_monitor.get_system_info()
    print(f"Temperature: {system_info['temperature']}°C")
    
    # Optimize performance / 优化性能
    jetson_optimizer.setup_environment()
    jetson_optimizer.optimize_opencv()
```

## Configuration API / 配置API

### Loading Configuration / 加载配置

```python
from src.utils.config_loader import config_loader

# Load model configuration / 加载模型配置
model_config = config_loader.get_model_config()

# Load camera configuration / 加载摄像头配置
camera_config = config_loader.get_camera_config()

# Update configuration / 更新配置
model_config['model']['confidence_threshold'] = 0.7
config_loader.update_model_config(model_config)
```

### Custom Configuration / 自定义配置

```python
# Create custom configuration / 创建自定义配置
custom_config = {
    'model': {
        'name': 'yolov8s',
        'confidence_threshold': 0.6,
        'nms_threshold': 0.4
    },
    'jetson': {
        'use_tensorrt': True,
        'fp16': True
    }
}

# Use with detector / 与检测器一起使用
detector = YOLOCharacterDetector(custom_config)
```

## Error Handling / 错误处理

### Common Exceptions / 常见异常

```python
try:
    detector = YOLOCharacterDetector()
    detector.load_model("nonexistent_model.pt")
except FileNotFoundError:
    print("Model file not found")
except ValueError as e:
    print(f"Invalid model format: {e}")

try:
    detections = detector.predict(image)
except ValueError as e:
    print(f"Model not loaded: {e}")
except Exception as e:
    print(f"Prediction failed: {e}")
```

## Examples / 示例

### Complete Detection Pipeline / 完整检测流水线

```python
import cv2
from src.models.yolo_character_detector import YOLOCharacterDetector
from src.utils.performance import PerformanceMonitor

# Initialize components / 初始化组件
detector = YOLOCharacterDetector()
monitor = PerformanceMonitor()

# Load model / 加载模型
detector.load_model("models/character_model.pt")

# Load and process image / 加载和处理图像
image = cv2.imread("test_image.jpg")

# Detect characters / 检测字符
monitor.start_timer('detection')
detections = detector.predict(image, confidence=0.6)
detection_time = monitor.stop_timer('detection')

# Display results / 显示结果
print(f"Detected {len(detections)} characters in {detection_time:.3f}s")

for detection in detections:
    char = detection['class_name']
    conf = detection['confidence']
    bbox = detection['bbox']
    print(f"Character: {char}, Confidence: {conf:.2f}, BBox: {bbox}")
    
    # Draw bounding box / 绘制边界框
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{char} {conf:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save result / 保存结果
cv2.imwrite("detection_result.jpg", image)
```

### Training Custom Model / 训练自定义模型

```python
from src.data.dataset_manager import DatasetManager
from src.models.yolo_character_detector import YOLOCharacterDetector

# Prepare dataset / 准备数据集
dataset_manager = DatasetManager()
synthetic_dir = dataset_manager.download_dataset("synthetic")
dataset_manager.prepare_yolo_dataset(synthetic_dir, "training_data")

# Initialize and train model / 初始化和训练模型
detector = YOLOCharacterDetector()
detector.load_model(pretrained=True)

# Train with custom parameters / 使用自定义参数训练
training_results = detector.train(
    dataset_yaml="training_data/dataset.yaml",
    epochs=100,
    batch_size=16,
    learning_rate=0.001,
    patience=10
)

# Save trained model / 保存训练好的模型
detector.model.save("models/custom_character_model.pt")
print(f"Training completed. Final accuracy: {training_results.results_dict['fitness']:.3f}")
```

### Real-time Detection with Camera / 摄像头实时检测

```python
from src.inference.realtime_detector import RealtimeCharacterDetector
from src.inference.camera_handler import CameraHandler
import cv2

# Initialize camera and detector / 初始化摄像头和检测器
camera = CameraHandler()
detector = RealtimeCharacterDetector("models/character_model.pt")

# Setup detection callback / 设置检测回调
def on_detection_result(result):
    frame = result.frame
    detections = result.detections

    # Draw detections on frame / 在帧上绘制检测结果
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, detection['class_name'], (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame / 显示帧
    cv2.imshow('Character Detection', frame)

    # Print statistics / 打印统计信息
    print(f"FPS: {result.fps:.1f}, Characters: {len(detections)}")

# Start real-time detection / 开始实时检测
try:
    detector.start_detection(callback=on_detection_result)

    # Wait for user to press 'q' to quit / 等待用户按'q'退出
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    detector.stop_detection()
    cv2.destroyAllWindows()
```

## API Reference Summary / API参考摘要

### Core Classes / 核心类
- `YOLOCharacterDetector`: Main detection class / 主检测类
- `RealtimeCharacterDetector`: Real-time detection pipeline / 实时检测流水线
- `DatasetManager`: Dataset management / 数据集管理
- `CameraHandler`: Camera interface / 摄像头接口

### Key Methods / 关键方法
- `load_model()`: Load trained model / 加载训练模型
- `predict()`: Single image detection / 单图像检测
- `predict_batch()`: Batch processing / 批处理
- `train()`: Model training / 模型训练
- `export_model()`: Model export / 模型导出

### Configuration / 配置
- Model configuration via YAML files / 通过YAML文件进行模型配置
- Runtime parameter adjustment / 运行时参数调整
- Jetson-specific optimizations / Jetson特定优化

### Performance Monitoring / 性能监控
- Real-time FPS tracking / 实时FPS跟踪
- Memory usage monitoring / 内存使用监控
- Inference time measurement / 推理时间测量
