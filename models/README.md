# Models Directory / 模型目录

## Overview / 概述

This directory contains trained models and model-related files for the Jetson Character Recognition system.

本目录包含Jetson字符识别系统的训练模型和相关文件。

## Directory Structure / 目录结构

```
models/
├── README.md                    # This file / 本文件
├── pretrained/                  # Pre-trained models / 预训练模型
│   ├── yolov8n_character.pt    # YOLOv8 nano character model / YOLOv8 nano字符模型
│   ├── yolov8s_character.pt    # YOLOv8 small character model / YOLOv8 small字符模型
│   └── model_info.yaml         # Model metadata / 模型元数据
├── custom/                      # Custom trained models / 自定义训练模型
│   ├── synthetic_trained.pt    # Model trained on synthetic data / 合成数据训练模型
│   ├── real_world_trained.pt   # Model trained on real data / 真实数据训练模型
│   └── training_logs/          # Training logs and metrics / 训练日志和指标
├── optimized/                   # Optimized models for deployment / 部署优化模型
│   ├── tensorrt/               # TensorRT optimized models / TensorRT优化模型
│   │   ├── character_fp16.engine
│   │   └── character_int8.engine
│   ├── onnx/                   # ONNX format models / ONNX格式模型
│   │   └── character_model.onnx
│   └── quantized/              # Quantized models / 量化模型
└── benchmarks/                  # Model performance benchmarks / 模型性能基准
    ├── accuracy_results.json   # Accuracy test results / 精度测试结果
    ├── speed_benchmarks.json   # Speed test results / 速度测试结果
    └── memory_usage.json       # Memory usage statistics / 内存使用统计
```

## Model Types / 模型类型

### 1. Pre-trained Models / 预训练模型

- **yolov8n_character.pt**: YOLOv8 Nano model optimized for Jetson Nano
  - Input size: 640x640
  - Classes: 36 (0-9, A-Z)
  - Model size: ~6MB
  - Inference speed: ~15 FPS on Jetson Nano

- **yolov8s_character.pt**: YOLOv8 Small model for higher accuracy
  - Input size: 640x640
  - Classes: 36 (0-9, A-Z)
  - Model size: ~22MB
  - Inference speed: ~8 FPS on Jetson Nano

### 2. Custom Models / 自定义模型

- **synthetic_trained.pt**: Trained on synthetic character dataset
- **real_world_trained.pt**: Fine-tuned on real-world images

### 3. Optimized Models / 优化模型

- **TensorRT**: GPU-optimized models for maximum speed
- **ONNX**: Cross-platform compatible models
- **Quantized**: Reduced precision models for memory efficiency

## Usage / 使用方法

### Loading a Model / 加载模型

```python
from src.models.yolo_character_detector import YOLOCharacterDetector

# Load pre-trained model / 加载预训练模型
detector = YOLOCharacterDetector()
detector.load_model("models/pretrained/yolov8n_character.pt")

# Load custom model / 加载自定义模型
detector.load_model("models/custom/synthetic_trained.pt")
```

### Model Information / 模型信息

```python
# Get model information / 获取模型信息
model_info = detector.get_model_info()
print(f"Model classes: {model_info['num_classes']}")
print(f"Model size: {model_info['model_size_mb']} MB")
```

## Model Training / 模型训练

To train a new model, use the training script:

要训练新模型，请使用训练脚本：

```bash
# Train on synthetic dataset / 在合成数据集上训练
python scripts/train_model.py --dataset synthetic --epochs 100 --output models/custom/my_model.pt

# Train on custom dataset / 在自定义数据集上训练
python scripts/train_model.py --dataset custom --data-path data/custom --epochs 100
```

## Model Optimization / 模型优化

### TensorRT Optimization / TensorRT优化

```python
from src.models.tensorrt_optimizer import TensorRTOptimizer

optimizer = TensorRTOptimizer()
optimizer.optimize_model(
    input_model="models/pretrained/yolov8n_character.pt",
    output_path="models/optimized/tensorrt/character_fp16.engine",
    precision="fp16"
)
```

### ONNX Export / ONNX导出

```python
detector = YOLOCharacterDetector()
detector.load_model("models/pretrained/yolov8n_character.pt")
detector.export_model("models/optimized/onnx/character_model.onnx", format="onnx")
```

## Performance Benchmarks / 性能基准

| Model | Size | FPS (Jetson Nano) | Accuracy | Memory Usage |
|-------|------|-------------------|----------|--------------|
| YOLOv8n | 6MB | 15 FPS | 94.2% | 1.2GB |
| YOLOv8s | 22MB | 8 FPS | 96.8% | 1.8GB |
| TensorRT FP16 | 3MB | 20 FPS | 94.0% | 0.8GB |

## Model Validation / 模型验证

Before deploying a model, validate its performance:

在部署模型之前，验证其性能：

```bash
# Run model validation / 运行模型验证
python scripts/validate_model.py --model models/pretrained/yolov8n_character.pt --test-data data/test

# Benchmark model performance / 基准测试模型性能
python scripts/benchmark_model.py --model models/pretrained/yolov8n_character.pt
```

## Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **Model loading fails / 模型加载失败**
   - Check file path and permissions / 检查文件路径和权限
   - Verify model format compatibility / 验证模型格式兼容性

2. **Low inference speed / 推理速度慢**
   - Use TensorRT optimization / 使用TensorRT优化
   - Reduce input resolution / 降低输入分辨率
   - Use smaller model variant / 使用更小的模型变体

3. **High memory usage / 内存使用过高**
   - Use quantized models / 使用量化模型
   - Enable FP16 precision / 启用FP16精度
   - Reduce batch size / 减少批次大小

## Model Updates / 模型更新

To update models, follow these steps:

要更新模型，请按照以下步骤：

1. Backup current models / 备份当前模型
2. Download/train new models / 下载/训练新模型
3. Validate new models / 验证新模型
4. Update model paths in configuration / 更新配置中的模型路径
5. Test system functionality / 测试系统功能

## Support / 支持

For model-related issues:

模型相关问题：

- Check model validation results / 检查模型验证结果
- Review training logs / 查看训练日志
- Run benchmark tests / 运行基准测试
- Contact development team / 联系开发团队
