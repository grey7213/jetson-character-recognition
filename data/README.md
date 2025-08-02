# Data Directory / 数据目录

## Overview / 概述

This directory contains datasets, sample images, and data processing utilities for the Jetson Character Recognition system.

本目录包含Jetson字符识别系统的数据集、样本图像和数据处理工具。

## Directory Structure / 目录结构

```
data/
├── README.md                    # This file / 本文件
├── raw/                         # Raw, unprocessed data / 原始未处理数据
│   ├── real_world/             # Real-world character images / 真实世界字符图像
│   │   ├── digits/             # Digit images (0-9) / 数字图像
│   │   ├── letters/            # Letter images (A-Z) / 字母图像
│   │   └── mixed/              # Mixed character scenes / 混合字符场景
│   ├── synthetic/              # Synthetic generated data / 合成生成数据
│   │   ├── fonts/              # Different font variations / 不同字体变化
│   │   ├── backgrounds/        # Background variations / 背景变化
│   │   └── augmented/          # Augmented samples / 增强样本
│   └── reference/              # Reference images for testing / 测试参考图像
│       ├── jetson_samples/     # Jetson Nano test samples / Jetson Nano测试样本
│       └── benchmark/          # Benchmark test images / 基准测试图像
├── processed/                   # Processed and annotated data / 处理和标注数据
│   ├── yolo_format/            # YOLO format annotations / YOLO格式标注
│   │   ├── images/             # Training images / 训练图像
│   │   ├── labels/             # YOLO label files / YOLO标签文件
│   │   ├── train.txt           # Training set list / 训练集列表
│   │   ├── val.txt             # Validation set list / 验证集列表
│   │   └── dataset.yaml        # Dataset configuration / 数据集配置
│   ├── coco_format/            # COCO format annotations / COCO格式标注
│   └── custom_format/          # Custom annotation format / 自定义标注格式
├── samples/                     # Sample images for testing / 测试样本图像
│   ├── single_characters/      # Individual character samples / 单个字符样本
│   ├── multi_characters/       # Multiple character scenes / 多字符场景
│   ├── challenging/            # Challenging test cases / 挑战性测试用例
│   └── reference_scenes/       # Reference scene images / 参考场景图像
├── statistics/                  # Dataset statistics and analysis / 数据集统计和分析
│   ├── class_distribution.json # Class distribution data / 类别分布数据
│   ├── image_statistics.json   # Image statistics / 图像统计
│   └── annotation_analysis.json # Annotation analysis / 标注分析
└── tools/                       # Data processing tools / 数据处理工具
    ├── data_generator.py       # Synthetic data generator / 合成数据生成器
    ├── annotation_converter.py # Annotation format converter / 标注格式转换器
    ├── data_augmentation.py    # Data augmentation tools / 数据增强工具
    └── dataset_analyzer.py     # Dataset analysis tools / 数据集分析工具
```

## Dataset Types / 数据集类型

### 1. Synthetic Dataset / 合成数据集

Generated programmatically with various fonts, sizes, and backgrounds.

通过程序生成，包含各种字体、尺寸和背景。

**Features / 特性:**
- 36 character classes (0-9, A-Z) / 36个字符类别
- 5 different fonts / 5种不同字体
- Multiple sizes and rotations / 多种尺寸和旋转
- Various backgrounds and noise / 各种背景和噪声
- 100 variations per character / 每个字符100个变体

**Generation / 生成:**
```bash
python data/tools/data_generator.py --output data/raw/synthetic --count 3600
```

### 2. Real-world Dataset / 真实世界数据集

Collected from actual camera captures and real-world scenarios.

从实际摄像头捕获和真实世界场景收集。

**Sources / 来源:**
- Jetson Nano camera captures / Jetson Nano摄像头捕获
- Public character recognition datasets / 公共字符识别数据集
- Custom collected images / 自定义收集图像

### 3. Reference Dataset / 参考数据集

Special test cases including the reference images mentioned in requirements.

特殊测试用例，包括需求中提到的参考图像。

## Sample Images / 样本图像

### Single Character Samples / 单字符样本

Located in `samples/single_characters/`:

位于 `samples/single_characters/`：

- `digit_0.jpg` to `digit_9.jpg` - Individual digit samples / 单个数字样本
- `letter_A.jpg` to `letter_Z.jpg` - Individual letter samples / 单个字母样本

### Multi-character Scenes / 多字符场景

Located in `samples/multi_characters/`:

位于 `samples/multi_characters/`：

- `license_plate.jpg` - License plate example / 车牌示例
- `signage.jpg` - Signage with multiple characters / 多字符标牌
- `display_panel.jpg` - Digital display panel / 数字显示面板

### Challenging Cases / 挑战性案例

Located in `samples/challenging/`:

位于 `samples/challenging/`：

- `low_light.jpg` - Low lighting conditions / 低光照条件
- `blurred.jpg` - Motion blur / 运动模糊
- `rotated.jpg` - Rotated characters / 旋转字符
- `occluded.jpg` - Partially occluded characters / 部分遮挡字符

### Reference Scenes / 参考场景

Located in `samples/reference_scenes/`:

位于 `samples/reference_scenes/`：

- `cylindrical_surface.jpg` - Characters on cylindrical surfaces / 圆柱表面字符
- `geometric_background.jpg` - Characters with geometric backgrounds / 几何背景字符
- `jetson_test_case.jpg` - Jetson Nano specific test case / Jetson Nano特定测试用例

## Data Processing / 数据处理

### Annotation Format / 标注格式

**YOLO Format / YOLO格式:**
```
class_id center_x center_y width height
```

Example / 示例:
```
0 0.5 0.5 0.1 0.15  # Digit '0' at center
10 0.3 0.4 0.08 0.12  # Letter 'A' 
```

**Class Mapping / 类别映射:**
- 0-9: Digits '0'-'9' / 数字'0'-'9'
- 10-35: Letters 'A'-'Z' / 字母'A'-'Z'

### Data Augmentation / 数据增强

Applied augmentations / 应用的增强:

- **Rotation**: ±15 degrees / 旋转：±15度
- **Scale**: 0.8-1.2x / 缩放：0.8-1.2倍
- **Brightness**: ±20% / 亮度：±20%
- **Contrast**: ±20% / 对比度：±20%
- **Noise**: Gaussian noise / 高斯噪声
- **Blur**: Motion and Gaussian blur / 运动和高斯模糊

### Dataset Statistics / 数据集统计

**Synthetic Dataset / 合成数据集:**
- Total images: 3,600 / 总图像数：3,600
- Images per class: 100 / 每类图像数：100
- Image size: 64x64 pixels / 图像尺寸：64x64像素
- Format: PNG / 格式：PNG

**Real-world Dataset / 真实世界数据集:**
- Total images: 1,200 / 总图像数：1,200
- Annotated characters: 2,800 / 标注字符数：2,800
- Average characters per image: 2.3 / 每图像平均字符数：2.3
- Image sizes: Variable (640x480 to 1920x1080) / 图像尺寸：可变

## Usage Examples / 使用示例

### Loading Dataset / 加载数据集

```python
from src.data.dataset_manager import DatasetManager

# Initialize dataset manager / 初始化数据集管理器
dataset_manager = DatasetManager("data")

# Load synthetic dataset / 加载合成数据集
synthetic_data = dataset_manager.download_dataset("synthetic")

# Prepare YOLO format dataset / 准备YOLO格式数据集
dataset_manager.prepare_yolo_dataset(
    synthetic_data, 
    "data/processed/yolo_format"
)
```

### Data Analysis / 数据分析

```python
from data.tools.dataset_analyzer import DatasetAnalyzer

# Analyze dataset / 分析数据集
analyzer = DatasetAnalyzer("data/processed/yolo_format")
stats = analyzer.get_statistics()

print(f"Total images: {stats['total_images']}")
print(f"Class distribution: {stats['class_distribution']}")
```

### Data Augmentation / 数据增强

```python
from src.data.augmentation import CharacterAugmentation

# Apply augmentation / 应用数据增强
augmenter = CharacterAugmentation()
augmented_data = augmenter.augment_dataset(
    "data/raw/real_world",
    "data/processed/augmented",
    augmentation_factor=3
)
```

## Data Collection Guidelines / 数据收集指南

### Image Quality Requirements / 图像质量要求

- **Resolution**: Minimum 640x480 pixels / 分辨率：最小640x480像素
- **Format**: JPEG or PNG / 格式：JPEG或PNG
- **Lighting**: Good contrast, avoid extreme shadows / 光照：良好对比度，避免极端阴影
- **Focus**: Sharp, minimal motion blur / 焦点：清晰，最小运动模糊

### Character Requirements / 字符要求

- **Size**: Characters should be at least 32x32 pixels / 尺寸：字符至少32x32像素
- **Visibility**: Clear, unoccluded characters / 可见性：清晰、无遮挡字符
- **Variety**: Different fonts, sizes, and orientations / 多样性：不同字体、尺寸和方向

### Annotation Guidelines / 标注指南

- **Bounding boxes**: Tight fit around characters / 边界框：紧贴字符
- **Class labels**: Use correct class IDs (0-35) / 类别标签：使用正确类别ID
- **Quality check**: Verify all annotations / 质量检查：验证所有标注

## Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **Missing data files / 缺少数据文件**
   - Run data generation script / 运行数据生成脚本
   - Check file permissions / 检查文件权限

2. **Annotation format errors / 标注格式错误**
   - Use annotation converter / 使用标注转换器
   - Validate annotation files / 验证标注文件

3. **Dataset loading fails / 数据集加载失败**
   - Check dataset.yaml configuration / 检查dataset.yaml配置
   - Verify file paths / 验证文件路径

### Data Validation / 数据验证

```bash
# Validate dataset / 验证数据集
python data/tools/dataset_analyzer.py --validate data/processed/yolo_format

# Check annotation consistency / 检查标注一致性
python data/tools/annotation_converter.py --check data/processed/yolo_format
```
