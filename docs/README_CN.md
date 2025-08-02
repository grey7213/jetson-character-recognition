# Jetson Nano字符识别系统

**Language / 语言**: [English](../README.md) | [中文](README_CN.md)

---

## 项目概述

这是一个专为NVIDIA Jetson Nano硬件优化的计算机视觉系统，用于自主船舶导航中的字母数字字符检测和识别（0-9，A-Z）。该系统基于YOLOv8架构，提供实时字符检测能力。

## 主要特性

- **实时字符检测和识别** - 支持摄像头输入的实时处理
- **完整字符支持** - 识别数字（0-9）和字母（A-Z）共36个字符类别
- **Jetson Nano优化** - 针对硬件约束进行专门优化
- **多种部署选项** - 支持USB/CSI摄像头输入
- **高性能推理** - 在Jetson Nano上实现>15 FPS的检测速度
- **低功耗设计** - 功耗控制在10W以内

## 系统要求

### 硬件要求
- **主板**: NVIDIA Jetson Nano开发套件（推荐4GB版本）
- **存储**: 64GB Class 10或更高级别的MicroSD卡
- **摄像头**: 兼容Jetson Nano的USB网络摄像头或CSI摄像头模块
- **电源**: 5V 4A桶形插头电源适配器
- **散热**: 主动散热风扇（推荐用于连续运行）

### 软件要求
- **操作系统**: JetPack 4.6或更高版本
- **Python**: 3.6+
- **CUDA**: 10.2+
- **OpenCV**: 4.5+
- **PyTorch**: 1.10+

## 项目结构

```
jetson_character_recognition/
├── README.md                 # 项目文档
├── requirements.txt          # Python依赖
├── setup.py                 # 安装脚本
├── config/                  # 配置文件
│   ├── model_config.yaml   # 模型配置
│   └── camera_config.yaml  # 摄像头设置
├── src/                     # 源代码
│   ├── models/             # 模型定义
│   ├── data/               # 数据处理
│   ├── utils/              # 工具函数
│   └── inference/          # 推理流水线
├── models/                  # 训练好的模型文件
├── data/                    # 数据集和样本
├── scripts/                 # 训练和部署脚本
├── tests/                   # 单元测试
└── docs/                    # 文档
```

## 快速开始

### 安装

#### 自动安装（推荐）
```bash
# 克隆仓库
git clone <repository-url>
cd jetson_character_recognition

# 在Jetson Nano上安装
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh --with-tests

# 或手动安装
pip install -r requirements.txt
pip install -e .
```

#### 手动安装步骤

1. **系统更新**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev build-essential cmake git
```

2. **安装机器学习框架**
```bash
# 安装OpenCV
sudo apt install -y python3-opencv

# 安装PyTorch（Jetson专用版本）
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torchvision
```

3. **安装依赖**
```bash
pip3 install -r requirements.txt
```

4. **安装软件包**
```bash
pip3 install -e .
```

### 使用方法

#### 基本使用

```bash
# 运行交互式演示
python scripts/demo.py

# 训练模型
python scripts/train_model.py --dataset synthetic --epochs 50

# 运行实时检测
python scripts/run_detection.py models/character_detector.pt

# 测试系统
python scripts/test_system.py
```

#### Python API使用

```python
from src.models.yolo_character_detector import YOLOCharacterDetector
from src.inference.realtime_detector import RealtimeCharacterDetector

# 加载模型并运行检测
detector = YOLOCharacterDetector()
detector.load_model("models/character_detector.pt")

# 单图像检测
import cv2
image = cv2.imread("test_image.jpg")
detections = detector.predict(image)

# 实时检测
realtime_detector = RealtimeCharacterDetector("models/character_detector.pt")
realtime_detector.start_detection()
```

## 性能指标

### 目标性能
- **推理速度**: 在Jetson Nano上>10 FPS
- **检测精度**: 清晰字符>95%
- **内存使用**: <2GB RAM
- **功耗**: <10W

### 实际性能（Jetson Nano 4GB）
- **输入分辨率**: 640x640
- **模型**: YOLOv8n + TensorRT
- **FPS**: 10-15 FPS
- **推理时间**: 50-80ms
- **内存使用**: 1.5-2GB
- **功耗**: 8-12W

## 配置

### 模型配置

编辑 `config/model_config.yaml`:

```yaml
model:
  name: "yolov8n"  # 使用nano模型适配Jetson
  input_size: [640, 640]
  confidence_threshold: 0.5
  nms_threshold: 0.4

jetson:
  use_tensorrt: true
  fp16: true
  max_batch_size: 1
```

### 摄像头配置

编辑 `config/camera_config.yaml`:

```yaml
camera:
  type: 'usb'  # 或'csi'用于CSI摄像头
  device_id: 0

capture:
  width: 1280
  height: 720
  fps: 30
  buffer_size: 1  # 减少延迟
```

## 模型训练

### 使用合成数据集

```bash
python scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 8
```

### 使用自定义数据集

1. 准备YOLO格式的数据集
2. 更新数据集配置
3. 运行训练：

```bash
python scripts/train_model.py --dataset custom --epochs 100
```

### 训练自定义模型示例

```python
from src.data.dataset_manager import DatasetManager
from src.models.yolo_character_detector import YOLOCharacterDetector

# 准备数据集
dataset_manager = DatasetManager()
dataset_dir = dataset_manager.download_dataset("synthetic")
dataset_manager.prepare_yolo_dataset(dataset_dir, "yolo_dataset")

# 训练模型
detector = YOLOCharacterDetector()
detector.load_model(pretrained=True)
results = detector.train("yolo_dataset/dataset.yaml", epochs=100)
```

## 模型优化

### TensorRT优化

```python
from src.models.tensorrt_optimizer import TensorRTOptimizer

optimizer = TensorRTOptimizer()
optimized_model = optimizer.optimize_onnx_to_tensorrt(
    "model.onnx", 
    "model_tensorrt.engine"
)
```

### 性能调优

```bash
# 设置最大性能模式
sudo nvpmodel -m 0

# 最大化CPU时钟频率
sudo jetson_clocks

# 监控性能
jtop
```

## 运行检测

### 实时检测

```bash
python scripts/run_detection.py models/character_detector.pt
```

### 批处理

```python
from src.inference.batch_processor import BatchProcessor

processor = BatchProcessor("models/character_detector.pt")
results = processor.process_directory("input_images/")
```

## 数据集

### 合成数据集

系统包含一个强大的合成数据生成器，可以创建多样化的训练数据：

- **字符类别**: 36个（0-9数字 + A-Z字母）
- **字体变化**: 5种不同字体
- **增强技术**: 旋转、缩放、噪声、模糊
- **每个字符**: 100个变体

```bash
# 生成合成数据集
python data/tools/data_generator.py --output data/synthetic --count 100 --yolo --samples
```

### 真实世界数据

支持从实际摄像头捕获和真实场景收集的数据，包括：

- Jetson Nano摄像头捕获
- 公共字符识别数据集
- 自定义收集的图像

## 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 系统测试

```bash
# 运行系统诊断
python scripts/test_system.py

# 性能基准测试
python tests/performance/test_speed_benchmarks.py
```

## 部署

### 生产部署

创建systemd服务以实现自动启动：

```bash
sudo tee /etc/systemd/system/jetson-char-recognition.service << EOF
[Unit]
Description=Jetson Character Recognition Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/jetson_character_recognition
ExecStart=/usr/bin/python3 scripts/run_detection.py models/character_detector.pt
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl enable jetson-char-recognition.service
sudo systemctl start jetson-char-recognition.service
```

### 开机自启动

```bash
# 添加到crontab
crontab -e

# 添加行：
@reboot /home/jetson/jetson_character_recognition/scripts/run_detection.py /home/jetson/models/character_detector.pt
```

## 故障排除

### 常见问题

#### 低FPS
- **原因**: 处理能力不足或内存不够
- **解决方案**: 
  - 降低输入分辨率
  - 启用TensorRT优化
  - 使用更小的模型（YOLOv8n）
  - 增加交换文件大小

#### 高内存使用
- **原因**: 模型过大或批次大小过大
- **解决方案**:
  - 使用批次大小为1
  - 启用FP16精度
  - 减小模型大小
  - 添加交换文件

#### 摄像头问题
- **原因**: 摄像头未检测到或权限问题
- **解决方案**:
  - 检查摄像头连接
  - 验证摄像头权限: `sudo usermod -a -G video $USER`
  - 测试摄像头: `v4l2-ctl --list-devices`

### 调试命令

```bash
# 检查系统信息
python scripts/test_system.py

# 测试摄像头
python -c "from src.inference.camera_handler import CameraHandler; CameraHandler().test_camera()"

# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 监控资源
htop
nvidia-smi
jtop
```

## 文档

- [部署指南](docs/DEPLOYMENT_GUIDE_CN.md) - 完整部署说明
- [API文档](docs/API.md) - 代码API参考
- [性能调优](docs/PERFORMANCE.md) - 优化指南

## 示例应用

### 参考图片场景检测

系统特别适合处理以下场景：

1. **圆柱形表面字符** - 如参考图片中的数字"5"
2. **几何背景字符** - 如六边形标志中的字母"M"
3. **多色彩环境** - 不同颜色背景下的字符识别

### 实际应用案例

- 船舶导航标识识别
- 工业设备标签读取
- 自动化检测系统
- 机器人视觉导航

## 贡献

1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 添加测试
5. 提交拉取请求

## 支持

如有问题和疑问：

- 查看[故障排除指南](docs/DEPLOYMENT_GUIDE_CN.md#troubleshooting)
- 运行系统诊断: `python scripts/test_system.py`
- 在GitHub上创建issue

## 许可证

MIT许可证

## 更新日志

### v1.0.0
- 初始发布
- 支持36个字符类别检测
- Jetson Nano优化
- 实时检测功能
- 合成数据集生成器
- 完整的测试套件
