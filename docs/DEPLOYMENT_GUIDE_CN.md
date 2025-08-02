# 部署指南

**Language / 语言**: [English](DEPLOYMENT_GUIDE.md) | [中文](DEPLOYMENT_GUIDE_CN.md)

---

## 概述

本指南涵盖在NVIDIA Jetson Nano硬件上部署Jetson字符识别系统用于生产环境的完整流程。

## 前提条件

### 硬件要求

- **NVIDIA Jetson Nano开发套件**（推荐4GB版本）
- **MicroSD卡**: 64GB Class 10或更高级别
- **摄像头**: 兼容Jetson Nano的USB网络摄像头或CSI摄像头模块
- **电源**: 5V 4A桶形插头电源适配器
- **散热**: 主动散热风扇（推荐用于连续运行）

### 软件要求

- **JetPack 4.6**或更高版本
- **Python 3.6+**
- **CUDA 10.2+**
- **OpenCV 4.5+**
- **PyTorch 1.10+**

## 安装

### 快速安装

```bash
# 克隆仓库
git clone <repository-url>
cd jetson_character_recognition

# 运行安装脚本
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh --with-tests
```

### 手动安装

#### 1. 系统设置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
sudo apt install -y python3-pip python3-dev build-essential cmake git
```

#### 2. 安装机器学习框架

```bash
# 安装OpenCV
sudo apt install -y python3-opencv

# 安装PyTorch（Jetson专用版本）
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torchvision
```

#### 3. 安装依赖

```bash
pip3 install -r requirements.txt
```

#### 4. 安装软件包

```bash
pip3 install -e .
```

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
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 8
```

### 使用自定义数据集

1. 准备YOLO格式的数据集
2. 更新数据集配置
3. 运行训练：

```bash
python3 scripts/train_model.py --dataset custom --epochs 100
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
python3 scripts/run_detection.py models/character_detector.pt
```

### 批处理

```python
from src.inference.batch_processor import BatchProcessor

processor = BatchProcessor("models/character_detector.pt")
results = processor.process_directory("input_images/")
```

## 性能优化

### 内存优化

- 使用批次大小为1进行实时推理
- 启用FP16精度
- 如需要可降低输入分辨率
- 设置交换文件以获得额外内存

### 速度优化

- 使用TensorRT优化
- 启用GPU加速
- 优化OpenCV设置
- 为摄像头输入使用线程

### 功耗管理

```bash
# 检查功耗模式
sudo nvpmodel -q

# 设置功耗模式（0 = 最大性能，1 = 5W模式）
sudo nvpmodel -m 0
```

## 监控和调试

### 系统监控

```bash
# 安装jetson-stats
sudo pip3 install jetson-stats

# 监控系统
jtop
```

### 性能监控

```python
from src.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor()
# 在推理期间监控
summary = monitor.get_performance_summary()
```

### 日志记录

```python
from src.utils.logger import setup_logger

logger = setup_logger("deployment", level="INFO", log_file="deployment.log")
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

#### CUDA内存不足

- **原因**: GPU内存不足
- **解决方案**:
  - 减少批次大小
  - 使用更小的输入分辨率
  - 启用内存优化
  - 重启系统以清理内存

### 调试命令

```bash
# 检查系统信息
python3 scripts/test_system.py

# 测试摄像头
python3 -c "from src.inference.camera_handler import CameraHandler; CameraHandler().test_camera()"

# 检查CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 监控资源
htop
nvidia-smi
```

## 生产部署

### 服务设置

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

### 远程监控

设置SSH访问以进行远程监控：

```bash
# 启用SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# 远程监控
ssh jetson@<jetson-ip>
jtop
```

## 安全考虑

### 网络安全

- 更改默认密码
- 使用SSH密钥而不是密码
- 如需要配置防火墙
- 禁用不必要的服务

### 文件权限

```bash
# 设置适当的权限
chmod 755 scripts/*.py
chmod 644 config/*.yaml
chmod 600 models/*.pt
```

## 备份和恢复

### 配置备份

```bash
# 备份配置
tar -czf config_backup.tar.gz config/

# 备份模型
tar -czf models_backup.tar.gz models/
```

### 系统镜像备份

```bash
# 创建SD卡镜像备份
sudo dd if=/dev/mmcblk0 of=jetson_backup.img bs=4M status=progress
```

## 性能基准

### 预期性能（Jetson Nano 4GB）

- **输入分辨率**: 640x640
- **模型**: YOLOv8n + TensorRT
- **FPS**: 10-15 FPS
- **推理时间**: 50-80ms
- **内存使用**: 1.5-2GB
- **功耗**: 8-12W

### 优化结果

| 配置 | FPS | 内存 | 功耗 |
|------|-----|------|------|
| 默认 | 5-8 | 2.5GB | 15W |
| TensorRT + FP16 | 10-15 | 1.8GB | 12W |
| 优化版 | 12-18 | 1.5GB | 10W |

## 支持和维护

### 定期维护

- 监控系统温度
- 检查磁盘空间
- 定期更新依赖
- 备份模型和配置

### 获取帮助

- 检查日志: `journalctl -u jetson-char-recognition.service`
- 运行诊断: `python3 scripts/test_system.py`
- 监控性能: `jtop`

如需额外支持，请参考项目文档或联系开发团队。

## 部署检查清单

### 安装前检查

- [ ] 确认硬件兼容性
- [ ] 准备所需的SD卡和电源
- [ ] 下载最新的JetPack镜像
- [ ] 准备摄像头设备

### 安装过程检查

- [ ] 系统更新完成
- [ ] 所有依赖安装成功
- [ ] 模型文件下载/训练完成
- [ ] 配置文件正确设置

### 部署后验证

- [ ] 系统测试通过
- [ ] 摄像头正常工作
- [ ] 检测精度满足要求
- [ ] 性能指标达标
- [ ] 服务自动启动正常

### 生产环境检查

- [ ] 监控系统设置
- [ ] 日志记录配置
- [ ] 备份策略实施
- [ ] 安全措施到位
- [ ] 文档更新完整

通过遵循本部署指南，您可以成功在Jetson Nano上部署字符识别系统并确保其在生产环境中的稳定运行。
