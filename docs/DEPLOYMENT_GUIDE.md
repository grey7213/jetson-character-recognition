# Deployment Guide

**Language / 语言**: [English](DEPLOYMENT_GUIDE.md) | [中文](DEPLOYMENT_GUIDE_CN.md)

---

## Overview
This guide covers deploying the Jetson Character Recognition system on NVIDIA Jetson Nano hardware for production use.

## Prerequisites

### Hardware Requirements
- **NVIDIA Jetson Nano Developer Kit** (4GB recommended)
- **MicroSD Card**: 64GB Class 10 or higher
- **Camera**: USB webcam or CSI camera module
- **Power Supply**: 5V 4A barrel jack power adapter
- **Cooling**: Active cooling fan (recommended for continuous operation)

### Software Requirements
- **JetPack 4.6** or later
- **Python 3.6+**
- **CUDA 10.2+**
- **OpenCV 4.5+**
- **PyTorch 1.10+**

## Installation

### Quick Installation
```bash
# Clone the repository
git clone <repository-url>
cd jetson_character_recognition

# Run the installation script
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh --with-tests
```

### Manual Installation

#### 1. System Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-dev build-essential cmake git
```

#### 2. Install ML Frameworks
```bash
# Install OpenCV
sudo apt install -y python3-opencv

# Install PyTorch (Jetson-specific wheel)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torchvision
```

#### 3. Install Dependencies
```bash
pip3 install -r requirements.txt
```

#### 4. Install Package
```bash
pip3 install -e .
```

## Configuration

### Model Configuration
Edit `config/model_config.yaml`:
```yaml
model:
  name: "yolov8n"  # Use nano model for Jetson
  input_size: [640, 640]
  confidence_threshold: 0.5
  nms_threshold: 0.4

jetson:
  use_tensorrt: true
  fp16: true
  max_batch_size: 1
```

### Camera Configuration
Edit `config/camera_config.yaml`:
```yaml
camera:
  type: 'usb'  # or 'csi' for CSI camera
  device_id: 0

capture:
  width: 1280
  height: 720
  fps: 30
  buffer_size: 1  # Reduce latency
```

## Model Training

### Using Synthetic Dataset
```bash
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 8
```

### Using Custom Dataset
1. Prepare dataset in YOLO format
2. Update dataset configuration
3. Run training:
```bash
python3 scripts/train_model.py --dataset custom --epochs 100
```

## Model Optimization

### TensorRT Optimization
```python
from src.models.tensorrt_optimizer import TensorRTOptimizer

optimizer = TensorRTOptimizer()
optimized_model = optimizer.optimize_onnx_to_tensorrt(
    "model.onnx", 
    "model_tensorrt.engine"
)
```

### Performance Tuning
```bash
# Set maximum performance mode
sudo nvpmodel -m 0

# Maximize CPU clocks
sudo jetson_clocks

# Monitor performance
jtop
```

## Running Detection

### Real-time Detection
```bash
python3 scripts/run_detection.py models/character_detector.pt
```

### Batch Processing
```python
from src.inference.batch_processor import BatchProcessor

processor = BatchProcessor("models/character_detector.pt")
results = processor.process_directory("input_images/")
```

## Performance Optimization

### Memory Optimization
- Use batch size of 1 for real-time inference
- Enable FP16 precision
- Reduce input resolution if needed
- Set up swap file for additional memory

### Speed Optimization
- Use TensorRT optimization
- Enable GPU acceleration
- Optimize OpenCV settings
- Use threading for camera input

### Power Management
```bash
# Check power mode
sudo nvpmodel -q

# Set power mode (0 = max performance, 1 = 5W mode)
sudo nvpmodel -m 0
```

## Monitoring and Debugging

### System Monitoring
```bash
# Install jetson-stats
sudo pip3 install jetson-stats

# Monitor system
jtop
```

### Performance Monitoring
```python
from src.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor()
# Monitor during inference
summary = monitor.get_performance_summary()
```

### Logging
```python
from src.utils.logger import setup_logger

logger = setup_logger("deployment", level="INFO", log_file="deployment.log")
```

## Troubleshooting

### Common Issues

#### Low FPS
- **Cause**: Insufficient processing power or memory
- **Solution**: 
  - Reduce input resolution
  - Enable TensorRT optimization
  - Use smaller model (YOLOv8n)
  - Increase swap file size

#### High Memory Usage
- **Cause**: Large model or batch size
- **Solution**:
  - Use batch size of 1
  - Enable FP16 precision
  - Reduce model size
  - Add swap file

#### Camera Issues
- **Cause**: Camera not detected or permission issues
- **Solution**:
  - Check camera connection
  - Verify camera permissions: `sudo usermod -a -G video $USER`
  - Test camera: `v4l2-ctl --list-devices`

#### CUDA Out of Memory
- **Cause**: Insufficient GPU memory
- **Solution**:
  - Reduce batch size
  - Use smaller input resolution
  - Enable memory optimization
  - Restart system to clear memory

### Debug Commands
```bash
# Check system info
python3 scripts/test_system.py

# Test camera
python3 -c "from src.inference.camera_handler import CameraHandler; CameraHandler().test_camera()"

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Monitor resources
htop
nvidia-smi
```

## Production Deployment

### Service Setup
Create systemd service for automatic startup:

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

# Enable and start service
sudo systemctl enable jetson-char-recognition.service
sudo systemctl start jetson-char-recognition.service
```

### Auto-start on Boot
```bash
# Add to crontab
crontab -e

# Add line:
@reboot /home/jetson/jetson_character_recognition/scripts/run_detection.py /home/jetson/models/character_detector.pt
```

### Remote Monitoring
Set up SSH access for remote monitoring:
```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Monitor remotely
ssh jetson@<jetson-ip>
jtop
```

## Security Considerations

### Network Security
- Change default passwords
- Use SSH keys instead of passwords
- Configure firewall if needed
- Disable unnecessary services

### File Permissions
```bash
# Set appropriate permissions
chmod 755 scripts/*.py
chmod 644 config/*.yaml
chmod 600 models/*.pt
```

## Backup and Recovery

### Backup Configuration
```bash
# Backup configuration
tar -czf config_backup.tar.gz config/

# Backup models
tar -czf models_backup.tar.gz models/
```

### System Image Backup
```bash
# Create SD card image backup
sudo dd if=/dev/mmcblk0 of=jetson_backup.img bs=4M status=progress
```

## Performance Benchmarks

### Expected Performance (Jetson Nano 4GB)
- **Input Resolution**: 640x640
- **Model**: YOLOv8n + TensorRT
- **FPS**: 10-15 FPS
- **Inference Time**: 50-80ms
- **Memory Usage**: 1.5-2GB
- **Power Consumption**: 8-12W

### Optimization Results
| Configuration | FPS | Memory | Power |
|---------------|-----|--------|-------|
| Default | 5-8 | 2.5GB | 15W |
| TensorRT + FP16 | 10-15 | 1.8GB | 12W |
| Optimized | 12-18 | 1.5GB | 10W |

## Support and Maintenance

### Regular Maintenance
- Monitor system temperature
- Check disk space
- Update dependencies periodically
- Backup models and configurations

### Getting Help
- Check logs: `journalctl -u jetson-char-recognition.service`
- Run diagnostics: `python3 scripts/test_system.py`
- Monitor performance: `jtop`

For additional support, refer to the project documentation or contact the development team.
