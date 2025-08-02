# Jetson Nano字符识别系统 - 快速参考卡

**这是一份快速参考指南，适合已经完成初始安装的用户**

## 常用命令速查

### 系统管理
```bash
# 检查系统状态
python3 scripts/test_system.py

# 更新系统
sudo apt update && sudo apt upgrade -y

# 查看GPU状态
nvidia-smi

# 性能优化
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### 数据生成
```bash
# 生成完整数据集（每个字符100张图片）
python3 data/tools/data_generator.py --output data/synthetic --count 100 --yolo --samples

# 生成小数据集（快速测试用）
python3 data/tools/data_generator.py --output data/synthetic --count 20 --yolo --samples

# 查看生成的数据
ls data/synthetic/
```

### 模型训练
```bash
# 标准训练（推荐）
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 8

# 快速训练（测试用）
python3 scripts/train_model.py --dataset synthetic --epochs 20 --batch-size 4

# 内存不足时使用
python3 scripts/train_model.py --dataset synthetic --epochs 50 --batch-size 2
```

### 模型使用
```bash
# 实时检测
python3 scripts/run_detection.py models/custom/synthetic_trained.pt

# 演示程序
python3 scripts/demo.py

# 系统验证
python3 scripts/final_system_validation.py
```

---

## Python API 速查

### 基本检测
```python
from src.models.yolo_character_detector import YOLOCharacterDetector
import cv2

# 加载模型
detector = YOLOCharacterDetector()
detector.load_model("models/custom/synthetic_trained.pt")

# 检测图片
image = cv2.imread("test.jpg")
detections = detector.predict(image)

# 显示结果
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

### 实时检测
```python
from src.inference.realtime_detector import RealtimeCharacterDetector

# 设置回调函数
def on_detection(result):
    print(f"检测到 {len(result.detections)} 个字符")

# 启动检测
detector = RealtimeCharacterDetector("models/custom/synthetic_trained.pt")
detector.set_detection_callback(on_detection)
detector.start_detection()
```

### 批量处理
```python
from src.inference.batch_processor import BatchProcessor

# 处理整个目录
processor = BatchProcessor("models/custom/synthetic_trained.pt")
results = processor.process_directory("input_images/")

# 查看结果
for image_path, detections in results.items():
    print(f"{image_path}: {len(detections)} 个字符")
```

---

## 常见问题快速解决

### 问题：命令找不到
```bash
# 检查当前目录
pwd
# 应该在项目根目录：/home/用户名/projects/jetson-character-recognition

# 如果不在，切换到项目目录
cd ~/projects/jetson-character-recognition
```

### 问题：内存不足
```bash
# 创建交换文件
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 减少batch-size
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 2
```

### 问题：摄像头无法使用
```bash
# 检查摄像头
lsusb | grep -i camera

# 测试摄像头
sudo apt install cheese -y
cheese
```

### 问题：训练中断
```bash
# 从断点继续训练
python3 scripts/train_model.py --dataset synthetic --epochs 100 --resume models/custom/last_checkpoint.pt
```

### 问题：识别准确率低
```bash
# 重新生成更多数据
python3 data/tools/data_generator.py --output data/synthetic --count 200 --yolo

# 增加训练轮数
python3 scripts/train_model.py --dataset synthetic --epochs 200 --batch-size 8
```

---

## 重要文件位置

### 模型文件
```
models/custom/synthetic_trained.pt     # 训练好的模型
models/custom/training_logs/           # 训练日志
models/model_info.yaml                 # 模型信息
```

### 数据文件
```
data/synthetic/                        # 生成的训练数据
data/synthetic/samples/                # 样本图片
data/processed/yolo_format/            # YOLO格式数据
```

### 配置文件
```
config/model_config.yaml               # 模型配置
config/camera_config.yaml              # 摄像头配置
requirements.txt                       # Python依赖
```

### 脚本文件
```
scripts/train_model.py                 # 训练脚本
scripts/run_detection.py               # 实时检测脚本
scripts/demo.py                        # 演示脚本
scripts/test_system.py                 # 系统测试脚本
```

---

## 性能监控

### 系统资源监控
```bash
# CPU和内存使用
htop

# GPU使用情况
watch -n 1 nvidia-smi

# 磁盘使用
df -h

# 温度监控
cat /sys/class/thermal/thermal_zone*/temp
```

### 训练进度监控
训练时观察这些指标：
- **Loss（损失）**：应该逐渐下降
- **Accuracy（准确率）**：应该逐渐上升
- **Val Loss（验证损失）**：不应该持续上升
- **FPS**：处理速度，越高越好

### 检测性能监控
```python
# 简单性能测试
import time
import cv2
from src.models.yolo_character_detector import YOLOCharacterDetector

detector = YOLOCharacterDetector()
detector.load_model("models/custom/synthetic_trained.pt")

# 测试图片
image = cv2.imread("test.jpg")

# 计时测试
start_time = time.time()
detections = detector.predict(image)
end_time = time.time()

print(f"检测时间: {end_time - start_time:.3f}秒")
print(f"FPS: {1/(end_time - start_time):.1f}")
print(f"检测到: {len(detections)} 个字符")
```

---

## 自定义配置

### 调整检测参数
```python
# 在代码中调整参数
detections = detector.predict(
    image,
    confidence=0.5,      # 置信度阈值 (0-1)
    nms_threshold=0.4    # 重叠检测过滤阈值
)
```

### 修改训练参数
```bash
# 自定义训练参数
python3 scripts/train_model.py \
    --dataset synthetic \
    --epochs 150 \
    --batch-size 16 \
    --output-dir models/custom \
    --data-dir data
```

### 摄像头设置
编辑 `config/camera_config.yaml`：
```yaml
camera:
  device_id: 0          # 摄像头设备号
  width: 640            # 图像宽度
  height: 480           # 图像高度
  fps: 30               # 帧率
  auto_exposure: true   # 自动曝光
```

---

## 获取帮助

### 查看日志
```bash
# 查看训练日志
tail -f models/custom/training_logs/train.log

# 查看系统日志
journalctl -u your-service-name -f
```

### 调试模式
```bash
# 启用详细输出
python3 scripts/train_model.py --dataset synthetic --epochs 100 --verbose

# Python调试模式
python3 -u scripts/run_detection.py models/custom/synthetic_trained.pt
```

### 社区支持
- **GitHub Issues**: https://github.com/grey7213/jetson-character-recognition/issues
- **NVIDIA论坛**: https://forums.developer.nvidia.com/
- **详细文档**: [完整初学者指南](BEGINNER_GUIDE_CN.md)

---

## 快速开始流程

### 新用户（5分钟快速测试）
```bash
# 1. 生成小数据集
python3 data/tools/data_generator.py --output data/synthetic --count 10 --yolo

# 2. 快速训练
python3 scripts/train_model.py --dataset synthetic --epochs 10 --batch-size 4

# 3. 测试系统
python3 scripts/test_system.py

# 4. 运行演示
python3 scripts/demo.py
```

### 正式使用（完整流程）
```bash
# 1. 生成完整数据集
python3 data/tools/data_generator.py --output data/synthetic --count 100 --yolo --samples

# 2. 正式训练
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 8

# 3. 系统验证
python3 scripts/final_system_validation.py

# 4. 实时检测
python3 scripts/run_detection.py models/custom/synthetic_trained.pt
```

---

**提示：将此页面加入书签，方便随时查阅！**
