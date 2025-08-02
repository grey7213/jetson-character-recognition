# 故障排除指南 - Jetson Nano字符识别系统

> 🔧 **遇到问题不要慌！这里有详细的解决方案**

## 🚨 紧急情况处理

### 系统完全无法启动
**现象**：Jetson Nano开机后黑屏或无反应

**解决步骤**：
1. **检查电源**：
   ```bash
   # 确保使用5V 4A的电源适配器
   # 检查电源指示灯是否亮起
   ```

2. **检查SD卡**：
   ```bash
   # 重新制作系统镜像
   # 使用Balena Etcher重新烧录
   ```

3. **硬件检查**：
   - 检查所有连接线
   - 尝试不同的显示器
   - 检查键盘鼠标连接

### 训练过程中系统崩溃
**现象**：训练进行到一半时系统重启或死机

**立即处理**：
```bash
# 1. 重启后检查系统状态
sudo dmesg | tail -20

# 2. 检查温度
cat /sys/class/thermal/thermal_zone*/temp

# 3. 检查内存使用
free -h

# 4. 创建交换文件（如果没有）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 🔍 常见错误诊断

### 错误1：ModuleNotFoundError
**完整错误信息**：
```
ModuleNotFoundError: No module named 'ultralytics'
```

**原因分析**：Python包没有正确安装

**解决方案**：
```bash
# 方案1：重新安装依赖
pip3 install -r requirements.txt

# 方案2：单独安装缺失的包
pip3 install ultralytics

# 方案3：使用国内镜像源
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics

# 验证安装
python3 -c "import ultralytics; print('安装成功')"
```

### 错误2：CUDA out of memory
**完整错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**原因分析**：GPU内存不足

**解决方案**：
```bash
# 方案1：减少batch size
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 2

# 方案2：清理GPU内存
python3 -c "
import torch
torch.cuda.empty_cache()
print('GPU内存已清理')
"

# 方案3：重启系统
sudo reboot
```

### 错误3：Permission denied
**完整错误信息**：
```
PermissionError: [Errno 13] Permission denied: '/dev/video0'
```

**原因分析**：没有摄像头访问权限

**解决方案**：
```bash
# 方案1：添加用户到video组
sudo usermod -a -G video $USER

# 方案2：修改设备权限
sudo chmod 666 /dev/video0

# 方案3：重新登录
# 注销并重新登录系统

# 验证权限
ls -l /dev/video*
```

### 错误4：No such file or directory
**完整错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/custom/synthetic_trained.pt'
```

**原因分析**：模型文件不存在或路径错误

**解决方案**：
```bash
# 方案1：检查文件是否存在
ls -la models/custom/

# 方案2：检查当前目录
pwd
# 应该在项目根目录

# 方案3：重新训练模型
python3 scripts/train_model.py --dataset synthetic --epochs 50 --batch-size 4

# 方案4：使用绝对路径
python3 scripts/run_detection.py /home/用户名/projects/jetson-character-recognition/models/custom/synthetic_trained.pt
```

---

## 🐛 训练问题诊断

### 问题1：训练速度极慢
**现象**：每个epoch需要很长时间（超过10分钟）

**诊断步骤**：
```bash
# 1. 检查GPU使用情况
nvidia-smi

# 2. 检查CPU使用情况
htop

# 3. 检查数据集大小
du -sh data/synthetic/

# 4. 检查系统负载
uptime
```

**解决方案**：
```bash
# 方案1：优化系统性能
sudo nvpmodel -m 0
sudo jetson_clocks

# 方案2：减少数据量
python3 data/tools/data_generator.py --output data/synthetic --count 50 --yolo

# 方案3：调整训练参数
python3 scripts/train_model.py --dataset synthetic --epochs 50 --batch-size 4 --workers 2
```

### 问题2：训练准确率不提升
**现象**：准确率停留在很低的水平（<50%）

**诊断检查**：
```python
# 检查数据集质量
import os
print("数据集统计:")
for char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    char_dir = f"data/synthetic/{char}"
    if os.path.exists(char_dir):
        count = len(os.listdir(char_dir))
        print(f"{char}: {count} 张图片")
```

**解决方案**：
```bash
# 方案1：重新生成数据集
rm -rf data/synthetic/
python3 data/tools/data_generator.py --output data/synthetic --count 100 --yolo --samples

# 方案2：检查标注文件
ls data/processed/yolo_format/labels/

# 方案3：使用预训练模型
python3 scripts/train_model.py --dataset synthetic --epochs 100 --pretrained
```

### 问题3：训练中断无法恢复
**现象**：训练意外停止，重新开始时从头训练

**解决方案**：
```bash
# 查找检查点文件
find . -name "*.pt" -type f | grep -E "(last|best|checkpoint)"

# 从检查点恢复训练
python3 scripts/train_model.py --dataset synthetic --epochs 100 --resume models/custom/last.pt

# 如果没有检查点，启用自动保存
python3 scripts/train_model.py --dataset synthetic --epochs 100 --save-period 10
```

---

## 📷 摄像头问题诊断

### 问题1：摄像头无法检测
**现象**：运行实时检测时提示"No camera found"

**诊断步骤**：
```bash
# 1. 检查USB设备
lsusb | grep -i camera

# 2. 检查视频设备
ls /dev/video*

# 3. 检查设备权限
ls -l /dev/video*

# 4. 测试摄像头
v4l2-ctl --list-devices
```

**解决方案**：
```bash
# 方案1：安装摄像头驱动
sudo apt update
sudo apt install v4l-utils -y

# 方案2：重新插拔摄像头
# 物理重新连接USB摄像头

# 方案3：修改设备权限
sudo chmod 666 /dev/video0

# 方案4：测试摄像头
cheese  # 如果能打开说明摄像头正常
```

### 问题2：摄像头图像质量差
**现象**：实时检测时图像模糊或颜色异常

**解决方案**：
```bash
# 调整摄像头参数
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=128
v4l2-ctl -d /dev/video0 --set-ctrl=contrast=128
v4l2-ctl -d /dev/video0 --set-ctrl=saturation=128

# 查看当前设置
v4l2-ctl -d /dev/video0 --list-ctrls
```

### 问题3：实时检测延迟严重
**现象**：摄像头画面延迟很大，FPS很低

**解决方案**：
```python
# 修改摄像头配置
# 编辑 config/camera_config.yaml
camera:
  device_id: 0
  width: 320      # 降低分辨率
  height: 240     # 降低分辨率
  fps: 15         # 降低帧率
  buffer_size: 1  # 减少缓冲
```

---

## 💾 存储空间问题

### 问题1：磁盘空间不足
**现象**：训练时提示"No space left on device"

**诊断**：
```bash
# 检查磁盘使用情况
df -h

# 查看大文件
du -sh * | sort -hr | head -10

# 查看项目占用空间
du -sh ~/projects/jetson-character-recognition/
```

**解决方案**：
```bash
# 方案1：清理临时文件
sudo apt autoremove -y
sudo apt autoclean
rm -rf ~/.cache/pip/

# 方案2：清理训练日志
rm -rf models/custom/training_logs/old_logs/

# 方案3：压缩数据集
tar -czf data_backup.tar.gz data/synthetic/
rm -rf data/synthetic/

# 方案4：移动到外部存储
sudo mkdir /mnt/usb
sudo mount /dev/sda1 /mnt/usb
mv data/synthetic/ /mnt/usb/
ln -s /mnt/usb/synthetic/ data/synthetic
```

---

## 🌡️ 温度和性能问题

### 问题1：系统过热
**现象**：系统运行缓慢，温度过高

**监控温度**：
```bash
# 实时监控温度
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'

# 查看温度历史
dmesg | grep -i thermal
```

**解决方案**：
```bash
# 方案1：降低性能模式
sudo nvpmodel -m 1  # 使用节能模式

# 方案2：添加散热
# 安装风扇或散热片

# 方案3：降低训练强度
python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 2

# 方案4：定时休息训练
# 训练一段时间后暂停，让系统降温
```

### 问题2：性能下降
**现象**：检测速度明显变慢

**诊断**：
```bash
# 检查系统负载
top

# 检查GPU状态
nvidia-smi

# 检查内存使用
free -h

# 检查交换分区使用
swapon -s
```

**解决方案**：
```bash
# 方案1：重启系统
sudo reboot

# 方案2：清理内存
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# 方案3：优化性能
sudo nvpmodel -m 0
sudo jetson_clocks

# 方案4：关闭不必要的程序
sudo systemctl stop 不需要的服务
```

---

## 🔧 高级故障排除

### 使用日志诊断问题
```bash
# 查看系统日志
sudo journalctl -f

# 查看训练日志
tail -f models/custom/training_logs/train.log

# 查看错误日志
grep -i error /var/log/syslog

# 查看GPU日志
nvidia-smi -l 1 > gpu_log.txt &
```

### 创建诊断脚本
创建文件 `diagnose.py`：
```python
#!/usr/bin/env python3
import os
import sys
import subprocess
import torch

def run_diagnostics():
    print("=== Jetson Nano 字符识别系统诊断 ===\n")
    
    # 检查Python环境
    print(f"Python版本: {sys.version}")
    
    # 检查CUDA
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    
    # 检查项目文件
    required_files = [
        "scripts/train_model.py",
        "scripts/run_detection.py",
        "src/models/yolo_character_detector.py",
        "requirements.txt"
    ]
    
    print("\n项目文件检查:")
    for file in required_files:
        exists = "✓" if os.path.exists(file) else "✗"
        print(f"{exists} {file}")
    
    # 检查数据集
    print("\n数据集检查:")
    if os.path.exists("data/synthetic"):
        char_count = len(os.listdir("data/synthetic"))
        print(f"✓ 合成数据集存在，包含 {char_count} 个字符类别")
    else:
        print("✗ 合成数据集不存在")
    
    # 检查模型
    print("\n模型文件检查:")
    if os.path.exists("models/custom/synthetic_trained.pt"):
        print("✓ 训练好的模型存在")
    else:
        print("✗ 训练好的模型不存在")
    
    # 检查摄像头
    print("\n摄像头检查:")
    video_devices = [f for f in os.listdir("/dev") if f.startswith("video")]
    if video_devices:
        print(f"✓ 发现摄像头设备: {video_devices}")
    else:
        print("✗ 未发现摄像头设备")
    
    print("\n诊断完成！")

if __name__ == "__main__":
    run_diagnostics()
```

运行诊断：
```bash
python3 diagnose.py
```

---

## 📞 获取帮助

### 自助解决流程
1. **查看错误信息** - 仔细阅读完整的错误提示
2. **搜索本文档** - 在本页面搜索关键词
3. **运行诊断脚本** - 使用上面的诊断工具
4. **查看日志文件** - 检查详细的错误日志
5. **尝试重启** - 有时重启能解决临时问题

### 寻求帮助时请提供
- **完整的错误信息**
- **操作系统版本**：`cat /etc/os-release`
- **Python版本**：`python3 --version`
- **CUDA版本**：`nvidia-smi`
- **执行的具体命令**
- **诊断脚本的输出**

### 联系渠道
- **GitHub Issues**: https://github.com/grey7213/jetson-character-recognition/issues
- **NVIDIA开发者论坛**: https://forums.developer.nvidia.com/
- **社区讨论**: 相关技术论坛和QQ群

---

**💡 记住：大多数问题都有解决方案，保持耐心，仔细阅读错误信息！**
