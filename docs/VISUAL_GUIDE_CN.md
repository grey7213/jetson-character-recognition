# 可视化操作指南 - Jetson Nano字符识别系统

> 📸 **本指南提供重要操作步骤的可视化说明**

## 🖥️ 终端界面说明

### 正确的终端界面
当您打开终端时，应该看到类似这样的界面：
```
nvidia@jetson-nano:~$ 
```

**解释**：
- `nvidia` - 您的用户名
- `jetson-nano` - 设备名称
- `~` - 表示您在用户主目录
- `$` - 命令提示符，等待您输入命令

### 项目目录结构
```
/home/nvidia/projects/jetson-character-recognition/
├── 📁 config/                    # 配置文件
│   ├── 📄 model_config.yaml     # 模型配置
│   └── 📄 camera_config.yaml    # 摄像头配置
├── 📁 data/                      # 数据文件夹
│   ├── 📁 synthetic/            # 生成的训练数据
│   │   ├── 📁 0/               # 数字0的图片
│   │   ├── 📁 1/               # 数字1的图片
│   │   ├── 📁 A/               # 字母A的图片
│   │   └── 📁 ...              # 其他字符
│   └── 📁 samples/              # 样本图片
├── 📁 docs/                      # 文档
│   ├── 📄 BEGINNER_GUIDE_CN.md  # 初学者指南
│   ├── 📄 QUICK_REFERENCE_CN.md # 快速参考
│   └── 📄 TROUBLESHOOTING_CN.md # 故障排除
├── 📁 models/                    # 模型文件
│   └── 📁 custom/               # 自定义模型
│       └── 📄 synthetic_trained.pt # 训练好的模型
├── 📁 scripts/                   # 脚本文件
│   ├── 📄 train_model.py        # 训练脚本
│   ├── 📄 run_detection.py      # 检测脚本
│   └── 📄 demo.py               # 演示脚本
└── 📁 src/                       # 源代码
    ├── 📁 models/               # 模型定义
    ├── 📁 data/                 # 数据处理
    └── 📁 inference/            # 推理代码
```

---

## 🎨 数据生成过程可视化

### 第一步：运行数据生成命令
```bash
nvidia@jetson-nano:~/projects/jetson-character-recognition$ python3 data/tools/data_generator.py --output data/synthetic --count 100 --yolo --samples
```

### 第二步：观察生成过程
您会看到类似这样的输出：
```
生成合成数据集...
参数设置:
  - 输出目录: data/synthetic
  - 每个字符图片数量: 100
  - 总字符类别: 36 (0-9, A-Z)
  - 预计生成图片总数: 3600

正在生成字符图片...
生成字符 '0' 的图片... (1/36) ████████████████████ 100%
生成字符 '1' 的图片... (2/36) ████████████████████ 100%
生成字符 '2' 的图片... (3/36) ████████████████████ 100%
...
生成字符 'Z' 的图片... (36/36) ████████████████████ 100%

✓ 数据生成完成！
✓ 总共生成了 3600 张图片
✓ YOLO格式标注文件已创建
✓ 样本图片已保存到 data/synthetic/samples/

生成统计:
  - 数字 (0-9): 1000 张图片
  - 字母 (A-Z): 2600 张图片
  - 标注文件: 3600 个
```

### 第三步：验证生成结果
```bash
nvidia@jetson-nano:~/projects/jetson-character-recognition$ ls data/synthetic/
0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z  samples
```

---

## 🎓 模型训练过程可视化

### 第一步：启动训练
```bash
nvidia@jetson-nano:~/projects/jetson-character-recognition$ python3 scripts/train_model.py --dataset synthetic --epochs 100 --batch-size 8
```

### 第二步：训练开始信息
```
=== Jetson Nano 字符识别模型训练 ===

配置信息:
  - 数据集: synthetic
  - 训练轮数: 100
  - 批次大小: 8
  - 设备: CUDA (GPU)
  - 模型: YOLOv8n

正在准备数据集...
✓ 数据集加载完成: 3600 张训练图片, 400 张验证图片
✓ 数据预处理完成
✓ 模型初始化完成

开始训练...
```

### 第三步：训练过程显示
```
Epoch 1/100: 100%|██████████████████████████| 450/450 [03:25<00:00,  2.19it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all       400       400      0.234      0.456      0.298      0.156
Train: Loss: 2.345 | Box: 1.234 | Cls: 0.567 | Obj: 0.544
Valid: Loss: 1.987 | Box: 1.123 | Cls: 0.432 | Obj: 0.432

Epoch 2/100: 100%|██████████████████████████| 450/450 [03:22<00:00,  2.22it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all       400       400      0.345      0.567      0.398      0.234
Train: Loss: 1.876 | Box: 1.023 | Cls: 0.456 | Obj: 0.397
Valid: Loss: 1.654 | Box: 0.987 | Cls: 0.345 | Obj: 0.322

...

Epoch 100/100: 100%|█████████████████████████| 450/450 [03:18<00:00,  2.27it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all       400       400      0.942      0.938      0.945      0.672
Train: Loss: 0.123 | Box: 0.067 | Cls: 0.034 | Obj: 0.022
Valid: Loss: 0.156 | Box: 0.089 | Cls: 0.045 | Obj: 0.022
```

### 第四步：训练完成信息
```
🎉 训练完成！

最终结果:
  - 训练准确率: 94.2%
  - 验证准确率: 92.8%
  - 最佳mAP50: 0.945
  - 训练时间: 3小时 45分钟 23秒

模型文件保存位置:
  ✓ 最佳模型: models/custom/synthetic_trained.pt
  ✓ 最后模型: models/custom/last.pt
  ✓ 训练日志: models/custom/training_logs/

下一步: 运行 python3 scripts/test_system.py 验证模型
```

---

## 🚀 实时检测界面说明

### 第一步：启动实时检测
```bash
nvidia@jetson-nano:~/projects/jetson-character-recognition$ python3 scripts/run_detection.py models/custom/synthetic_trained.pt
```

### 第二步：系统初始化信息
```
=== Jetson Nano 实时字符检测系统 ===

系统信息:
  - 设备: NVIDIA Jetson Nano
  - GPU内存: 3.96 GB
  - CUDA版本: 10.2
  - 模型: models/custom/synthetic_trained.pt

正在初始化...
✓ 模型加载完成
✓ 摄像头初始化完成: USB Camera (640x480)
✓ 性能监控启动
✓ 实时检测已启动

控制说明:
  - 按 'q' 键退出
  - 按 's' 键截图
  - 按 'r' 键开始/停止录制
```

### 第三步：实时检测输出
```
[实时检测中...]
FPS: 15.2 | 处理时间: 65ms | 检测数: 2
  - 字符 'A' 位置:(120, 85, 180, 145) 置信度:0.95
  - 字符 '3' 位置:(200, 150, 260, 210) 置信度:0.87

FPS: 14.8 | 处理时间: 68ms | 检测数: 1
  - 字符 'B' 位置:(150, 100, 210, 160) 置信度:0.92

FPS: 15.5 | 处理时间: 64ms | 检测数: 3
  - 字符 '5' 位置:(80, 120, 140, 180) 置信度:0.89
  - 字符 'K' 位置:(180, 90, 240, 150) 置信度:0.94
  - 字符 '9' 位置:(250, 200, 310, 260) 置信度:0.91
```

### 第四步：检测窗口界面
在屏幕上会出现一个摄像头窗口，显示：
- **实时视频画面**
- **检测到的字符周围有绿色方框**
- **每个字符上方显示识别结果和置信度**
- **窗口标题显示当前FPS**

---

## 📊 性能监控界面

### GPU使用情况
```bash
nvidia@jetson-nano:~$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA Tegra X1     On   | 00000000:00:00.0 Off |                  N/A |
| N/A   45C    P0    N/A /  N/A |   1234MiB /  3956MiB |     85%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

**解释**：
- **温度**: 45°C（正常范围）
- **内存使用**: 1234MB / 3956MB（约31%）
- **GPU利用率**: 85%（训练时正常）

### 系统资源监控
```bash
nvidia@jetson-nano:~$ htop
```
会显示类似这样的界面：
```
  1  [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]
  2  [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]
  3  [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]
  4  [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||100.0%]
  Mem[||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 3.2G/3.9G]
  Swp[||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 2.0G/4.0G]

  PID USER      PRI  NI  VIRT   RES   SHR S CPU% MEM%   TIME+  Command
 1234 nvidia     20   0 2.1G  1.2G  234M S 95.0 31.2  12:34.56 python3
```

---

## 🔧 常见界面问题说明

### 问题1：终端显示权限错误
```bash
nvidia@jetson-nano:~$ python3 scripts/train_model.py
bash: python3: command not found
```
**解决**：检查是否在正确目录，使用 `cd ~/projects/jetson-character-recognition`

### 问题2：摄像头无法打开
```bash
[ERROR] Failed to open camera device 0
[ERROR] Camera initialization failed
```
**解决**：检查摄像头连接，运行 `ls /dev/video*` 查看设备

### 问题3：GPU内存不足
```bash
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**解决**：减少batch-size，使用 `--batch-size 2`

### 问题4：训练速度异常慢
```bash
Epoch 1/100:   0%|          | 1/450 [05:23<40:15:32, 322.45s/it]
```
**解决**：检查是否启用GPU，运行性能优化命令

---

## 📱 移动端查看建议

如果您需要在手机或平板上查看这些文档：

1. **GitHub移动版**：直接在手机浏览器中访问GitHub仓库
2. **Markdown阅读器**：下载支持Markdown的阅读器应用
3. **离线查看**：将文档下载到本地查看

---

## 🎯 成功标志总结

### ✅ 数据生成成功
- 看到 `data/synthetic/` 目录包含36个字符文件夹
- 每个文件夹包含指定数量的图片文件
- 生成过程无错误信息

### ✅ 模型训练成功
- 训练准确率逐渐提升到90%以上
- 验证损失不再下降
- 生成 `synthetic_trained.pt` 模型文件

### ✅ 实时检测成功
- 摄像头窗口正常显示
- 能够识别字符并显示绿色框
- FPS保持在10以上

### ✅ 系统整体成功
- 所有测试脚本运行无错误
- 性能监控显示正常资源使用
- 能够处理各种输入图片

---

**💡 提示：保存这个页面作为参考，在操作过程中随时对照检查！**
