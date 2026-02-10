# KITCHEN-MONITOR

[English](./README.md) | [中文](./README_CN.md)

## 项目概述

Kitchen Monitor 是一个用于厨房炊具检测和烹饪状态分类的计算机视觉项目，采用基于YOLO的目标检测、时序分类模型，以及在树莓派5上实现带语音提示的实时监控。

## 项目结构

### 文档
- **`README.md`** - 项目文档（英文）
- **`README_CN.md`** - 项目文档（中文）
- **`FINE_TUNING_JOURNEY.md`** - 模型微调过程文档（英文）
- **`FINE_TUNING_JOURNEY_CN.md`** - 模型微调过程文档（中文）

### 模型
- **`yolo_training/kitchenware_detector2/weights/best.pt`** - **阶段1**：YOLOv8n目标检测模型，用于检测锅具
  - 497张训练图像，2个类别（cooking-pot炒锅，frying-pan平底锅）
  - 性能：99.9%精确率，100%召回率
- **`yolo_training/pan_pot_classifier_temporal.pth`** - **阶段2**：MobileNet v2时序分类模型，用于烹饪状态检测
  - 143个时序组（每组3帧），4个类别（boiling沸腾、normal正常、on_fire着火、smoking冒烟）
  - 性能：93.10%验证准确率，92.9%测试准确率

### 脚本
- **`start_monitoring.py`** - 树莓派5实时监控脚本，支持摄像头和语音提示
- **`test_audio.py`** - 音频播放测试脚本（连续播放assets/sound/中的所有MP3文件）
- **`train_classifier_temporal.py`** - 时序分类器训练脚本
- **`verify_yolo.py`** - YOLO模型验证脚本
- **`verify_temporal_on_originals.py`** - 在原始图像上进行时序模型验证
- **`resize_and_copy_images.py`** - 图像预处理工具

### 目录

#### `modules/`
监控系统核心Python模块。
- **`yolo_detector.py`** - 基于YOLO的厨具目标检测器
- **`dataset.py`** - 数据集加载和预处理
- **`__init__.py`** - 包初始化

#### `assets/sound/`
实时监控系统的语音提示文件。
- **`kitchenware_0.mp3` ~ `kitchenware_6.mp3`** - 厨具数量播报（0-6件）
- **`kitchenware_x.mp3`** - 厨具数量播报（超过6件）
- **`statues_normal.mp3`** - 正常烹饪状态提示
- **`statues_boiling.mp3`** - 沸腾状态提示
- **`statues_smoking.mp3`** - 冒烟状态提示
- **`statues_on-fire.mp3`** - 着火状态提示

#### `yolo_training/`
YOLO模型训练结果和产物。
- **`kitchenware_detector/`** - 第一次训练运行结果
- **`kitchenware_detector2/`** - 第二次训练运行（使用更多数据的改进模型）

#### `verification_results_on_originals/`
在原始图像上的模型验证输出。
- 标注图像，显示检测结果
- `verification_results_on_originals.json` - 验证指标和结果

## 主要功能

- **实时监控** - 在树莓派5上处理实时摄像头画面
- **YOLO目标检测** - 使用微调的YOLOv8n进行实时厨具检测
- **时序分类** - 3帧时序模型检测烹饪状态
- **语音提示** - 通过蓝牙音箱播报检测到的状态
- **多类别检测** - 检测厨具数量并分类烹饪状态（沸腾、正常、着火、冒烟）
- **边缘部署** - 针对树莓派5优化，采用MobileNet v2骨干网络（350万参数）

## 模型性能

### 阶段1：YOLO检测器（目标检测）
- 模型：`yolo_training/kitchenware_detector2/weights/best.pt`
- 类别：cooking-pot（炒锅）、frying-pan（平底锅）
- 性能：99.9%精确率，100%召回率

### 阶段2：时序分类器（状态分类）
- 模型：`yolo_training/pan_pot_classifier_temporal.pth`
- 类别：boiling（沸腾）、normal（正常）、on_fire（着火）、smoking（冒烟）
- 性能：93.10%验证准确率，92.9%测试准确率

## 使用方法

### 实时监控（树莓派5）

```bash
python start_monitoring.py
```

启动实时监控循环：
1. 以30 FPS从Pi摄像头捕获3个连续帧
2. 将帧缩放至最大512x512像素
3. 对每帧运行YOLO检测
4. 将裁剪区域输入时序分类器
5. 通过音箱播放语音提示（厨具数量和烹饪状态）

### 测试音频播放

```bash
python test_audio.py
```

连续播放 `assets/sound/` 中的所有MP3文件，用于验证音频输出是否正常工作。

### 验证YOLO模型
```bash
python verify_yolo.py
```

### 验证时序模型
```bash
python verify_temporal_on_originals.py
```

## 依赖要求

### PC（Windows/Linux x86_64）

```bash
# 可选：创建并激活虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 树莓派5（ARM64）

```bash
# 创建带系统包的虚拟环境（Picamera2和libcamera需要）
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 安装 PyTorch 轮子（ARM64）
pip3 install https://github.com/KumaTea/pytorch-aarch64/releases/download/v2.3.0/torch-2.3.0a0+gitc8f7e6d-cp311-cp311-linux_aarch64.whl
pip3 install https://github.com/KumaTea/pytorch-aarch64/releases/download/v2.3.0/torchvision-0.18.0a0+gitc8f7e6d-cp311-cp311-linux_aarch64.whl

# 使用 apt 安装 OpenCV（Pi上更稳定）
sudo apt update
sudo apt install -y python3-opencv

# 安装 Picamera2（用于Pi摄像头）
sudo apt install -y python3-picamera2 python3-libcamera

# 安装音频播放器（用于语音提示）
sudo apt install -y mpg123
# 或者
sudo apt install -y ffmpeg

# 安装其余依赖
pip3 install -r requirements.txt
```

### 树莓派5蓝牙音箱设置

```bash
# 安装 PipeWire 蓝牙音频支持
sudo apt install -y libspa-0.2-bluetooth pipewire-audio

# 重启 PipeWire
systemctl --user restart pipewire pipewire-pulse

# 连接蓝牙音箱
bluetoothctl connect <MAC地址>

# 验证蓝牙设备出现在音频输出中
pactl list sinks short

# 设置蓝牙音箱为默认输出
pactl set-default-sink bluez_output.<MAC地址>.1

# 测试音频
mpg123 ./assets/sound/kitchenware_1.mp3
```

## 文档

有关微调过程的详细信息：
- 英文：[FINE_TUNING_JOURNEY.md](FINE_TUNING_JOURNEY.md)
- 中文：[FINE_TUNING_JOURNEY_CN.md](FINE_TUNING_JOURNEY_CN.md)

## 仓库内容

本仓库包含：
- 训练好的模型（PyTorch权重）
- 训练结果和指标
- 树莓派5实时监控脚本
- 语音提示音频文件
- 验证脚本和结果
- 文档（英文和中文）

**注意**：训练数据、标注工具（X-AnyLabeling、ChatRex）和实用脚本未包含在版本控制中。

## 许可证

详见各个组件的许可证。
