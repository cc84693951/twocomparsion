# 安装指南

## 环境要求

### 基本要求
- Python 3.7 或更高版本
- 操作系统: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- 内存: 8GB RAM (推荐16GB)
- 硬盘: 2GB 可用空间

### GPU加速 (可选)
- NVIDIA GPU (支持CUDA)
- CUDA Toolkit 11.0+ 或 12.0+
- GPU内存: 4GB+ (推荐8GB+)

## 快速安装

### 1. 下载项目
```bash
# 如果从Git仓库下载
git clone <repository-url>
cd panorama_change_detection_system

# 或者直接下载ZIP文件并解压
```

### 2. 安装基础依赖
```bash
# 安装必需的Python包
pip install opencv-python numpy matplotlib scikit-image tqdm pandas

# 或使用requirements.txt
pip install -r requirements.txt
```

### 3. 安装CUDA支持 (可选但推荐)
```bash
# 根据您的CUDA版本选择
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# 如果不确定CUDA版本，可以运行:
nvcc --version
```

### 4. 验证安装
```bash
python test_system.py
```

## 详细安装步骤

### Windows 安装

#### 步骤 1: 安装Python
1. 从 https://python.org 下载Python 3.7+
2. 安装时勾选 "Add Python to PATH"
3. 验证安装: `python --version`

#### 步骤 2: 安装依赖包
```cmd
# 打开命令提示符 (cmd) 或 PowerShell
pip install opencv-python numpy matplotlib scikit-image tqdm pandas
```

#### 步骤 3: 安装CUDA支持 (可选)
1. 检查是否有NVIDIA GPU: 设备管理器 > 显示适配器
2. 下载安装CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. 安装CuPy:
```cmd
pip install cupy-cuda11x  # 或 cupy-cuda12x
```

### macOS 安装

#### 步骤 1: 安装Python
```bash
# 使用Homebrew (推荐)
brew install python

# 或从python.org下载
```

#### 步骤 2: 安装依赖包
```bash
pip3 install opencv-python numpy matplotlib scikit-image tqdm pandas
```

#### 注意: macOS通常不支持NVIDIA GPU，因此跳过CUDA安装

### Ubuntu/Linux 安装

#### 步骤 1: 更新系统和安装Python
```bash
sudo apt update
sudo apt install python3 python3-pip
```

#### 步骤 2: 安装系统依赖
```bash
sudo apt install python3-opencv
# 或者
sudo apt install libopencv-dev python3-opencv
```

#### 步骤 3: 安装Python包
```bash
pip3 install numpy matplotlib scikit-image tqdm pandas
```

#### 步骤 4: 安装CUDA支持 (如果有NVIDIA GPU)
```bash
# 安装NVIDIA驱动
sudo ubuntu-drivers autoinstall

# 安装CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# 安装CuPy
pip3 install cupy-cuda11x
```

## 验证安装

### 1. 基础功能测试
```python
python -c "
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
print('所有基础包导入成功!')
"
```

### 2. CUDA测试 (如果安装了)
```python
python -c "
try:
    import cupy as cp
    a = cp.array([1, 2, 3])
    print('CUDA支持正常!')
    print(f'GPU设备: {cp.cuda.Device()}')
except Exception as e:
    print(f'CUDA不可用: {e}')
"
```

### 3. 完整系统测试
```bash
python test_system.py
```

## 常见问题解决

### 问题 1: OpenCV导入失败
**错误**: `ImportError: No module named 'cv2'`

**解决方案**:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

### 问题 2: CuPy安装失败
**错误**: `Failed building wheel for cupy`

**解决方案**:
1. 确认CUDA版本: `nvcc --version`
2. 安装对应版本:
```bash
pip install cupy-cuda11x  # 替换为正确版本
```
3. 如果仍失败，尝试预编译版本:
```bash
pip install cupy-cuda11x -f https://pip.cupy.dev/aarch64  # ARM架构
pip install cupy-cuda11x -f https://pip.cupy.dev/pre     # 预发布版本
```

### 问题 3: 内存不足
**错误**: `CUDA out of memory` 或系统内存不足

**解决方案**:
1. 减少处理尺寸:
```python
config.panorama_splitter.cube_size = 512  # 默认1024
```
2. 禁用CUDA:
```python
config.cuda.use_cuda = False
```

### 问题 4: 权限错误 (Linux/macOS)
**错误**: `Permission denied`

**解决方案**:
```bash
# 使用用户安装
pip install --user package_name

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install package_name
```

### 问题 5: 中文路径问题
**错误**: 无法读取包含中文的文件路径

**解决方案**: 
系统已内置中文路径支持，使用UTF-8编码。如仍有问题:
```python
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

## 性能优化建议

### 1. CUDA优化
- 确保GPU驱动是最新版本
- 监控GPU内存使用: `nvidia-smi`
- 调整batch size以适应GPU内存

### 2. CPU优化
- 使用多核心处理器
- 确保有足够内存 (推荐16GB)
- 关闭不必要的后台程序

### 3. 存储优化
- 使用SSD存储输入和输出文件
- 确保有足够磁盘空间 (每次处理约需要1-2GB)

## 下一步

安装完成后，您可以:

1. **运行测试**: `python test_system.py`
2. **查看示例**: `python example_usage.py`
3. **处理您的数据**: 修改`main.py`中的路径
4. **查看文档**: 阅读`README.md`了解详细功能

如果遇到其他问题，请检查:
- Python版本兼容性
- 包版本冲突
- 系统资源限制
- 防火墙或权限设置

祝您使用愉快！
