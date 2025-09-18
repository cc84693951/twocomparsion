# 全景图变化检测系统

基于计算机视觉的全景图变化检测系统，支持CUDA加速处理。该系统能够检测两期全景图像之间的变化区域，适用于建筑变化监测、环境变化分析等应用场景。

## 🚀 主要功能

### 核心处理模块

1. **全景图立方体分割模块** (`PanoramaSplitter`)
   - 将全景图转换为立方体六面图
   - 支持CUDA并行加速
   - 高精度双线性插值

2. **图像预处理模块** (`ImagePreprocessor`)  
   - 图像去噪（双边滤波、高斯滤波、中值滤波）
   - 光照归一化和自适应CLAHE
   - 直方图均衡化

3. **图像配准模块** (`ImageRegistration`)
   - AKAZE特征检测和匹配
   - RANSAC单应性矩阵计算
   - 透视变换和掩码处理

4. **变化区域检测模块** (`ChangeDetector`)
   - 图像差分和阈值分割
   - 形态学操作去除噪声
   - 轮廓检测和边界框提取

5. **结果映射与全景图还原模块** (`ResultMapper`)
   - 检测结果逆向映射
   - 全景图重建和可视化
   - 结果统计和分析

### 工具模块

- **CUDA工具** (`CUDAUtils`): GPU加速的图像处理操作
- **可视化工具** (`VisualizationUtils`): 生成分析图表和报告

## 📋 系统要求

### 必需环境
```bash
Python >= 3.7
OpenCV >= 4.5.0  
NumPy >= 1.19.0
matplotlib >= 3.3.0
scikit-image >= 0.18.0
tqdm >= 4.60.0
```

### 可选环境（CUDA加速）
```bash
CuPy >= 9.0.0  # CUDA 11.x: pip install cupy-cuda11x
NVIDIA GPU with CUDA support
GPU Memory >= 4GB (推荐)
```

## 🔧 安装和配置

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd panorama_change_detection_system

# 安装基础依赖
pip install opencv-python numpy matplotlib scikit-image tqdm

# 安装CUDA支持（可选，根据CUDA版本选择）
pip install cupy-cuda11x  # CUDA 11.x
# 或
pip install cupy-cuda12x  # CUDA 12.x
```

### 2. 验证安装

```python
from main import PanoramaChangeDetectionSystem

# 创建系统实例
system = PanoramaChangeDetectionSystem()

# 查看系统信息
info = system.get_system_info()
print(info)
```

## 🎯 快速开始

### 基本使用

```python
from main import PanoramaChangeDetectionSystem

# 初始化系统
system = PanoramaChangeDetectionSystem()

# 处理全景图对
results = system.process_panorama_pair(
    panorama1_path="path/to/first_panorama.jpg",
    panorama2_path="path/to/second_panorama.jpg"
)

# 查看检测结果
print(f"检测到 {results['statistics']['total_detections']} 个变化区域")
print(f"输出目录: {results['output_directory']}")
```

### 自定义配置

```python
from config import SystemConfig, get_default_config

# 获取默认配置
config = get_default_config()

# 自定义参数
config.cuda.use_cuda = True
config.panorama_splitter.cube_size = 2048
config.change_detector.fixed_threshold = 60
config.image_preprocessor.enable_clahe = True

# 使用自定义配置
system = PanoramaChangeDetectionSystem(config=config)
```

## 📊 配置参数

### CUDA配置
```python
cuda_config = CUDAConfig(
    use_cuda=True,          # 启用CUDA加速
    device_id=0,           # GPU设备ID
    memory_pool_cleanup=True
)
```

### 全景图分割配置
```python
splitter_config = PanoramaSplitterConfig(
    cube_size=1024,         # 立方体面尺寸
    interpolation_method="bilinear",
    output_format="jpg"
)
```

### 预处理配置
```python
preprocessor_config = ImagePreprocessorConfig(
    denoise_method="bilateral",    # 去噪方法
    enable_clahe=True,            # 启用CLAHE
    clahe_clip_limit=3.0,
    enable_lighting_normalization=True
)
```

### 配准配置
```python
registration_config = ImageRegistrationConfig(
    akaze_threshold=0.001,        # AKAZE阈值
    match_ratio_threshold=0.7,    # 匹配比率
    ransac_threshold=5.0,         # RANSAC阈值
    min_match_count=10
)
```

### 变化检测配置
```python
detector_config = ChangeDetectorConfig(
    diff_method="absdiff",        # 差分方法
    threshold_method="otsu",      # 阈值方法
    min_contour_area=500,         # 最小轮廓面积
    morphology_operations=["close", "open"]
)
```

## 📁 输出结果

系统处理完成后，会在输出目录生成以下文件：

```
results/change_detection_YYYYMMDD_HHMMSS/
├── system_config.json                    # 系统配置
├── system.log                           # 处理日志
├── cube_faces_period1/                  # 第一期立方体面
│   ├── front.jpg
│   ├── right.jpg
│   └── ...
├── cube_faces_period2/                  # 第二期立方体面
├── cube_faces_preprocessed_period1/     # 预处理后立方体面
├── cube_faces_aligned_period2/          # 配准后立方体面
├── cube_faces_with_mapped_detections/   # 包含检测结果的立方体面
├── registration_info.json               # 配准信息
├── registration_summary.json            # 配准摘要
├── detection_summary.json               # 检测摘要
├── mapping_statistics.json              # 映射统计
├── final_panorama_with_detections.jpg   # 最终结果全景图
├── processing_overview_YYYYMMDD_HHMMSS.jpg  # 处理总览图
├── detailed_report_YYYYMMDD_HHMMSS.json     # 详细JSON报告
└── intermediate/                        # 中间处理结果
    └── change_detection/
        ├── front_difference.jpg
        ├── front_binary.jpg
        └── front_detection.jpg
```

## 🔍 处理流程

1. **输入验证**: 检查全景图格式和尺寸
2. **立方体分割**: 将全景图转换为6个立方体面
3. **图像预处理**: 去噪、光照归一化、对比度增强
4. **特征配准**: AKAZE特征匹配和透视变换
5. **变化检测**: 差分计算、阈值分割、轮廓提取
6. **结果映射**: 检测框映射回原始坐标
7. **全景重建**: 重建包含检测结果的全景图
8. **报告生成**: 生成可视化分析和统计报告

## 🎨 可视化功能

### 处理总览图
- 显示所有处理步骤的结果
- 包含输入、分割、配准、检测等各阶段

### 立方体面分析图
- 单个面的详细处理过程
- 配准质量评估
- 检测结果可视化

### 统计报告
- 检测数量和分布
- 配准质量指标
- 处理参数记录

## ⚡ 性能优化

### CUDA加速
- 全景图分割: 3-5x加速
- 图像预处理: 2-4x加速  
- 形态学操作: 2-3x加速

### 内存管理
- 分块处理大图像
- 自动GPU内存清理
- 优化的数据传输

### 处理策略
- 并行处理多个立方体面
- 自适应参数调整
- 增量配准验证

## 🐛 故障排除

### 常见问题

1. **CUDA初始化失败**
   ```
   解决方案: 检查CUDA版本，重新安装对应的CuPy
   pip uninstall cupy
   pip install cupy-cuda11x  # 替换为对应版本
   ```

2. **内存不足**
   ```
   解决方案: 降低立方体尺寸或禁用CUDA
   config.panorama_splitter.cube_size = 512
   config.cuda.use_cuda = False
   ```

3. **配准失败**
   ```
   解决方案: 调整AKAZE参数或降低匹配要求
   config.image_registration.akaze_threshold = 0.01
   config.image_registration.min_match_count = 5
   ```

4. **检测结果过多/过少**
   ```
   解决方案: 调整阈值和最小区域面积
   config.change_detector.fixed_threshold = 80  # 增加减少检测
   config.change_detector.min_contour_area = 1000  # 过滤小区域
   ```

### 日志分析
系统日志保存在 `system.log` 中，包含：
- 处理进度和时间
- 错误信息和警告
- 性能统计数据

## 📈 扩展开发

### 添加新的检测方法
```python
class CustomChangeDetector(ChangeDetector):
    def compute_image_difference(self, img1, img2):
        # 实现自定义差分算法
        return custom_diff_result
```

### 添加新的可视化
```python
def custom_visualization(results):
    # 实现自定义可视化
    pass
```

### 批量处理
```python
# 批量处理多对全景图
file_pairs = [("pan1_1.jpg", "pan1_2.jpg"), ("pan2_1.jpg", "pan2_2.jpg")]

for path1, path2 in file_pairs:
    results = system.process_panorama_pair(path1, path2)
    print(f"处理完成: {results['statistics']['total_detections']} 个检测")
```

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📞 联系方式

如有问题，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 首次使用时建议先在小图像上测试，确认系统正常工作后再处理大尺寸全景图。
