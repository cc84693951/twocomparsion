# 全景图像变化检测系统

## 📋 概述

这是一个完整的全景图像变化检测系统，整合了全景图处理和AI视觉算法，可以自动检测两期全景图像之间的变化区域。系统支持GPU加速，提供完整的可视化结果。

## 🚀 主要功能

### 核心流程
1. **全景图立方体分割** - 将全景图分解为6个立方体面
2. **图像预处理** - 去噪、直方图均衡化
3. **AKAZE特征匹配** - 高精度图像配准
4. **变化检测** - 图像差分、阈值分割
5. **目标识别** - 轮廓提取、几何过滤
6. **结果映射** - 坐标变换回全景图

### 技术特点
- ✅ GPU/CPU自适应加速
- ✅ 高精度坐标映射
- ✅ 自动配准校正
- ✅ 完整可视化输出
- ✅ 批量处理支持
- ✅ 中文路径支持

## 📦 安装依赖

### 基础依赖
```bash
pip install opencv-python numpy matplotlib tqdm
```

### GPU加速（可选）
```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x  
pip install cupy-cuda12x
```

### 图像处理（可选）
```bash
pip install scikit-image  # 用于SSIM计算
```

## 🛠 基础使用

### 快速开始
```python
from panorama_change_detection_system import PanoramaChangeDetectionSystem

# 创建检测系统
system = PanoramaChangeDetectionSystem(
    output_dir="results",
    use_cuda=True
)

# 执行检测
results = system.process_panorama_pair(
    "path/to/panorama1.jpg",
    "path/to/panorama2.jpg"
)

# 查看结果
if results['processing_successful']:
    print(f"检测到 {results['total_detection_count']} 个变化区域")
```

### 运行演示
```bash
python demo_panorama_change_detection.py
```

## ⚙️ 配置参数

### 系统配置
```python
config = {
    'cube_size': 1024,                    # 立方体面尺寸
    'diff_threshold': 30,                 # 差异检测阈值
    'min_contour_area': 500,              # 最小有效区域面积
    'max_contour_area': 50000,            # 最大有效区域面积
    'min_aspect_ratio': 0.2,              # 最小长宽比
    'max_aspect_ratio': 5.0,              # 最大长宽比
    'morphology_kernel_size': (5, 5),     # 形态学操作核大小
    'gaussian_blur_kernel': (3, 3),       # 高斯模糊核大小
    'clahe_clip_limit': 2.0,              # CLAHE对比度限制
    'clahe_tile_grid_size': (8, 8),       # CLAHE网格大小
}
```

### 参数调优指南

#### 检测灵敏度调节
- **高灵敏度**：`diff_threshold=20`, `min_contour_area=300`
- **中等灵敏度**：`diff_threshold=30`, `min_contour_area=500` (默认)
- **低灵敏度**：`diff_threshold=50`, `min_contour_area=1000`

#### 性能优化
- **高速模式**：`cube_size=512`, `use_cuda=True`
- **平衡模式**：`cube_size=1024`, `use_cuda=True` (默认)
- **高精度模式**：`cube_size=2048`, `use_cuda=True`

#### 特殊场景适配
- **室内场景**：降低 `diff_threshold` 到 20-25
- **室外场景**：提高 `diff_threshold` 到 35-45
- **复杂纹理**：增大 `morphology_kernel_size` 到 (7,7)

## 📊 输出结果

### 文件结构
```
output_directory/
├── panorama_change_detection_comprehensive_YYYYMMDD_HHMMSS.jpg  # 综合可视化
├── detection_results_YYYYMMDD_HHMMSS.json                      # 检测数据
└── final_panorama_with_detections_YYYYMMDD_HHMMSS.jpg          # 最终全景图
```

### 检测数据格式
```json
{
  "timestamp": "20250918_143022",
  "system_config": {...},
  "processing_summary": {
    "total_faces_processed": 6,
    "faces_with_detections": 2,
    "total_detections": 5,
    "panorama_bboxes_count": 3
  },
  "face_results": [...],
  "panorama_bboxes": [
    {
      "id": 1,
      "face_name": "front",
      "bbox": [100, 200, 150, 80],
      "panorama_bbox": [1200, 400, 180, 90],
      "confidence": 0.85,
      "area": 12000.0
    }
  ]
}
```

## 🔧 高级功能

### 自定义配置
```python
# 创建自定义配置
system = PanoramaChangeDetectionSystem()
system.config.update({
    'cube_size': 512,           # 更快处理
    'diff_threshold': 40,       # 更严格检测
    'min_contour_area': 300,    # 检测更小变化
})
```

### 批量处理
```python
image_pairs = [
    ("img1_period1.jpg", "img1_period2.jpg"),
    ("img2_period1.jpg", "img2_period2.jpg"),
]

for i, (img1, img2) in enumerate(image_pairs):
    system.output_dir = f"batch_results/pair_{i}"
    results = system.process_panorama_pair(img1, img2)
```

### 结果分析
```python
def analyze_results(results):
    # 分析检测质量
    for face_result in results['face_results']:
        print(f"{face_result['face_name']}: {len(face_result['bboxes'])} 检测")
        print(f"匹配质量: {face_result['match_info']['inlier_ratio']:.2%}")
```

## 🔍 故障排除

### 常见问题

#### 1. GPU初始化失败
```
⚠️ GPU初始化失败，使用CPU版本
```
**解决方案**：
- 检查CUDA安装：`nvidia-smi`
- 重新安装CuPy：`pip install cupy-cuda11x`
- 或强制使用CPU：`use_cuda=False`

#### 2. 特征匹配失败
```
⚠️ 特征点不足，跳过配准
```
**解决方案**：
- 确保图像质量良好
- 检查图像对比度和清晰度
- 降低 `cube_size` 提高特征检测
- 使用SIFT替代AKAZE

#### 3. 检测结果过多/过少
**过多检测**：
- 提高 `diff_threshold`
- 增大 `min_contour_area`
- 增大 `morphology_kernel_size`

**过少检测**：
- 降低 `diff_threshold`
- 减小 `min_contour_area`
- 检查图像配准质量

#### 4. 内存不足
```
CUDA out of memory
```
**解决方案**：
- 降低 `cube_size` 到 512
- 使用CPU模式：`use_cuda=False`
- 分批处理立方体面

### 性能优化建议

#### 硬件要求
- **最小配置**：4GB RAM, 无GPU
- **推荐配置**：8GB RAM, GTX 1060/RTX 2060
- **高性能配置**：16GB RAM, RTX 3080/4080

#### 图像要求
- **格式**：JPG, PNG
- **尺寸**：建议 2048×1024 或以上
- **质量**：清晰度良好，无过度压缩
- **对齐**：两期图像拍摄位置基本一致

#### 处理时间参考
| 配置 | 立方体尺寸 | GPU | 处理时间 |
|------|------------|-----|----------|
| 最小 | 512×512 | 无 | ~30秒 |
| 标准 | 1024×1024 | GTX 1060 | ~15秒 |
| 高精度 | 2048×2048 | RTX 3080 | ~25秒 |

## 📚 技术原理

### 坐标变换
全景图 ↔ 立方体面的坐标转换基于球面几何：

```python
# 全景图 → 球面坐标
theta = (u / width) * 2π - π
phi = (v / height) * π

# 球面坐标 → 3D坐标
x = sin(phi) * sin(theta)
y = cos(phi)  
z = sin(phi) * cos(theta)

# 3D坐标 → 立方体面坐标
face_u = (x/z + 1) * 0.5 * cube_size  # front面示例
face_v = (-y/z + 1) * 0.5 * cube_size
```

### 特征匹配
使用AKAZE算法进行特征检测和匹配：
- **优势**：对旋转、缩放、光照变化鲁棒
- **速度**：比SIFT快2-3倍
- **精度**：适中，满足全景图配准需求

### 变化检测
多层次检测策略：
1. **像素级差分**：基础变化检测
2. **形态学操作**：连接和过滤
3. **几何约束**：面积、长宽比过滤
4. **置信度评估**：基于形状特征

## 📈 扩展开发

### 添加新的检测算法
```python
class CustomChangeDetectionSystem(PanoramaChangeDetectionSystem):
    def custom_detection_method(self, img1, img2):
        # 自定义检测算法
        pass
        
    def process_face_pair(self, face1, face2, face_name):
        # 重写处理逻辑
        result = super().process_face_pair(face1, face2, face_name)
        # 添加自定义处理
        return result
```

### 集成其他AI模型
```python
def integrate_ai_model(self, face_img):
    # 集成深度学习模型
    # 如：YOLO、Mask R-CNN等
    detections = ai_model.detect(face_img)
    return detections
```

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📞 支持

如有问题，请在 GitHub 上创建 Issue 或联系开发团队。 