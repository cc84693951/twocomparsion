# 中文字体显示问题解决方案

## 问题描述
在生成的图表中，中文文字显示为方框(□)或乱码，无法正常显示。

## 解决方案

### 1. 自动字体检测与配置
已在 `utils/visualization.py` 中实现了跨平台的中文字体自动检测：

```python
def setup_chinese_fonts():
    """配置中文字体支持"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        fonts = ['Heiti TC', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    elif system == 'Windows':  # Windows
        fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'DejaVu Sans']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans', 'Liberation Sans']
```

### 2. 支持的系统及字体

#### macOS系统
- ✅ Heiti TC (黑体繁体) - **当前使用**
- ✅ PingFang SC (苹方简体)
- ✅ Hiragino Sans GB (冬青黑体)
- ✅ STHeiti (华文黑体)
- ✅ Arial Unicode MS

#### Windows系统
- SimHei (黑体)
- Microsoft YaHei (微软雅黑)
- KaiTi (楷体)
- FangSong (仿宋)
- STSong (华文宋体)

#### Linux系统
- WenQuanYi Micro Hei (文泉驿微米黑)
- WenQuanYi Zen Hei (文泉驿正黑)
- Noto Sans CJK SC (思源黑体简体)

### 3. 测试验证
运行字体测试脚本验证配置：
```bash
python test_chinese_fonts.py
```

#### 测试结果示例：
```
使用中文字体: Heiti TC
当前字体设置: ['Heiti TC', 'DejaVu Sans', ...]
✅ 测试图片已保存: chinese_font_test_20250918_154317.png
```

### 4. 已解决的问题

#### ✅ 前：方框显示
```
统计信息： □□□ 
检测结果： □□□□
置信度：   □□□
```

#### ✅ 后：正常中文显示
```
统计信息： 检测统计
检测结果： 变化区域检测
置信度：   高置信度
```

### 5. 常见问题解决

#### 问题1：仍然显示方框
**原因**：系统缺少中文字体  
**解决**：安装中文字体包
```bash
# macOS - 通常已预装
# Windows - 通常已预装
# Ubuntu/Debian
sudo apt-get install fonts-noto-cjk fonts-wqy-zenhei
# CentOS/RHEL
sudo yum install google-noto-cjk-fonts wqy-zenhei-fonts
```

#### 问题2：部分emoji显示警告
**现象**：📊 📈 等emoji字符显示警告  
**影响**：不影响中文字符正常显示  
**解决**：可忽略或移除emoji字符

### 6. 手动配置选项
如果自动配置失败，可手动指定字体：

```python
import matplotlib.pyplot as plt

# 方法1：全局设置
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # macOS
plt.rcParams['font.sans-serif'] = ['SimHei']    # Windows
plt.rcParams['axes.unicode_minus'] = False

# 方法2：临时设置
plt.title('中文标题', fontfamily='Heiti TC')
```

### 7. 验证步骤

1. **运行字体测试**
   ```bash
   cd panorama_change_detection_system
   python test_chinese_fonts.py
   ```

2. **检查输出**
   - 控制台显示 "使用中文字体: [字体名]"
   - 生成测试图片文件
   - 打开图片检查中文显示

3. **运行完整系统**
   ```bash
   python main.py
   ```

### 8. 技术实现细节

- **自动检测**：扫描系统可用字体列表
- **优先级排序**：按平台选择最适合的中文字体
- **回退机制**：如果没有中文字体，使用Unicode兼容字体
- **全局配置**：一次设置，所有matplotlib图表生效

### 9. 性能影响
- 字体检测仅在导入时执行一次
- 不影响图表生成性能
- 内存使用增加微乎其微

---

## 更新记录
- 2024-09-18: 实现跨平台中文字体自动配置
- 2024-09-18: 添加字体测试验证脚本
- 2024-09-18: 验证macOS Heiti TC字体正常工作
