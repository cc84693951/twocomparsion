#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
360度全景图六图分割工具 - 无乱码版
快速将大疆无人机的360度全景图转换为六个平面图像
使用英文文件名避免Windows系统乱码问题
"""

import cv2
import numpy as np
import os
from datetime import datetime
import math

def equirectangular_to_cube_face(panorama_img, face_type, cube_size=1024):
    """
    将等距圆柱投影转换为立方体的单个面
    
    Args:
        panorama_img: 全景图像
        face_type: 面类型 (0-5: front, right, back, left, top, bottom)
        cube_size: 输出图像尺寸
    """
    height, width = panorama_img.shape[:2]
    face_img = np.zeros((cube_size, cube_size, 3), dtype=np.uint8)
    
    for i in range(cube_size):
        for j in range(cube_size):
            # 标准化坐标到[-1, 1]
            x = (2.0 * j / cube_size) - 1.0
            y = (2.0 * i / cube_size) - 1.0
            
            # 根据面类型计算3D坐标
            if face_type == 0:    # front
                xyz = [x, -y, 1.0]
            elif face_type == 1:  # right
                xyz = [1.0, -y, -x]
            elif face_type == 2:  # back
                xyz = [-x, -y, -1.0]
            elif face_type == 3:  # left
                xyz = [-1.0, -y, x]
            elif face_type == 4:  # top
                xyz = [x, 1.0, y]
            elif face_type == 5:  # bottom
                xyz = [x, -1.0, -y]
            
            # 转换为球面坐标
            x3d, y3d, z3d = xyz
            r = math.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
            
            # 计算球面角度
            theta = math.atan2(x3d, z3d)  # 方位角
            phi = math.acos(y3d / r)      # 极角
            
            # 转换为全景图坐标
            u = (theta + math.pi) / (2 * math.pi) * width
            v = phi / math.pi * height
            
            # 边界检查和像素采样
            if 0 <= u < width and 0 <= v < height:
                u_int, v_int = int(u), int(v)
                face_img[i, j] = panorama_img[v_int, u_int]
    
    return face_img

def convert_panorama_to_cubemap(input_path, output_dir=None, cube_size=1024):
    """
    将全景图转换为立方体贴图的六个面
    
    Args:
        input_path: 输入全景图路径
        output_dir: 输出目录（可选）
        cube_size: 立方体面尺寸
    """
    # 读取图像
    print(f"📖 读取全景图: {os.path.basename(input_path)}")
    panorama = cv2.imread(input_path)
    
    if panorama is None:
        raise ValueError(f"无法读取图像: {input_path}")
    
    print(f"📏 原图尺寸: {panorama.shape[1]}x{panorama.shape[0]}")
    
    # 创建输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"cubemap_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 面名称 - 使用英文避免乱码
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    face_descriptions = {
        'front': '前面 (正前方)',
        'right': '右面 (右侧)', 
        'back': '后面 (正后方)',
        'left': '左面 (左侧)',
        'top': '上面 (天空)',
        'bottom': '下面 (地面)'
    }
    
    print(f"🔄 开始转换为 {cube_size}x{cube_size} 立方体贴图...")
    
    # 转换每个面
    faces = {}
    for i, face_name in enumerate(face_names):
        description = face_descriptions[face_name]
        print(f"  🎯 处理 {description} ({i+1}/6)...")
        
        face_img = equirectangular_to_cube_face(panorama, i, cube_size)
        faces[face_name] = face_img
        
        # 保存单个面 - 使用英文文件名
        filename = f"{face_name}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, face_img)
        print(f"    ✅ 已保存: {filename}")
    
    # 创建组合预览图
    print("🖼️  创建预览图...")
    create_cubemap_preview(faces, output_dir, cube_size)
    
    # 创建说明文件
    create_readme_file(output_dir, input_path, cube_size)
    
    print(f"\n🎉 转换完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 输出尺寸: {cube_size}x{cube_size}")
    print(f"📄 共生成: 6个面图像 + 1个预览图 + 1个说明文件")
    
    return faces, output_dir

def create_cubemap_preview(faces, output_dir, cube_size):
    """
    创建立方体贴图预览图（十字形布局）
    """
    # 创建4x3的布局
    preview_width = cube_size * 4
    preview_height = cube_size * 3
    preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
    
    # 标准立方体贴图布局
    #       [top]
    # [left][front][right][back]
    #       [bottom]
    
    # 放置各个面
    preview[0:cube_size, cube_size:cube_size*2] = faces['top']                    # 上
    preview[cube_size:cube_size*2, 0:cube_size] = faces['left']                  # 左
    preview[cube_size:cube_size*2, cube_size:cube_size*2] = faces['front']       # 前
    preview[cube_size:cube_size*2, cube_size*2:cube_size*3] = faces['right']     # 右
    preview[cube_size:cube_size*2, cube_size*3:cube_size*4] = faces['back']      # 后
    preview[cube_size*2:cube_size*3, cube_size:cube_size*2] = faces['bottom']    # 下
    
    # 保存预览图
    preview_path = os.path.join(output_dir, "cubemap_preview.jpg")
    cv2.imwrite(preview_path, preview)
    print(f"    ✅ 预览图: cubemap_preview.jpg")

def create_readme_file(output_dir, input_path, cube_size):
    """
    创建说明文件
    """
    readme_content = f"""# 立方体贴图转换结果

## 输入信息
- 原始文件: {os.path.basename(input_path)}
- 转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 输出尺寸: {cube_size}x{cube_size}

## 文件说明

### 立方体面图像
- front.jpg   - 前面 (正前方视图)
- right.jpg   - 右面 (右侧视图)
- back.jpg    - 后面 (正后方视图)
- left.jpg    - 左面 (左侧视图)
- top.jpg     - 上面 (天空视图)
- bottom.jpg  - 下面 (地面视图)

### 预览图
- cubemap_preview.jpg - 十字形布局的组合预览图

## 布局说明

预览图采用标准立方体贴图布局：
```
        [top]
[left] [front] [right] [back]
        [bottom]
```

## 使用建议
- 这些图像可用于VR内容制作、游戏开发等
- 每个面都是正方形，可直接用作立方体贴图
- 如需其他格式或尺寸，可重新运行转换程序
"""
    
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"    ✅ 说明文件: README.txt")

def main():
    """主函数 - 处理指定的测试图片"""
    input_image = r"C:\Users\admin\Desktop\two-phase comparison\test\20250910163759_0001_V.jpeg"
    
    # 检查文件
    if not os.path.exists(input_image):
        print(f"❌ 找不到文件: {input_image}")
        return
    
    try:
        # 执行转换
        faces, output_dir = convert_panorama_to_cubemap(
            input_path=input_image,
            cube_size=1024  # 可以调整为 512, 1024, 2048 等
        )
        
        print(f"\n✨ 全景图六图分割完成！")
        print(f"🔗 查看结果: {os.path.abspath(output_dir)}")
        
        # 显示文件列表
        print(f"\n📋 生成的文件:")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  📄 {file} ({file_size:.1f} KB)")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 