#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图逆向转换与坐标映射系统
1. 全景图 → 立方体贴图（分割）
2. 立方体贴图 → 全景图（合并）
3. 立方体面坐标 → 全景图坐标（映射）
4. 目标检测框映射到原图
"""

import cv2
import numpy as np
import os
import json
import math
from datetime import datetime

class PanoramaCubemapConverter:
    def __init__(self):
        self.face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        self.face_descriptions = {
            'front': '前面', 'right': '右面', 'back': '后面',
            'left': '左面', 'top': '上面', 'bottom': '下面'
        }
    
    def panorama_to_cubemap(self, panorama_img, cube_size=1024):
        """
        全景图转换为立方体贴图
        """
        print("🔄 全景图 → 立方体贴图转换中...")
        faces = {}
        
        for i, face_name in enumerate(self.face_names):
            print(f"  处理 {self.face_descriptions[face_name]} ({i+1}/6)...")
            faces[face_name] = self._equirectangular_to_cube_face(panorama_img, i, cube_size)
        
        return faces
    
    def cubemap_to_panorama(self, faces, output_width=None, output_height=None):
        """
        立方体贴图转换回全景图 - 改进版本
        """
        cube_size = faces['front'].shape[0]
        
        # 默认输出尺寸为2:1比例
        if output_width is None:
            output_width = cube_size * 4
        if output_height is None:
            output_height = cube_size * 2
        
        print(f"🔄 立方体贴图 → 全景图转换中 ({output_width}x{output_height})...")
        
        panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        for v in range(output_height):
            for u in range(output_width):
                # 全景图坐标转换为球面坐标
                theta = (u / output_width) * 2 * math.pi - math.pi
                phi = (v / output_height) * math.pi
                
                # 球面坐标转换为3D坐标
                x = math.sin(phi) * math.sin(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.cos(theta)
                
                # 确定属于哪个立方体面
                face_name, face_u, face_v = self._xyz_to_cube_face(x, y, z, cube_size)
                
                if face_name and 0 <= face_u < cube_size and 0 <= face_v < cube_size:
                    # 使用双线性插值提高质量
                    pixel_value = self._bilinear_interpolate(faces[face_name], face_u, face_v)
                    panorama[v, u] = pixel_value
        
        return panorama
    
    def map_bbox_to_panorama(self, bbox, face_name, cube_size, panorama_width, panorama_height):
        """
        将立方体面上的检测框映射到全景图坐标
        
        Args:
            bbox: [x1, y1, x2, y2] 在立方体面上的坐标
            face_name: 面名称
            cube_size: 立方体面尺寸
            panorama_width, panorama_height: 全景图尺寸
        
        Returns:
            mapped_points: 映射到全景图的坐标点列表
        """
        x1, y1, x2, y2 = bbox
        
        # 获取检测框的四个角点和边界点
        points = []
        
        # 四个角点
        corner_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # 边界上的额外点（提高映射精度）
        edge_points = []
        num_edge_points = 10
        
        # 上边和下边
        for i in range(num_edge_points + 1):
            t = i / num_edge_points
            edge_points.append((x1 + t * (x2 - x1), y1))  # 上边
            edge_points.append((x1 + t * (x2 - x1), y2))  # 下边
        
        # 左边和右边
        for i in range(1, num_edge_points):  # 避免重复角点
            t = i / num_edge_points
            edge_points.append((x1, y1 + t * (y2 - y1)))  # 左边
            edge_points.append((x2, y1 + t * (y2 - y1)))  # 右边
        
        all_points = corner_points + edge_points
        
        # 将所有点映射到全景图坐标
        mapped_points = []
        for face_x, face_y in all_points:
            panorama_coords = self._face_coord_to_panorama(
                face_x, face_y, face_name, cube_size, panorama_width, panorama_height
            )
            if panorama_coords:
                mapped_points.append(panorama_coords)
        
        return mapped_points
    
    def _face_coord_to_panorama(self, face_x, face_y, face_name, cube_size, panorama_width, panorama_height):
        """
        将立方体面坐标转换为全景图坐标
        """
        # 标准化到[-1, 1]
        x = (2.0 * face_x / cube_size) - 1.0
        y = (2.0 * face_y / cube_size) - 1.0
        
        # 根据面类型计算3D坐标
        face_index = self.face_names.index(face_name)
        
        if face_index == 0:    # front
            xyz = [x, -y, 1.0]
        elif face_index == 1:  # right
            xyz = [1.0, -y, -x]
        elif face_index == 2:  # back
            xyz = [-x, -y, -1.0]
        elif face_index == 3:  # left
            xyz = [-1.0, -y, x]
        elif face_index == 4:  # top
            xyz = [x, 1.0, y]
        elif face_index == 5:  # bottom
            xyz = [x, -1.0, -y]
        
        x3d, y3d, z3d = xyz
        
        # 转换为球面坐标
        r = math.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
        theta = math.atan2(x3d, z3d)
        phi = math.acos(y3d / r)
        
        # 转换为全景图坐标
        u = (theta + math.pi) / (2 * math.pi) * panorama_width
        v = phi / math.pi * panorama_height
        
        return (u, v)
    
    def _equirectangular_to_cube_face(self, panorama_img, face_type, cube_size):
        """转换全景图到立方体面"""
        height, width = panorama_img.shape[:2]
        face_img = np.zeros((cube_size, cube_size, 3), dtype=np.uint8)
        
        for i in range(cube_size):
            for j in range(cube_size):
                x = (2.0 * j / cube_size) - 1.0
                y = (2.0 * i / cube_size) - 1.0
                
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
                
                x3d, y3d, z3d = xyz
                r = math.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
                theta = math.atan2(x3d, z3d)
                phi = math.acos(y3d / r)
                
                u = (theta + math.pi) / (2 * math.pi) * width
                v = phi / math.pi * height
                
                if 0 <= u < width and 0 <= v < height:
                    face_img[i, j] = panorama_img[int(v), int(u)]
        
        return face_img
    
    def _xyz_to_cube_face(self, x, y, z, cube_size):
        """确定3D坐标属于哪个立方体面 - 修复版本"""
        abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
        
        # 确定主要方向
        if abs_z >= abs_x and abs_z >= abs_y:
            if z > 0:  # front
                face_name = 'front'
                # 保持与正向转换一致：xyz = [x, -y, 1.0]
                # 逆向：x = face_x, -y = face_y -> face_x = x, face_y = -y
                face_u = (x / z + 1) * 0.5 * cube_size
                face_v = (-y / z + 1) * 0.5 * cube_size
            else:  # back
                face_name = 'back'
                # 保持与正向转换一致：xyz = [-x, -y, -1.0]
                # 逆向：-x = face_x, -y = face_y -> face_x = -x, face_y = -y
                face_u = (-x / (-z) + 1) * 0.5 * cube_size
                face_v = (-y / (-z) + 1) * 0.5 * cube_size
        elif abs_x >= abs_y:
            if x > 0:  # right
                face_name = 'right'
                # 保持与正向转换一致：xyz = [1.0, -y, -x]
                # 逆向：1.0 = face_x, -y = face_y, -x = face_z -> face_x = z/x, face_y = -y/x
                face_u = (-z / x + 1) * 0.5 * cube_size
                face_v = (-y / x + 1) * 0.5 * cube_size
            else:  # left
                face_name = 'left'
                # 保持与正向转换一致：xyz = [-1.0, -y, x]
                # 逆向：-1.0 = face_x, -y = face_y, x = face_z -> face_x = z/(-x), face_y = -y/(-x)
                face_u = (z / (-x) + 1) * 0.5 * cube_size
                face_v = (-y / (-x) + 1) * 0.5 * cube_size
        else:
            if y > 0:  # top
                face_name = 'top'
                # 保持与正向转换一致：xyz = [x, 1.0, y]
                # 逆向：x = face_x, 1.0 = face_y, y = face_z -> face_x = x/y, face_z = z/y
                face_u = (x / y + 1) * 0.5 * cube_size
                face_v = (z / y + 1) * 0.5 * cube_size
            else:  # bottom
                face_name = 'bottom'
                # 保持与正向转换一致：xyz = [x, -1.0, -y]
                # 逆向：x = face_x, -1.0 = face_y, -y = face_z -> face_x = x/(-y), face_z = -z/(-y)
                face_u = (x / (-y) + 1) * 0.5 * cube_size
                face_v = (-z / (-y) + 1) * 0.5 * cube_size
        
        return face_name, face_u, face_v

    def _bilinear_interpolate(self, img, x, y):
        """
        双线性插值
        """
        h, w = img.shape[:2]
        
        # 边界检查
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # 获取四个最近的像素
        x1, y1 = int(math.floor(x)), int(math.floor(y))
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        # 计算权重
        dx = x - x1
        dy = y - y1
        
        # 双线性插值
        pixel = (1 - dx) * (1 - dy) * img[y1, x1] + \
                dx * (1 - dy) * img[y1, x2] + \
                (1 - dx) * dy * img[y2, x1] + \
                dx * dy * img[y2, x2]
        
        return pixel.astype(np.uint8)

class DetectionMapper:
    def __init__(self, converter):
        self.converter = converter
    
    def create_detection_demo(self, faces, cube_size, panorama_width, panorama_height):
        """
        创建检测框映射演示
        """
        print("🎯 创建目标检测映射演示...")
        
        # 模拟一些检测框
        demo_detections = {
            'front': [[200, 300, 400, 500], [600, 200, 800, 400]],
            'right': [[100, 150, 300, 350]],
            'top': [[400, 400, 600, 600]],
        }
        
        # 在立方体面上绘制检测框
        faces_with_boxes = {}
        for face_name in self.converter.face_names:
            face_with_box = faces[face_name].copy()
            
            if face_name in demo_detections:
                for bbox in demo_detections[face_name]:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(face_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(face_with_box, f'Det-{face_name}', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            faces_with_boxes[face_name] = face_with_box
        
        # 重建全景图
        panorama_reconstructed = self.converter.cubemap_to_panorama(
            faces, panorama_width, panorama_height
        )
        
        # 映射检测框到全景图
        panorama_with_mapped_boxes = panorama_reconstructed.copy()
        
        for face_name, bboxes in demo_detections.items():
            for i, bbox in enumerate(bboxes):
                # 获取映射点
                mapped_points = self.converter.map_bbox_to_panorama(
                    bbox, face_name, cube_size, panorama_width, panorama_height
                )
                
                if mapped_points:
                    # 绘制映射区域（使用凸包）
                    points_array = np.array(mapped_points, dtype=np.int32)
                    
                    # 处理跨越边界的情况
                    hull = cv2.convexHull(points_array)
                    cv2.polylines(panorama_with_mapped_boxes, [hull], True, (255, 0, 0), 2)
                    
                    # 添加标签
                    center = np.mean(points_array, axis=0).astype(int)
                    cv2.putText(panorama_with_mapped_boxes, f'{face_name}-{i+1}', 
                              tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return faces_with_boxes, panorama_with_mapped_boxes
    
    def save_mapping_results(self, faces_with_boxes, panorama_mapped, output_dir):
        """保存映射结果"""
        # 保存带检测框的立方体面
        faces_dir = os.path.join(output_dir, 'faces_with_detections')
        os.makedirs(faces_dir, exist_ok=True)
        
        for face_name, face_img in faces_with_boxes.items():
            face_path = os.path.join(faces_dir, f'{face_name}_detected.jpg')
            cv2.imwrite(face_path, face_img)
        
        # 保存映射后的全景图
        panorama_path = os.path.join(output_dir, 'panorama_with_mapped_detections.jpg')
        cv2.imwrite(panorama_path, panorama_mapped)
        
        print(f"✅ 映射结果已保存到: {output_dir}")
        return faces_dir, panorama_path

def analyze_conversion_quality(original_panorama, reconstructed_panorama):
    """
    分析转换质量和信息损失
    """
    print("📊 分析转换质量...")
    
    # 确保尺寸一致
    if original_panorama.shape != reconstructed_panorama.shape:
        reconstructed_panorama = cv2.resize(reconstructed_panorama, 
                                          (original_panorama.shape[1], original_panorama.shape[0]))
    
    # 计算各种质量指标
    # 1. MSE (均方误差)
    mse = np.mean((original_panorama.astype(float) - reconstructed_panorama.astype(float)) ** 2)
    
    # 2. PSNR (峰值信噪比)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    
    # 3. SSIM (结构相似性指数) - 简化版本
    def ssim_simple(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    # 转换为灰度图计算SSIM
    gray1 = cv2.cvtColor(original_panorama, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reconstructed_panorama, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim_simple(gray1, gray2)
    
    quality_report = {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_value,
        'quality_assessment': 'excellent' if psnr > 30 else 'good' if psnr > 25 else 'fair'
    }
    
    print(f"  📈 MSE: {mse:.2f}")
    print(f"  📈 PSNR: {psnr:.2f} dB")
    print(f"  📈 SSIM: {ssim_value:.4f}")
    print(f"  📈 质量评估: {quality_report['quality_assessment']}")
    
    return quality_report

def main():
    """主函数演示"""
    input_image = os.path.join("test", "20250910163759_0001_V.jpeg")
    
    if not os.path.exists(input_image):
        print(f"❌ 找不到文件: {input_image}")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"mapping_demo_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("🚀 开始全景图转换与映射演示...")
        
        # 读取原始全景图
        original_panorama = cv2.imread(input_image)
        print(f"📖 原图尺寸: {original_panorama.shape}")
        
        # 创建转换器
        converter = PanoramaCubemapConverter()
        
        # 1. 全景图 → 立方体贴图
        cube_size = 1024
        faces = converter.panorama_to_cubemap(original_panorama, cube_size)
        
        # 保存分割后的面
        for face_name, face_img in faces.items():
            face_path = os.path.join(output_dir, f'{face_name}.jpg')
            cv2.imwrite(face_path, face_img)
        
        # 2. 立方体贴图 → 全景图（重建）
        panorama_width, panorama_height = original_panorama.shape[1], original_panorama.shape[0]
        reconstructed_panorama = converter.cubemap_to_panorama(faces, panorama_width, panorama_height)
        
        # 保存重建的全景图
        reconstructed_path = os.path.join(output_dir, 'reconstructed_panorama.jpg')
        cv2.imwrite(reconstructed_path, reconstructed_panorama)
        
        # 3. 质量分析
        quality_report = analyze_conversion_quality(original_panorama, reconstructed_panorama)
        
        # 4. 检测框映射演示
        mapper = DetectionMapper(converter)
        faces_with_boxes, panorama_mapped = mapper.create_detection_demo(
            faces, cube_size, panorama_width, panorama_height
        )
        
        # 保存映射结果
        mapper.save_mapping_results(faces_with_boxes, panorama_mapped, output_dir)
        
        # 保存质量报告
        report_path = os.path.join(output_dir, 'quality_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✨ 演示完成！")
        print(f"📁 结果目录: {output_dir}")
        print(f"📊 质量评估: {quality_report['quality_assessment']}")
        print(f"📈 PSNR: {quality_report['psnr']:.2f} dB")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 