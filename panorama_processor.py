#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图处理器 - 合并版本
功能：
1. 全景图分割为立方体贴图
2. 立方体贴图重建全景图  
3. 随机生成检测框并映射坐标
4. 输出处理结果图片
"""

import cv2
import numpy as np
import os
import json
import math
import random
from datetime import datetime
from tqdm import tqdm

class PanoramaProcessor:
    def __init__(self):
        self.face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        self.face_descriptions = {
            'front': '前面', 'right': '右面', 'back': '后面',
            'left': '左面', 'top': '上面', 'bottom': '下面'
        }
    
    def process_panorama(self, input_path, output_dir=None, cube_size=1024, 
                        num_random_boxes=8, min_box_size=50, max_box_size=200):
        """
        完整的全景图处理流程
        """
        print(f"开始处理全景图: {os.path.basename(input_path)}")
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"panorama_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取原始全景图
        original_panorama = cv2.imread(input_path)
        if original_panorama is None:
            raise ValueError(f"无法读取图像: {input_path}")
        
        print(f"原图尺寸: {original_panorama.shape[1]}x{original_panorama.shape[0]}")
        
        # 步骤1: 全景图转换为立方体贴图
        print("步骤1: 全景图 → 立方体贴图")
        faces = self._panorama_to_cubemap(original_panorama, cube_size)
        
        # 保存原始立方体面
        self._save_cube_faces(faces, output_dir, "original")
        
        # 步骤2: 生成随机检测框
        print("步骤2: 生成随机检测框")
        detections = self._generate_random_detections(
            cube_size, num_random_boxes, min_box_size, max_box_size
        )
        
        # 步骤3: 在立方体面上绘制检测框
        print("步骤3: 绘制检测框")
        faces_with_boxes = self._draw_detections_on_faces(faces, detections)
        
        # 保存带检测框的立方体面
        self._save_cube_faces(faces_with_boxes, output_dir, "with_boxes")
        
        # 步骤4: 直接在原图上映射检测框（避免重建带来的像素损失）
        print("步骤4: 映射检测框到原始全景图")
        panorama_width, panorama_height = original_panorama.shape[1], original_panorama.shape[0]
        panorama_with_mapped_boxes = self._map_detections_to_panorama(
            original_panorama, detections, cube_size, panorama_width, panorama_height
        )
        
        # 步骤5: 可选的重建全景图（用于质量对比）
        print("步骤5: 重建全景图（可选，用于对比）")
        reconstructed_panorama = self._cubemap_to_panorama_improved(faces, panorama_width, panorama_height)
        
        # 保存重建的全景图
        reconstructed_path = os.path.join(output_dir, 'reconstructed_panorama.jpg')
        cv2.imwrite(reconstructed_path, reconstructed_panorama)
        
        # 保存映射后的全景图
        mapped_path = os.path.join(output_dir, 'panorama_with_mapped_boxes.jpg')
        cv2.imwrite(mapped_path, panorama_with_mapped_boxes)
        
        # 步骤6: 保存检测信息
        detection_info_path = os.path.join(output_dir, 'detection_info.json')
        with open(detection_info_path, 'w', encoding='utf-8') as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)
        
        # 创建结果汇总
        self._create_summary(output_dir, input_path, cube_size, detections)
        
        print(f"处理完成！结果保存在: {output_dir}")
        return output_dir, detections
    
    def _panorama_to_cubemap(self, panorama_img, cube_size):
        """全景图转换为立方体贴图"""
        faces = {}
        height, width = panorama_img.shape[:2]
        
        for i, face_name in enumerate(tqdm(self.face_names, desc="转换立方体面")):
            face_img = np.zeros((cube_size, cube_size, 3), dtype=np.uint8)
            
            for row in range(cube_size):
                for col in range(cube_size):
                    # 标准化坐标到[-1, 1]
                    x = (2.0 * col / cube_size) - 1.0
                    y = (2.0 * row / cube_size) - 1.0
                    
                    # 根据面类型计算3D坐标
                    if i == 0:    # front
                        xyz = [x, -y, 1.0]
                    elif i == 1:  # right
                        xyz = [1.0, -y, -x]
                    elif i == 2:  # back
                        xyz = [-x, -y, -1.0]
                    elif i == 3:  # left
                        xyz = [-1.0, -y, x]
                    elif i == 4:  # top
                        xyz = [x, 1.0, y]
                    elif i == 5:  # bottom
                        xyz = [x, -1.0, -y]
                    
                    # 转换为球面坐标
                    x3d, y3d, z3d = xyz
                    r = math.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
                    theta = math.atan2(x3d, z3d)
                    phi = math.acos(y3d / r)
                    
                    # 转换为全景图坐标
                    u = (theta + math.pi) / (2 * math.pi) * width
                    v = phi / math.pi * height
                    
                    # 边界检查和像素采样
                    if 0 <= u < width and 0 <= v < height:
                        face_img[row, col] = panorama_img[int(v), int(u)]
            
            faces[face_name] = face_img
        
        return faces
    
    def _cubemap_to_panorama_improved(self, faces, output_width, output_height):
        """改进的立方体贴图转换回全景图 - 提高质量"""
        cube_size = faces['front'].shape[0]
        panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        for v in tqdm(range(output_height), desc="重建全景图"):
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
                    # 使用改进的双线性插值
                    pixel_value = self._bilinear_interpolate_improved(faces[face_name], face_u, face_v)
                    panorama[v, u] = pixel_value
        
        return panorama
    
    def _xyz_to_cube_face(self, x, y, z, cube_size):
        """确定3D坐标属于哪个立方体面"""
        abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
        
        if abs_z >= abs_x and abs_z >= abs_y:
            if z > 0:  # front
                face_name = 'front'
                face_u = (x / z + 1) * 0.5 * cube_size
                face_v = (-y / z + 1) * 0.5 * cube_size
            else:  # back
                face_name = 'back'
                face_u = (-x / (-z) + 1) * 0.5 * cube_size
                face_v = (-y / (-z) + 1) * 0.5 * cube_size
        elif abs_x >= abs_y:
            if x > 0:  # right
                face_name = 'right'
                face_u = (-z / x + 1) * 0.5 * cube_size
                face_v = (-y / x + 1) * 0.5 * cube_size
            else:  # left
                face_name = 'left'
                face_u = (z / (-x) + 1) * 0.5 * cube_size
                face_v = (-y / (-x) + 1) * 0.5 * cube_size
        else:
            if y > 0:  # top
                face_name = 'top'
                face_u = (x / y + 1) * 0.5 * cube_size
                face_v = (z / y + 1) * 0.5 * cube_size
            else:  # bottom
                face_name = 'bottom'
                face_u = (x / (-y) + 1) * 0.5 * cube_size
                face_v = (-z / (-y) + 1) * 0.5 * cube_size
        
        return face_name, face_u, face_v
    
    def _bilinear_interpolate(self, img, x, y):
        """双线性插值"""
        h, w = img.shape[:2]
        
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        x1, y1 = int(math.floor(x)), int(math.floor(y))
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        dx = x - x1
        dy = y - y1
        
        pixel = (1 - dx) * (1 - dy) * img[y1, x1] + \
                dx * (1 - dy) * img[y1, x2] + \
                (1 - dx) * dy * img[y2, x1] + \
                dx * dy * img[y2, x2]
        
        return pixel.astype(np.uint8)
    
    def _bilinear_interpolate_improved(self, img, x, y):
        """改进的双线性插值 - 减少量化误差"""
        h, w = img.shape[:2]
        
        # 确保坐标在有效范围内
        x = max(0.0, min(float(x), w - 1.0))
        y = max(0.0, min(float(y), h - 1.0))
        
        # 获取整数部分和小数部分
        x1, y1 = int(math.floor(x)), int(math.floor(y))
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        # 计算权重
        dx = x - x1
        dy = y - y1
        
        # 进行高精度插值计算
        if img.dtype == np.uint8:
            # 转换为float进行计算，避免整数溢出
            img_float = img.astype(np.float32)
            
            pixel = (1.0 - dx) * (1.0 - dy) * img_float[y1, x1] + \
                    dx * (1.0 - dy) * img_float[y1, x2] + \
                    (1.0 - dx) * dy * img_float[y2, x1] + \
                    dx * dy * img_float[y2, x2]
            
            # 安全地转换回uint8
            pixel = np.clip(pixel, 0, 255)
            return pixel.astype(np.uint8)
        else:
            # 对于其他数据类型直接计算
            pixel = (1.0 - dx) * (1.0 - dy) * img[y1, x1] + \
                    dx * (1.0 - dy) * img[y1, x2] + \
                    (1.0 - dx) * dy * img[y2, x1] + \
                    dx * dy * img[y2, x2]
            
            return pixel.astype(img.dtype)
    
    def _generate_random_detections(self, cube_size, num_boxes, min_size, max_size):
        """生成随机检测框"""
        detections = {}
        
        # 选择要放置检测框的面（随机选择几个面）
        selected_faces = random.sample(self.face_names, min(len(self.face_names), max(3, num_boxes//2)))
        
        boxes_per_face = num_boxes // len(selected_faces)
        remaining_boxes = num_boxes % len(selected_faces)
        
        for i, face_name in enumerate(selected_faces):
            face_boxes = []
            num_face_boxes = boxes_per_face + (1 if i < remaining_boxes else 0)
            
            for _ in range(num_face_boxes):
                # 随机生成检测框
                box_width = random.randint(min_size, max_size)
                box_height = random.randint(min_size, max_size)
                
                x1 = random.randint(0, cube_size - box_width)
                y1 = random.randint(0, cube_size - box_height)
                x2 = x1 + box_width
                y2 = y1 + box_height
                
                face_boxes.append([x1, y1, x2, y2])
            
            if face_boxes:
                detections[face_name] = face_boxes
        
        return detections
    
    def _draw_detections_on_faces(self, faces, detections):
        """在立方体面上绘制检测框 - 增强的融合效果"""
        faces_with_boxes = {}
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        for face_name in self.face_names:
            face_with_box = faces[face_name].copy()
            overlay = face_with_box.copy()
            
            if face_name in detections:
                for i, bbox in enumerate(detections[face_name]):
                    x1, y1, x2, y2 = bbox
                    color = colors[i % len(colors)]
                    
                    # 绘制半透明填充
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    
                    # 绘制检测框边框 - 增强效果
                    cv2.rectangle(face_with_box, (x1, y1), (x2, y2), color, 3)
                    
                    # 添加内部渐变边缘
                    inner_margin = 8
                    if x2 - x1 > 2 * inner_margin and y2 - y1 > 2 * inner_margin:
                        cv2.rectangle(face_with_box, 
                                    (x1 + inner_margin, y1 + inner_margin), 
                                    (x2 - inner_margin, y2 - inner_margin), 
                                    color, 2)
                    
                    # 添加角点标记
                    corner_size = 10
                    cv2.line(face_with_box, (x1, y1), (x1 + corner_size, y1), color, 4)
                    cv2.line(face_with_box, (x1, y1), (x1, y1 + corner_size), color, 4)
                    cv2.line(face_with_box, (x2, y1), (x2 - corner_size, y1), color, 4)
                    cv2.line(face_with_box, (x2, y1), (x2, y1 + corner_size), color, 4)
                    cv2.line(face_with_box, (x1, y2), (x1 + corner_size, y2), color, 4)
                    cv2.line(face_with_box, (x1, y2), (x1, y2 - corner_size), color, 4)
                    cv2.line(face_with_box, (x2, y2), (x2 - corner_size, y2), color, 4)
                    cv2.line(face_with_box, (x2, y2), (x2, y2 - corner_size), color, 4)
                    
                    # 添加增强标签
                    label = f'{face_name}-{i+1}'
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # 标签背景
                    label_bg_x1 = max(0, x1 - 5)
                    label_bg_y1 = max(0, y1 - text_size[1] - 15)
                    label_bg_x2 = min(face_with_box.shape[1], x1 + text_size[0] + 5)
                    label_bg_y2 = y1 - 5
                    
                    cv2.rectangle(face_with_box, 
                                (label_bg_x1, label_bg_y1), 
                                (label_bg_x2, label_bg_y2), 
                                (0, 0, 0), -1)
                    cv2.rectangle(face_with_box, 
                                (label_bg_x1, label_bg_y1), 
                                (label_bg_x2, label_bg_y2), 
                                color, 2)
                    
                    # 绘制标签文字
                    cv2.putText(face_with_box, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 融合半透明效果
                cv2.addWeighted(face_with_box, 0.8, overlay, 0.2, 0, face_with_box)
            
            faces_with_boxes[face_name] = face_with_box
        
        return faces_with_boxes
    
    def _map_detections_to_panorama(self, panorama, detections, cube_size, 
                                   panorama_width, panorama_height):
        """将检测框映射到全景图 - 增强的融合效果版本"""
        panorama_mapped = panorama.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        # 创建半透明覆盖层
        overlay = panorama_mapped.copy()
        
        for face_name, bboxes in detections.items():
            for i, bbox in enumerate(bboxes):
                color = colors[i % len(colors)]
                
                # 使用更精确的矩形框映射
                mapped_rects = self._map_bbox_to_panorama_accurate(
                    bbox, face_name, cube_size, panorama_width, panorama_height
                )
                
                # 绘制映射的矩形区域 - 添加填充和渐变效果
                for rect_points in mapped_rects:
                    if len(rect_points) >= 4:
                        points_array = np.array(rect_points, dtype=np.int32)
                        
                        # 填充半透明区域
                        cv2.fillPoly(overlay, [points_array], color)
                        
                        # 绘制边框 - 使用更粗的线条
                        cv2.polylines(panorama_mapped, [points_array], True, color, 3)
                        
                        # 添加内部渐变效果
                        mask = np.zeros(panorama_mapped.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [points_array], 255)
                        
                        # 创建渐变效果
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        gradient_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
                        
                        # 应用渐变边缘
                        gradient_color = np.array(color, dtype=np.uint8)
                        for c in range(3):
                            panorama_mapped[:, :, c] = np.where(
                                gradient_mask > 0,
                                np.clip(panorama_mapped[:, :, c].astype(np.int32) + 
                                       (gradient_mask * gradient_color[c] // 255), 0, 255),
                                panorama_mapped[:, :, c]
                            )
                
                # 添加增强的标签
                if mapped_rects and len(mapped_rects[0]) >= 4:
                    center = np.mean(mapped_rects[0], axis=0).astype(int)
                    label = f'{face_name}-{i+1}'
                    
                    # 添加标签背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_bg_pt1 = (center[0] - text_size[0]//2 - 5, center[1] - text_size[1]//2 - 5)
                    text_bg_pt2 = (center[0] + text_size[0]//2 + 5, center[1] + text_size[1]//2 + 5)
                    cv2.rectangle(panorama_mapped, text_bg_pt1, text_bg_pt2, (0, 0, 0), -1)
                    cv2.rectangle(panorama_mapped, text_bg_pt1, text_bg_pt2, color, 2)
                    
                    # 绘制文本
                    cv2.putText(panorama_mapped, label, 
                              (center[0] - text_size[0]//2, center[1] + text_size[1]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 融合半透明填充效果
        cv2.addWeighted(panorama_mapped, 0.7, overlay, 0.3, 0, panorama_mapped)
        
        return panorama_mapped
    
    def _map_bbox_to_panorama_accurate(self, bbox, face_name, cube_size, panorama_width, panorama_height):
        """精确映射检测框到全景图，保持矩形形状并处理边界情况"""
        x1, y1, x2, y2 = bbox
        
        # 映射四个角点
        corners = [
            self._face_coord_to_panorama(x1, y1, face_name, cube_size, panorama_width, panorama_height),
            self._face_coord_to_panorama(x2, y1, face_name, cube_size, panorama_width, panorama_height),
            self._face_coord_to_panorama(x2, y2, face_name, cube_size, panorama_width, panorama_height),
            self._face_coord_to_panorama(x1, y2, face_name, cube_size, panorama_width, panorama_height)
        ]
        
        # 过滤有效坐标
        valid_corners = [c for c in corners if c is not None]
        if len(valid_corners) < 4:
            return []
        
        # 检查是否跨越全景图边界（左右边界连续）
        u_coords = [c[0] for c in valid_corners]
        min_u, max_u = min(u_coords), max(u_coords)
        
        # 如果跨越边界（u坐标差异过大），分割处理
        if max_u - min_u > panorama_width * 0.5:
            return self._handle_boundary_crossing_bbox(valid_corners, panorama_width, panorama_height)
        
        # 不跨越边界，绘制规整的矩形
        return [self._create_regular_rectangle(valid_corners, panorama_width, panorama_height)]
    
    def _handle_boundary_crossing_bbox(self, corners, panorama_width, panorama_height):
        """处理跨越全景图边界的检测框"""
        rectangles = []
        
        # 将坐标调整到同一侧
        adjusted_corners_left = []
        adjusted_corners_right = []
        
        for u, v in corners:
            if u > panorama_width * 0.5:  # 右侧部分
                adjusted_corners_left.append((u - panorama_width, v))
                adjusted_corners_right.append((u, v))
            else:  # 左侧部分
                adjusted_corners_left.append((u, v))
                adjusted_corners_right.append((u + panorama_width, v))
        
        # 创建左侧矩形
        if adjusted_corners_left:
            left_rect = self._create_regular_rectangle(adjusted_corners_left, panorama_width, panorama_height)
            # 将负坐标调整回正值
            left_rect = [(max(0, u), v) for u, v in left_rect]
            rectangles.append(left_rect)
        
        # 创建右侧矩形
        if adjusted_corners_right:
            right_rect = self._create_regular_rectangle(adjusted_corners_right, panorama_width, panorama_height)
            # 将超出边界的坐标调整
            right_rect = [(min(panorama_width-1, u), v) for u, v in right_rect]
            rectangles.append(right_rect)
        
        return rectangles
    
    def _create_regular_rectangle(self, corners, panorama_width, panorama_height):
        """基于角点创建规整矩形"""
        if len(corners) < 4:
            return corners
        
        u_coords = [c[0] for c in corners]
        v_coords = [c[1] for c in corners]
        
        min_u, max_u = min(u_coords), max(u_coords)
        min_v, max_v = min(v_coords), max(v_coords)
        
        # 确保坐标在有效范围内
        min_u = max(0, min_u)
        max_u = min(panorama_width - 1, max_u)
        min_v = max(0, min_v)
        max_v = min(panorama_height - 1, max_v)
        
        # 返回规整的矩形四个角点
        return [(min_u, min_v), (max_u, min_v), (max_u, max_v), (min_u, max_v)]
    
    def _face_coord_to_panorama(self, face_x, face_y, face_name, cube_size, 
                               panorama_width, panorama_height):
        """将立方体面坐标转换为全景图坐标"""
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
    
    def _save_cube_faces(self, faces, output_dir, suffix=""):
        """保存立方体面图像"""
        faces_dir = os.path.join(output_dir, f'cube_faces_{suffix}' if suffix else 'cube_faces')
        os.makedirs(faces_dir, exist_ok=True)
        
        for face_name, face_img in faces.items():
            filename = f'{face_name}.jpg'
            face_path = os.path.join(faces_dir, filename)
            cv2.imwrite(face_path, face_img)
    
    def _create_summary(self, output_dir, input_path, cube_size, detections):
        """创建处理结果汇总"""
        summary = {
            'input_file': os.path.basename(input_path),
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cube_size': cube_size,
            'total_detections': sum(len(boxes) for boxes in detections.values()),
            'detections_by_face': {face: len(boxes) for face, boxes in detections.items()},
            'output_files': {
                'cube_faces_original': 'cube_faces_original/',
                'cube_faces_with_boxes': 'cube_faces_with_boxes/',
                'reconstructed_panorama': 'reconstructed_panorama.jpg',
                'panorama_with_mapped_boxes': 'panorama_with_mapped_boxes.jpg',
                'detection_info': 'detection_info.json'
            }
        }
        
        summary_path = os.path.join(output_dir, 'processing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    input_image = os.path.join("test", "20250910163759_0001_V.jpeg")
    
    if not os.path.exists(input_image):
        print(f"错误: 找不到文件 {input_image}")
        return
    
    try:
        processor = PanoramaProcessor()
        
        # 处理参数
        cube_size = 1024
        num_random_boxes = 12  # 随机生成12个检测框
        min_box_size = 80
        max_box_size = 250
        
        # 执行处理
        output_dir, detections = processor.process_panorama(
            input_path=input_image,
            cube_size=cube_size,
            num_random_boxes=num_random_boxes,
            min_box_size=min_box_size,
            max_box_size=max_box_size
        )
        
        print(f"\n处理完成！")
        print(f"输出目录: {output_dir}")
        print(f"生成的检测框数量: {sum(len(boxes) for boxes in detections.values())}")
        print("主要输出文件:")
        print("  - cube_faces_original/: 原始立方体面")
        print("  - cube_faces_with_boxes/: 带检测框的立方体面")
        print("  - reconstructed_panorama.jpg: 重建的全景图")
        print("  - panorama_with_mapped_boxes.jpg: 映射检测框的全景图")
        print("  - detection_info.json: 检测框信息")
        print("  - processing_summary.json: 处理汇总")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
