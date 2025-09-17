#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图处理器 - 合并版本 (CUDA加速)
功能：
1. 全景图分割为立方体贴图
2. 立方体贴图重建全景图  
3. 随机生成检测框并映射坐标
4. 输出处理结果图片
5. CUDA GPU加速支持

CUDA优化说明：
- 全景图到立方体贴图转换：GPU并行处理所有像素坐标转换和双线性插值
- 立方体贴图到全景图重建：批量处理面映射和向量化坐标计算
- 检测框坐标映射：批量处理多个检测框的坐标转换
- 双线性插值：GPU加速的高精度插值计算
- 自动回退：如果CuPy不可用或GPU初始化失败，自动使用CPU版本


依赖要求：
- CuPy (pip install cupy-cuda11x 或 cupy-cuda12x，根据CUDA版本选择)
- NVIDIA GPU with CUDA support
- 足够的GPU内存 (建议4GB以上)
"""

import cv2
import numpy as np
import os
import json
import math
import random
from datetime import datetime
from tqdm import tqdm

# CUDA支持
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # 创建CuPy的替代实现
    class cp:
        @staticmethod
        def array(x):
            return np.array(x)
        @staticmethod
        def asnumpy(x):
            return np.array(x)
        @staticmethod
        def asarray(x):
            return np.array(x)
        @staticmethod
        def zeros_like(x):
            return np.zeros_like(x)
        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)
        @staticmethod
        def ones_like(x):
            return np.ones_like(x)
        @staticmethod
        def sqrt(x):
            return np.sqrt(x)
        @staticmethod
        def sin(x):
            return np.sin(x)
        @staticmethod
        def cos(x):
            return np.cos(x)
        @staticmethod
        def arctan2(x, y):
            return np.arctan2(x, y)
        @staticmethod
        def arccos(x):
            return np.arccos(x)
        @staticmethod
        def meshgrid(*args, **kwargs):
            return np.meshgrid(*args, **kwargs)
        @staticmethod
        def arange(*args, **kwargs):
            return np.arange(*args, **kwargs)
        @staticmethod
        def stack(*args, **kwargs):
            return np.stack(*args, **kwargs)
        @staticmethod
        def abs(x):
            return np.abs(x)
        @staticmethod
        def any(x):
            return np.any(x)
        @staticmethod
        def floor(x):
            return np.floor(x)
        @staticmethod
        def clip(x, a_min, a_max):
            return np.clip(x, a_min, a_max)
        @staticmethod
        def where(condition, x, y):
            return np.where(condition, x, y)
        pi = np.pi
        uint8 = np.uint8
        float32 = np.float32
        int32 = np.int32

class PanoramaProcessor:
    def __init__(self, use_cuda=True):
        self.face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        self.face_descriptions = {
            'front': '前面', 'right': '右面', 'back': '后面',
            'left': '左面', 'top': '上面', 'bottom': '下面'
        }
        
        # CUDA设置
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        if self.use_cuda:
            try:
                cp.cuda.Device(0).use()  # 使用第一个GPU
            except Exception:
                self.use_cuda = False
    
    def process_panorama(self, input_path, output_dir=None, cube_size=1024, 
                        num_random_boxes=8, min_box_size=50, max_box_size=200):
        """
        完整的全景图处理流程 (CUDA加速)
        """        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"panorama_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # GPU内存管理
        if self.use_cuda:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        
        # 读取原始全景图
        original_panorama = cv2.imread(input_path)
        if original_panorama is None:
            raise ValueError(f"无法读取图像: {input_path}")
        
        # 步骤1: 全景图转换为立方体贴图
        faces = self._panorama_to_cubemap(original_panorama, cube_size)
        
        # 保存原始立方体面
        self._save_cube_faces(faces, output_dir, "original")
        
        # 步骤2: 生成随机检测框
        detections = self._generate_random_detections(
            cube_size, num_random_boxes, min_box_size, max_box_size
        )
        
        # 步骤3: 在立方体面上绘制检测框
        faces_with_boxes = self._draw_detections_on_faces(faces, detections)
        
        # 保存带检测框的立方体面
        self._save_cube_faces(faces_with_boxes, output_dir, "with_boxes")
        
        # 步骤4: 直接在原图上映射检测框（避免重建带来的像素损失）
        panorama_width, panorama_height = original_panorama.shape[1], original_panorama.shape[0]
        panorama_with_mapped_boxes = self._map_detections_to_panorama(
            original_panorama, detections, cube_size, panorama_width, panorama_height
        )
        
        # 步骤5: 可选的重建全景图（用于质量对比）
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
        
        # GPU内存清理
        if self.use_cuda:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        
        return output_dir, detections
    
    def _panorama_to_cubemap(self, panorama_img, cube_size):
        """全景图转换为立方体贴图 (CUDA加速版本)"""
        faces = {}
        height, width = panorama_img.shape[:2]
        
        if self.use_cuda:
            return self._panorama_to_cubemap_cuda(panorama_img, cube_size, height, width)
        else:
            return self._panorama_to_cubemap_cpu(panorama_img, cube_size, height, width)
    
    def _panorama_to_cubemap_cuda(self, panorama_img, cube_size, height, width):
        """CUDA加速的全景图转换"""
        faces = {}
        
        # 将全景图传输到GPU
        panorama_gpu = cp.asarray(panorama_img)
        
        # 生成坐标网格
        row_coords, col_coords = cp.meshgrid(cp.arange(cube_size), cp.arange(cube_size), indexing='ij')
        
        for i, face_name in enumerate(tqdm(self.face_names, desc="CUDA转换立方体面")):
            # 标准化坐标到[-1, 1]
            x = (2.0 * col_coords / cube_size) - 1.0
            y = (2.0 * row_coords / cube_size) - 1.0
            
            # 根据面类型计算3D坐标 (GPU并行计算)
            if i == 0:    # front
                x3d, y3d, z3d = x, -y, cp.ones_like(x)
            elif i == 1:  # right
                x3d, y3d, z3d = cp.ones_like(x), -y, -x
            elif i == 2:  # back
                x3d, y3d, z3d = -x, -y, -cp.ones_like(x)
            elif i == 3:  # left
                x3d, y3d, z3d = -cp.ones_like(x), -y, x
            elif i == 4:  # top
                x3d, y3d, z3d = x, cp.ones_like(x), y
            elif i == 5:  # bottom
                x3d, y3d, z3d = x, -cp.ones_like(x), y
            
            # 转换为球面坐标 (GPU向量化计算)
            r = cp.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
            theta = cp.arctan2(x3d, z3d)
            phi = cp.arccos(y3d / r)
            
            # 转换为全景图坐标
            u = (theta + cp.pi) / (2 * cp.pi) * width
            v = phi / cp.pi * height
            
            # 边界检查
            valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            
            # 使用GPU双线性插值采样
            face_img_gpu = self._cuda_bilinear_sample(panorama_gpu, u, v, valid_mask, cube_size)
            
            # 传输回CPU
            faces[face_name] = cp.asnumpy(face_img_gpu)
        
        return faces
    
    def _panorama_to_cubemap_cpu(self, panorama_img, cube_size, height, width):
        """CPU版本的全景图转换 (原始实现)"""
        faces = {}
        
        for i, face_name in enumerate(tqdm(self.face_names, desc="CPU转换立方体面")):
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
                        xyz = [x, -1.0, y]
                    
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
    
    def _cuda_bilinear_sample(self, img_gpu, u, v, valid_mask, cube_size):
        """CUDA双线性插值采样 - 内存优化版本"""
        # 只在有效像素处进行计算，减少内存使用
        valid_indices = cp.where(valid_mask)
        if len(valid_indices[0]) == 0:
            return cp.zeros((cube_size, cube_size, 3), dtype=cp.uint8)
        
        face_img = cp.zeros((cube_size, cube_size, 3), dtype=cp.uint8)
        
        # 只处理有效区域
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        # 获取整数和小数部分
        u_int = cp.floor(u_valid).astype(cp.int32)
        v_int = cp.floor(v_valid).astype(cp.int32)
        u_frac = u_valid - u_int
        v_frac = v_valid - v_int
        
        # 确保索引在边界内
        height, width = img_gpu.shape[:2]
        u_int = cp.clip(u_int, 0, width - 2)
        v_int = cp.clip(v_int, 0, height - 2)
        
        # 双线性插值 - 只处理有效像素
        for c in range(3):  # RGB通道
            p00 = img_gpu[v_int, u_int, c]
            p01 = img_gpu[v_int, u_int + 1, c]
            p10 = img_gpu[v_int + 1, u_int, c]
            p11 = img_gpu[v_int + 1, u_int + 1, c]
            
            interpolated = (p00 * (1 - u_frac) * (1 - v_frac) +
                          p01 * u_frac * (1 - v_frac) +
                          p10 * (1 - u_frac) * v_frac +
                          p11 * u_frac * v_frac)
            
            face_img[valid_indices[0], valid_indices[1], c] = interpolated
        
        return face_img
    
    def _cubemap_to_panorama_improved(self, faces, output_width, output_height):
        """改进的立方体贴图转换回全景图 - CUDA加速版本"""
        cube_size = faces['front'].shape[0]
        
        if self.use_cuda:
            return self._cubemap_to_panorama_cuda(faces, output_width, output_height, cube_size)
        else:
            return self._cubemap_to_panorama_cpu(faces, output_width, output_height, cube_size)
    
    def _cubemap_to_panorama_cuda(self, faces, output_width, output_height, cube_size):
        """CUDA加速的立方体贴图重建全景图 - 内存优化版本"""
        # 将立方体面传输到GPU
        faces_gpu = {}
        for face_name, face_img in faces.items():
            faces_gpu[face_name] = cp.asarray(face_img)
        
        # 创建输出全景图 (在CPU上创建，避免大内存分配)
        panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 分块处理以节省GPU内存
        chunk_height = min(512, output_height)  # 每次处理512行
        
        for start_row in tqdm(range(0, output_height, chunk_height), desc="CUDA重建全景图"):
            end_row = min(start_row + chunk_height, output_height)
            current_height = end_row - start_row
            
            # 生成当前块的坐标网格
            v_coords, u_coords = cp.meshgrid(
                cp.arange(start_row, end_row), 
                cp.arange(output_width), 
                indexing='ij'
            )
            
            # 全景图坐标转换为球面坐标
            theta = (u_coords / output_width) * 2 * cp.pi - cp.pi
            phi = (v_coords / output_height) * cp.pi
            
            # 球面坐标转换为3D坐标
            x = cp.sin(phi) * cp.sin(theta)
            y = cp.cos(phi)
            z = cp.sin(phi) * cp.cos(theta)
            
            # 创建当前块的输出缓冲区
            panorama_chunk_gpu = cp.zeros((current_height, output_width, 3), dtype=cp.uint8)
            
            # 处理当前块
            panorama_chunk_gpu = self._cuda_xyz_to_panorama_batch(x, y, z, faces_gpu, cube_size, panorama_chunk_gpu)
            
            # 将结果复制到CPU
            panorama[start_row:end_row, :, :] = cp.asnumpy(panorama_chunk_gpu)
            
            # 清理GPU内存
            del panorama_chunk_gpu, v_coords, u_coords, theta, phi, x, y, z
            cp.get_default_memory_pool().free_all_blocks()
        
        # 清理GPU立方体面
        for face_name in list(faces_gpu.keys()):
            del faces_gpu[face_name]
        cp.get_default_memory_pool().free_all_blocks()
        
        return panorama
    
    def _cubemap_to_panorama_cpu(self, faces, output_width, output_height, cube_size):
        """CPU版本的立方体贴图重建全景图 (原始实现)"""
        panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        for v in tqdm(range(output_height), desc="CPU重建全景图"):
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
    
    def _cuda_xyz_to_panorama_batch(self, x, y, z, faces_gpu, cube_size, panorama_gpu):
        """CUDA批量处理3D坐标到全景图的映射"""
        abs_x, abs_y, abs_z = cp.abs(x), cp.abs(y), cp.abs(z)
        
        # 为每个立方体面创建掩码
        face_masks = {}
        face_coords = {}
        
        # Front face
        front_mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
        if cp.any(front_mask):
            face_u = (x / z + 1) * 0.5 * cube_size
            face_v = (-y / z + 1) * 0.5 * cube_size
            face_masks['front'] = front_mask
            face_coords['front'] = (face_u, face_v)
        
        # Back face
        back_mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z <= 0)
        if cp.any(back_mask):
            face_u = (-x / (-z) + 1) * 0.5 * cube_size
            face_v = (-y / (-z) + 1) * 0.5 * cube_size
            face_masks['back'] = back_mask
            face_coords['back'] = (face_u, face_v)
        
        # Right face
        right_mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
        if cp.any(right_mask):
            face_u = (-z / x + 1) * 0.5 * cube_size
            face_v = (-y / x + 1) * 0.5 * cube_size
            face_masks['right'] = right_mask
            face_coords['right'] = (face_u, face_v)
        
        # Left face
        left_mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x <= 0)
        if cp.any(left_mask):
            face_u = (z / (-x) + 1) * 0.5 * cube_size
            face_v = (-y / (-x) + 1) * 0.5 * cube_size
            face_masks['left'] = left_mask
            face_coords['left'] = (face_u, face_v)
        
        # Top face
        top_mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
        if cp.any(top_mask):
            face_u = (x / y + 1) * 0.5 * cube_size
            face_v = (z / y + 1) * 0.5 * cube_size
            face_masks['top'] = top_mask
            face_coords['top'] = (face_u, face_v)
        
        # Bottom face
        bottom_mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)
        if cp.any(bottom_mask):
            face_u = (x / (-y) + 1) * 0.5 * cube_size
            face_v = (z / (-y) + 1) * 0.5 * cube_size
            face_masks['bottom'] = bottom_mask
            face_coords['bottom'] = (face_u, face_v)
        
        # 为每个面执行双线性插值
        for face_name in face_masks:
            if face_name in faces_gpu:
                mask = face_masks[face_name]
                face_u, face_v = face_coords[face_name]
                
                # 边界检查
                valid_coords = (face_u >= 0) & (face_u < cube_size) & (face_v >= 0) & (face_v < cube_size)
                combined_mask = mask & valid_coords
                
                if cp.any(combined_mask):
                    # GPU双线性插值
                    sampled_pixels = self._cuda_face_bilinear_sample(faces_gpu[face_name], face_u, face_v, combined_mask)
                    panorama_gpu[combined_mask] = sampled_pixels[combined_mask]
        
        return panorama_gpu
    
    def _cuda_face_bilinear_sample(self, face_img_gpu, u, v, mask):
        """CUDA立方体面双线性插值采样"""
        height, width = face_img_gpu.shape[:2]
        result = cp.zeros(u.shape + (3,), dtype=cp.uint8)
        
        # 获取整数和小数部分
        u_int = cp.floor(u).astype(cp.int32)
        v_int = cp.floor(v).astype(cp.int32)
        u_frac = u - u_int
        v_frac = v - v_int
        
        # 确保索引在边界内
        u_int = cp.clip(u_int, 0, width - 2)
        v_int = cp.clip(v_int, 0, height - 2)
        
        # 双线性插值
        for c in range(3):  # RGB通道
            p00 = face_img_gpu[v_int, u_int, c]
            p01 = face_img_gpu[v_int, u_int + 1, c]
            p10 = face_img_gpu[v_int + 1, u_int, c]
            p11 = face_img_gpu[v_int + 1, u_int + 1, c]
            
            interpolated = (p00 * (1 - u_frac) * (1 - v_frac) +
                          p01 * u_frac * (1 - v_frac) +
                          p10 * (1 - u_frac) * v_frac +
                          p11 * u_frac * v_frac)
            
            result[:, :, c] = interpolated
        
        return result
    
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
                face_v = (z / (-y) + 1) * 0.5 * cube_size
        
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
        """改进的双线性插值 - CUDA加速版本"""
        if self.use_cuda and hasattr(img, 'shape') and len(img.shape) >= 2:
            # 如果是numpy数组且使用CUDA，转换到GPU
            if isinstance(img, np.ndarray):
                img_gpu = cp.asarray(img)
                result_gpu = self._cuda_bilinear_interpolate_single(img_gpu, x, y)
                return cp.asnumpy(result_gpu)
            else:
                # 已经在GPU上
                return cp.asnumpy(self._cuda_bilinear_interpolate_single(img, x, y))
        else:
            # CPU版本
            return self._cpu_bilinear_interpolate_improved(img, x, y)
    
    def _cuda_bilinear_interpolate_single(self, img_gpu, x, y):
        """CUDA单点双线性插值"""
        h, w = img_gpu.shape[:2]
        
        # 确保坐标在有效范围内
        x = max(0.0, min(float(x), w - 1.0))
        y = max(0.0, min(float(y), h - 1.0))
        
        # 获取整数部分和小数部分
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        # 计算权重
        dx = x - x1
        dy = y - y1
        
        # GPU双线性插值计算
        if img_gpu.dtype == cp.uint8:
            img_float = img_gpu.astype(cp.float32)
            
            pixel = (1.0 - dx) * (1.0 - dy) * img_float[y1, x1] + \
                    dx * (1.0 - dy) * img_float[y1, x2] + \
                    (1.0 - dx) * dy * img_float[y2, x1] + \
                    dx * dy * img_float[y2, x2]
            
            pixel = cp.clip(pixel, 0, 255)
            return pixel.astype(cp.uint8)
        else:
            pixel = (1.0 - dx) * (1.0 - dy) * img_gpu[y1, x1] + \
                    dx * (1.0 - dy) * img_gpu[y1, x2] + \
                    (1.0 - dx) * dy * img_gpu[y2, x1] + \
                    dx * dy * img_gpu[y2, x2]
            
            return pixel.astype(img_gpu.dtype)
    
    def _cpu_bilinear_interpolate_improved(self, img, x, y):
        """CPU版本的改进双线性插值"""
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
        """将检测框映射到全景图 - CUDA加速版本"""
        panorama_mapped = panorama.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        if self.use_cuda:
            # CUDA批量处理所有检测框
            all_mapped_rects = self._cuda_batch_map_detections(
                detections, cube_size, panorama_width, panorama_height
            )
        else:
            # CPU逐个处理
            all_mapped_rects = {}
            for face_name, bboxes in detections.items():
                all_mapped_rects[face_name] = []
                for bbox in bboxes:
                    mapped_rects = self._map_bbox_to_panorama_accurate(
                        bbox, face_name, cube_size, panorama_width, panorama_height
                    )
                    all_mapped_rects[face_name].append(mapped_rects)
        
        # 绘制所有映射的检测框 - 增强的融合效果
        overlay = panorama_mapped.copy()
        
        for face_name, face_rects in all_mapped_rects.items():
            for i, mapped_rects in enumerate(face_rects):
                color = colors[i % len(colors)]
                
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
    
    def _cuda_batch_map_detections(self, detections, cube_size, panorama_width, panorama_height):
        """CUDA批量处理检测框映射"""
        all_mapped_rects = {}
        
        for face_name, bboxes in detections.items():
            face_mapped_rects = []
            
            if len(bboxes) > 0:
                # 将所有检测框转换为GPU数组
                bboxes_gpu = cp.array(bboxes)
                
                # 批量计算四个角点
                corners_batch = self._cuda_batch_face_to_panorama_corners(
                    bboxes_gpu, face_name, cube_size, panorama_width, panorama_height
                )
                
                # 将结果转换回CPU并处理边界情况
                corners_cpu = cp.asnumpy(corners_batch)
                
                for i, corners in enumerate(corners_cpu):
                    # 处理边界跨越
                    mapped_rects = self._handle_cuda_boundary_crossing(
                        corners, panorama_width, panorama_height
                    )
                    face_mapped_rects.append(mapped_rects)
            
            all_mapped_rects[face_name] = face_mapped_rects
        
        return all_mapped_rects
    
    def _cuda_batch_face_to_panorama_corners(self, bboxes_gpu, face_name, cube_size, panorama_width, panorama_height):
        """CUDA批量计算立方体面坐标到全景图角点"""
        num_boxes = bboxes_gpu.shape[0]
        corners_batch = cp.zeros((num_boxes, 4, 2), dtype=cp.float32)
        
        # 获取面索引
        face_index = self.face_names.index(face_name)
        
        for box_idx in range(num_boxes):
            x1, y1, x2, y2 = bboxes_gpu[box_idx]
            
            # 四个角点坐标
            face_corners = cp.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=cp.float32)
            
            # 批量转换到全景图坐标
            panorama_corners = self._cuda_face_coords_to_panorama_batch(
                face_corners, face_index, cube_size, panorama_width, panorama_height
            )
            
            corners_batch[box_idx] = panorama_corners
        
        return corners_batch
    
    def _cuda_face_coords_to_panorama_batch(self, face_coords, face_index, cube_size, panorama_width, panorama_height):
        """CUDA批量转换立方体面坐标到全景图坐标"""
        # 标准化到[-1, 1]
        x = (2.0 * face_coords[:, 0] / cube_size) - 1.0
        y = (2.0 * face_coords[:, 1] / cube_size) - 1.0
        
        # 根据面类型计算3D坐标 (GPU并行计算)
        if face_index == 0:    # front
            x3d, y3d, z3d = x, -y, cp.ones_like(x)
        elif face_index == 1:  # right
            x3d, y3d, z3d = cp.ones_like(x), -y, -x
        elif face_index == 2:  # back
            x3d, y3d, z3d = -x, -y, -cp.ones_like(x)
        elif face_index == 3:  # left
            x3d, y3d, z3d = -cp.ones_like(x), -y, x
        elif face_index == 4:  # top
            x3d, y3d, z3d = x, cp.ones_like(x), y
        elif face_index == 5:  # bottom
            x3d, y3d, z3d = x, -cp.ones_like(x), y
        
        # 转换为球面坐标 (GPU向量化计算)
        r = cp.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
        theta = cp.arctan2(x3d, z3d)
        phi = cp.arccos(y3d / r)
        
        # 转换为全景图坐标
        u = (theta + cp.pi) / (2 * cp.pi) * panorama_width
        v = phi / cp.pi * panorama_height
        
        return cp.stack([u, v], axis=1)
    
    def _handle_cuda_boundary_crossing(self, corners, panorama_width, panorama_height):
        """处理CUDA计算结果的边界跨越情况"""
        # 过滤有效坐标
        valid_corners = [(u, v) for u, v in corners if not (np.isnan(u) or np.isnan(v))]
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
            xyz = [x, -1.0, y]
        
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
    """主函数 - CUDA加速版本"""
    input_image = os.path.join("test", "20250910163759_0001_V.jpeg")
    
    if not os.path.exists(input_image):
        return
    
    try:
        processor = PanoramaProcessor(use_cuda=True)
        
        cube_size = 1024
        num_random_boxes = 12
        min_box_size = 80
        max_box_size = 250
        
        output_dir, detections = processor.process_panorama(
            input_path=input_image,
            cube_size=cube_size,
            num_random_boxes=num_random_boxes,
            min_box_size=min_box_size,
            max_box_size=max_box_size
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
