#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图立方体分割模块
将全景图分割为立方体六面图，支持CUDA加速
"""

import cv2
import numpy as np
import os
import math
import logging
from typing import Dict, Tuple, Optional, Union, Any
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cuda_utils import CUDAUtils, ensure_cuda_available, cp, CUDA_AVAILABLE
from config import PanoramaSplitterConfig, CUDAConfig


class PanoramaSplitter:
    """全景图立方体分割器"""
    
    def __init__(self, config: PanoramaSplitterConfig = None, 
                 cuda_config: CUDAConfig = None):
        self.config = config or PanoramaSplitterConfig()
        self.cuda_config = cuda_config or CUDAConfig()
        
        # 初始化CUDA工具
        self.cuda_utils = CUDAUtils(
            use_cuda=self.cuda_config.use_cuda,
            device_id=self.cuda_config.device_id
        )
        
        # 立方体面配置
        self.face_names = list(self.config.face_names)
        self.face_descriptions = {
            'front': '前面', 'right': '右面', 'back': '后面',
            'left': '左面', 'top': '上面', 'bottom': '下面'
        }
        
        logging.info(f"全景图分割器初始化完成，CUDA: {'启用' if self.cuda_utils.use_cuda else '禁用'}")
    
    def split_panorama(self, panorama_image: np.ndarray, 
                      cube_size: int = None) -> Dict[str, np.ndarray]:
        """
        将全景图分割为立方体六面图
        
        Args:
            panorama_image: 输入全景图
            cube_size: 立方体面尺寸
            
        Returns:
            包含六个面图像的字典
        """
        if cube_size is None:
            cube_size = self.config.cube_size
        
        logging.info(f"开始全景图分割，输入尺寸: {panorama_image.shape}, 立方体尺寸: {cube_size}")
        
        # 清理GPU内存
        if self.cuda_utils.use_cuda:
            self.cuda_utils.cleanup_memory()
        
        # 执行分割
        if self.cuda_utils.use_cuda:
            faces = self._split_panorama_cuda(panorama_image, cube_size)
        else:
            faces = self._split_panorama_cpu(panorama_image, cube_size)
        
        # 清理GPU内存
        if self.cuda_utils.use_cuda:
            self.cuda_utils.cleanup_memory()
        
        logging.info(f"全景图分割完成，生成 {len(faces)} 个面")
        return faces
    
    @ensure_cuda_available
    def _split_panorama_cuda(self, panorama_img: np.ndarray, 
                            cube_size: int) -> Dict[str, np.ndarray]:
        """CUDA加速的全景图分割"""
        faces = {}
        height, width = panorama_img.shape[:2]
        
        # 将全景图传输到GPU
        panorama_gpu = self.cuda_utils.to_gpu(panorama_img)
        
        # 生成坐标网格
        row_coords, col_coords = cp.meshgrid(
            cp.arange(cube_size), cp.arange(cube_size), indexing='ij'
        )
        
        for i, face_name in enumerate(tqdm(self.face_names, desc="CUDA转换立方体面")):
            # 标准化坐标到[-1, 1]
            x = (2.0 * col_coords / cube_size) - 1.0
            y = (2.0 * row_coords / cube_size) - 1.0
            
            # 根据面类型计算3D坐标
            x3d, y3d, z3d = self._get_face_3d_coords_cuda(i, x, y)
            
            # 转换为球面坐标
            r = cp.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
            theta = cp.arctan2(x3d, z3d)
            phi = cp.arccos(y3d / r)
            
            # 转换为全景图坐标
            u = (theta + cp.pi) / (2 * cp.pi) * width
            v = phi / cp.pi * height
            
            # 边界检查
            valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            
            # 使用GPU双线性插值采样
            face_img_gpu = self._cuda_bilinear_sample(
                panorama_gpu, u, v, valid_mask, cube_size
            )
            
            # 传输回CPU
            faces[face_name] = self.cuda_utils.to_cpu(face_img_gpu)
        
        return faces
    
    def _split_panorama_cpu(self, panorama_img: np.ndarray, 
                           cube_size: int) -> Dict[str, np.ndarray]:
        """CPU版本的全景图分割"""
        faces = {}
        height, width = panorama_img.shape[:2]
        
        for i, face_name in enumerate(tqdm(self.face_names, desc="CPU转换立方体面")):
            face_img = np.zeros((cube_size, cube_size, 3), dtype=np.uint8)
            
            for row in range(cube_size):
                for col in range(cube_size):
                    # 标准化坐标到[-1, 1]
                    x = (2.0 * col / cube_size) - 1.0
                    y = (2.0 * row / cube_size) - 1.0
                    
                    # 根据面类型计算3D坐标
                    x3d, y3d, z3d = self._get_face_3d_coords_cpu(i, x, y)
                    
                    # 转换为球面坐标
                    r = math.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
                    theta = math.atan2(x3d, z3d)
                    phi = math.acos(y3d / r)
                    
                    # 转换为全景图坐标
                    u = (theta + math.pi) / (2 * math.pi) * width
                    v = phi / math.pi * height
                    
                    # 边界检查和像素采样
                    if 0 <= u < width and 0 <= v < height:
                        if self.config.interpolation_method == "bilinear":
                            face_img[row, col] = self._bilinear_interpolate_cpu(
                                panorama_img, u, v
                            )
                        else:
                            face_img[row, col] = panorama_img[int(v), int(u)]
            
            faces[face_name] = face_img
        
        return faces
    
    def _get_face_3d_coords_cuda(self, face_index: int, x, y):
        """获取立方体面的3D坐标（CUDA版本）"""
        if face_index == 0:    # front
            return x, -y, cp.ones_like(x)
        elif face_index == 1:  # right
            return cp.ones_like(x), -y, -x
        elif face_index == 2:  # back
            return -x, -y, -cp.ones_like(x)
        elif face_index == 3:  # left
            return -cp.ones_like(x), -y, x
        elif face_index == 4:  # top
            return x, cp.ones_like(x), y
        elif face_index == 5:  # bottom
            return x, -cp.ones_like(x), y
        else:
            raise ValueError(f"无效的面索引: {face_index}")
    
    def _get_face_3d_coords_cpu(self, face_index: int, x: float, y: float):
        """获取立方体面的3D坐标（CPU版本）"""
        if face_index == 0:    # front
            return x, -y, 1.0
        elif face_index == 1:  # right
            return 1.0, -y, -x
        elif face_index == 2:  # back
            return -x, -y, -1.0
        elif face_index == 3:  # left
            return -1.0, -y, x
        elif face_index == 4:  # top
            return x, 1.0, y
        elif face_index == 5:  # bottom
            return x, -1.0, y
        else:
            raise ValueError(f"无效的面索引: {face_index}")
    
    def _cuda_bilinear_sample(self, img_gpu, u, v, valid_mask, cube_size):
        """CUDA双线性插值采样"""
        # 只在有效像素处进行计算
        valid_indices = cp.where(valid_mask)
        if len(valid_indices[0]) == 0:
            if len(img_gpu.shape) == 3:
                return cp.zeros((cube_size, cube_size, 3), dtype=cp.uint8)
            else:
                return cp.zeros((cube_size, cube_size), dtype=cp.uint8)
        
        if len(img_gpu.shape) == 3:
            face_img = cp.zeros((cube_size, cube_size, 3), dtype=cp.uint8)
            channels = 3
        else:
            face_img = cp.zeros((cube_size, cube_size), dtype=cp.uint8)
            channels = 1
        
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
        
        # 双线性插值
        if channels == 3:
            for c in range(3):
                p00 = img_gpu[v_int, u_int, c]
                p01 = img_gpu[v_int, u_int + 1, c]
                p10 = img_gpu[v_int + 1, u_int, c]
                p11 = img_gpu[v_int + 1, u_int + 1, c]
                
                interpolated = (p00 * (1 - u_frac) * (1 - v_frac) +
                              p01 * u_frac * (1 - v_frac) +
                              p10 * (1 - u_frac) * v_frac +
                              p11 * u_frac * v_frac)
                
                face_img[valid_indices[0], valid_indices[1], c] = interpolated
        else:
            p00 = img_gpu[v_int, u_int]
            p01 = img_gpu[v_int, u_int + 1]
            p10 = img_gpu[v_int + 1, u_int]
            p11 = img_gpu[v_int + 1, u_int + 1]
            
            interpolated = (p00 * (1 - u_frac) * (1 - v_frac) +
                          p01 * u_frac * (1 - v_frac) +
                          p10 * (1 - u_frac) * v_frac +
                          p11 * u_frac * v_frac)
            
            face_img[valid_indices[0], valid_indices[1]] = interpolated
        
        return face_img
    
    def _bilinear_interpolate_cpu(self, img: np.ndarray, x: float, y: float):
        """CPU双线性插值"""
        h, w = img.shape[:2]
        
        x = max(0.0, min(float(x), w - 1.0))
        y = max(0.0, min(float(y), h - 1.0))
        
        x1, y1 = int(math.floor(x)), int(math.floor(y))
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        dx = x - x1
        dy = y - y1
        
        if len(img.shape) == 3:
            pixel = ((1.0 - dx) * (1.0 - dy) * img[y1, x1] +
                    dx * (1.0 - dy) * img[y1, x2] +
                    (1.0 - dx) * dy * img[y2, x1] +
                    dx * dy * img[y2, x2])
        else:
            pixel = ((1.0 - dx) * (1.0 - dy) * img[y1, x1] +
                    dx * (1.0 - dy) * img[y1, x2] +
                    (1.0 - dx) * dy * img[y2, x1] +
                    dx * dy * img[y2, x2])
        
        return pixel.astype(np.uint8)
    
    def split_two_panoramas(self, panorama1: np.ndarray, panorama2: np.ndarray, 
                          cube_size: int = None) -> Tuple[Dict[str, np.ndarray], 
                                                         Dict[str, np.ndarray]]:
        """
        同时分割两张全景图
        
        Args:
            panorama1: 第一张全景图
            panorama2: 第二张全景图
            cube_size: 立方体面尺寸
            
        Returns:
            两个包含立方体面的字典
        """
        logging.info("开始分割两张全景图")
        
        faces1 = self.split_panorama(panorama1, cube_size)
        faces2 = self.split_panorama(panorama2, cube_size)
        
        return faces1, faces2
    
    def save_cube_faces(self, faces: Dict[str, np.ndarray], 
                       output_dir: str, suffix: str = "") -> Dict[str, str]:
        """
        保存立方体面图像
        
        Args:
            faces: 立方体面字典
            output_dir: 输出目录
            suffix: 文件名后缀
            
        Returns:
            保存的文件路径字典
        """
        if suffix:
            faces_dir = os.path.join(output_dir, f'cube_faces_{suffix}')
        else:
            faces_dir = os.path.join(output_dir, 'cube_faces')
        
        os.makedirs(faces_dir, exist_ok=True)
        
        saved_paths = {}
        
        for face_name, face_img in faces.items():
            filename = f'{face_name}.{self.config.output_format}'
            face_path = os.path.join(faces_dir, filename)
            
            success = cv2.imwrite(face_path, face_img)
            if success:
                saved_paths[face_name] = face_path
                logging.debug(f"保存立方体面: {face_path}")
            else:
                logging.error(f"保存立方体面失败: {face_path}")
        
        logging.info(f"立方体面保存完成，目录: {faces_dir}")
        return saved_paths
    
    def get_face_info(self) -> Dict[str, str]:
        """获取立方体面信息"""
        return {
            'face_names': self.face_names,
            'face_descriptions': self.face_descriptions,
            'cube_size': self.config.cube_size,
            'interpolation_method': self.config.interpolation_method,
            'cuda_enabled': self.cuda_utils.use_cuda
        }
    
    def validate_panorama(self, panorama_image: np.ndarray) -> bool:
        """
        验证全景图是否符合要求
        
        Args:
            panorama_image: 全景图
            
        Returns:
            是否有效
        """
        if panorama_image is None:
            logging.error("全景图为空")
            return False
        
        if len(panorama_image.shape) != 3:
            logging.error(f"全景图维度错误: {panorama_image.shape}")
            return False
        
        height, width, channels = panorama_image.shape
        
        if channels != 3:
            logging.error(f"全景图通道数错误: {channels}")
            return False
        
        # 检查长宽比（全景图通常是2:1）
        aspect_ratio = width / height
        if not (1.8 <= aspect_ratio <= 2.2):
            logging.warning(f"全景图长宽比可能不正确: {aspect_ratio:.2f} (期望约为2.0)")
        
        # 检查最小尺寸
        if width < 1000 or height < 500:
            logging.warning(f"全景图尺寸较小: {width}x{height}")
        
        logging.info(f"全景图验证通过: {width}x{height}, 长宽比: {aspect_ratio:.2f}")
        return True
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = self.cuda_utils.get_memory_info()
        return {
            'cuda_memory': memory_info,
            'face_count': len(self.face_names),
            'cube_size': self.config.cube_size,
            'estimated_memory_per_face_mb': (self.config.cube_size ** 2 * 3) / (1024 * 1024)
        }
