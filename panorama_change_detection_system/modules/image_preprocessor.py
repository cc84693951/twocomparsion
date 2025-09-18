#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理模块
对立方体面图像进行去噪、直方图均衡化、光照归一化等预处理，支持CUDA加速
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cuda_utils import CUDAUtils, ensure_cuda_available, cp, CUDA_AVAILABLE
from config import ImagePreprocessorConfig, CUDAConfig


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: ImagePreprocessorConfig = None,
                 cuda_config: CUDAConfig = None):
        self.config = config or ImagePreprocessorConfig()
        self.cuda_config = cuda_config or CUDAConfig()
        
        # 初始化CUDA工具
        self.cuda_utils = CUDAUtils(
            use_cuda=self.cuda_config.use_cuda,
            device_id=self.cuda_config.device_id
        )
        
        logging.info(f"图像预处理器初始化完成，CUDA: {'启用' if self.cuda_utils.use_cuda else '禁用'}")
    
    def preprocess_cube_faces(self, cube_faces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        批量预处理立方体面图像
        
        Args:
            cube_faces: 立方体面图像字典
            
        Returns:
            预处理后的立方体面图像字典
        """
        logging.info(f"开始批量预处理 {len(cube_faces)} 个立方体面")
        
        preprocessed_faces = {}
        
        for face_name, face_image in cube_faces.items():
            logging.debug(f"预处理立方体面: {face_name}")
            preprocessed_faces[face_name] = self.preprocess_single_image(face_image)
        
        logging.info("立方体面预处理完成")
        return preprocessed_faces
    
    def preprocess_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理单张图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 复制图像
        processed_image = image.copy()
        
        # 1. 图像去噪
        processed_image = self.denoise_image(processed_image)
        
        # 2. 光照归一化
        if self.config.enable_lighting_normalization:
            processed_image = self.normalize_lighting(processed_image)
        
        # 3. 对比度增强（CLAHE）
        if self.config.enable_clahe:
            processed_image = self.apply_clahe(processed_image)
        
        return processed_image
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        if self.config.denoise_method == "bilateral":
            return self.bilateral_filter(image)
        elif self.config.denoise_method == "gaussian":
            return self.gaussian_filter(image)
        elif self.config.denoise_method == "median":
            return self.median_filter(image)
        else:
            return image
    
    def bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """双边滤波去噪"""
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.bilateral_filter_cuda(
                image,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )
        else:
            return cv2.bilateralFilter(
                image,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )
    
    def gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """高斯滤波去噪"""
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.gaussian_blur_cuda(
                image,
                self.config.gaussian_kernel_size
            )
        else:
            return cv2.GaussianBlur(
                image,
                (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size),
                0
            )
    
    def median_filter(self, image: np.ndarray) -> np.ndarray:
        """中值滤波去噪"""
        # 中值滤波目前使用CPU版本，因为OpenCV实现已经很高效
        return cv2.medianBlur(image, self.config.median_kernel_size)
    
    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        光照归一化
        
        Args:
            image: 输入图像
            
        Returns:
            光照归一化后的图像
        """
        # 分析光照特性
        lighting_analysis = self.analyze_image_lighting(image)
        
        # 自适应归一化
        normalized_image, _ = self.adaptive_normalize_image(
            image, 
            lighting_analysis,
            conservative_mode=self.config.conservative_mode
        )
        
        return normalized_image
    
    def analyze_image_lighting(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像光照特性
        
        Args:
            image: 输入图像
            
        Returns:
            光照分析结果
        """
        if self.cuda_utils.use_cuda:
            return self._analyze_image_lighting_cuda(image)
        else:
            return self._analyze_image_lighting_cpu(image)
    
    def _analyze_image_lighting_cpu(self, image: np.ndarray) -> Dict[str, Any]:
        """CPU版本的光照分析"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算光照统计信息
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_brightness = np.min(gray)
        max_brightness = np.max(gray)
        
        # 计算动态范围
        dynamic_range = max_brightness - min_brightness
        
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 判断光照类型
        if mean_brightness < 80:
            lighting_type = "dark"
        elif mean_brightness > 180:
            lighting_type = "bright"
        else:
            lighting_type = "normal"
        
        # 判断对比度
        if std_brightness < 30:
            contrast_type = "low"
        elif std_brightness > 80:
            contrast_type = "high"
        else:
            contrast_type = "normal"
        
        return {
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'min_brightness': int(min_brightness),
            'max_brightness': int(max_brightness),
            'dynamic_range': int(dynamic_range),
            'lighting_type': lighting_type,
            'contrast_type': contrast_type,
            'histogram': hist
        }
    
    @ensure_cuda_available
    def _analyze_image_lighting_cuda(self, image: np.ndarray) -> Dict[str, Any]:
        """CUDA版本的光照分析"""
        img_gpu = self.cuda_utils.to_gpu(image)
        
        # 转换为灰度图
        if len(img_gpu.shape) == 3:
            # 简单的BGR到灰度转换
            gray_gpu = 0.299 * img_gpu[:, :, 2] + 0.587 * img_gpu[:, :, 1] + 0.114 * img_gpu[:, :, 0]
        else:
            gray_gpu = img_gpu
        
        # 计算统计信息
        mean_brightness = float(cp.mean(gray_gpu))
        std_brightness = float(cp.std(gray_gpu))
        min_brightness = int(cp.min(gray_gpu))
        max_brightness = int(cp.max(gray_gpu))
        dynamic_range = max_brightness - min_brightness
        
        # 计算直方图
        hist_gpu, _ = cp.histogram(gray_gpu, bins=256, range=(0, 256))
        hist = self.cuda_utils.to_cpu(hist_gpu)
        
        # 判断光照和对比度类型
        if mean_brightness < 80:
            lighting_type = "dark"
        elif mean_brightness > 180:
            lighting_type = "bright"
        else:
            lighting_type = "normal"
        
        if std_brightness < 30:
            contrast_type = "low"
        elif std_brightness > 80:
            contrast_type = "high"
        else:
            contrast_type = "normal"
        
        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'min_brightness': min_brightness,
            'max_brightness': max_brightness,
            'dynamic_range': dynamic_range,
            'lighting_type': lighting_type,
            'contrast_type': contrast_type,
            'histogram': hist
        }
    
    def adaptive_normalize_image(self, image: np.ndarray, 
                               lighting_analysis: Dict[str, Any] = None,
                               conservative_mode: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        自适应图像归一化
        
        Args:
            image: 输入图像
            lighting_analysis: 光照分析结果，如果为None则自动分析
            conservative_mode: 保守模式，减少过度处理
            
        Returns:
            (归一化后的图像, 光照分析结果)
        """
        if lighting_analysis is None:
            lighting_analysis = self.analyze_image_lighting(image)
        
        logging.debug(f"光照类型: {lighting_analysis['lighting_type']}, "
                     f"对比度: {lighting_analysis['contrast_type']}, "
                     f"平均亮度: {lighting_analysis['mean_brightness']:.1f}")
        
        if self.cuda_utils.use_cuda:
            normalized = self._adaptive_normalize_cuda(image, lighting_analysis, conservative_mode)
        else:
            normalized = self._adaptive_normalize_cpu(image, lighting_analysis, conservative_mode)
        
        return normalized, lighting_analysis
    
    def _adaptive_normalize_cpu(self, image: np.ndarray, 
                              lighting_analysis: Dict[str, Any],
                              conservative_mode: bool) -> np.ndarray:
        """CPU版本的自适应归一化"""
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 根据光照类型调整CLAHE参数
        if lighting_analysis['lighting_type'] == 'dark':
            clip_limit = 3.0 if conservative_mode else 4.0
            tile_grid_size = (8, 8) if conservative_mode else (6, 6)
        elif lighting_analysis['lighting_type'] == 'bright':
            clip_limit = 1.5 if conservative_mode else 2.0
            tile_grid_size = (12, 12) if conservative_mode else (10, 10)
        else:
            clip_limit = 2.5 if conservative_mode else 3.0
            tile_grid_size = (8, 8)
        
        # 根据对比度类型进一步调整
        if lighting_analysis['contrast_type'] == 'low':
            clip_limit += 0.5 if conservative_mode else 1.0
        elif lighting_analysis['contrast_type'] == 'high':
            clip_limit -= 0.3 if conservative_mode else 0.5
        
        # 应用自适应CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        
        # 合并通道
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # 转回BGR颜色空间
        normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # 保守模式下减少全局直方图均衡化的使用
        if (not conservative_mode and 
            (lighting_analysis['contrast_type'] == 'low' or 
             lighting_analysis['dynamic_range'] < 100)):
            yuv = cv2.cvtColor(normalized, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return normalized
    
    @ensure_cuda_available
    def _adaptive_normalize_cuda(self, image: np.ndarray,
                               lighting_analysis: Dict[str, Any],
                               conservative_mode: bool) -> np.ndarray:
        """CUDA版本的自适应归一化"""
        # 使用CUDA工具进行CLAHE处理
        if lighting_analysis['lighting_type'] == 'dark':
            clip_limit = 3.0 if conservative_mode else 4.0
            grid_size = (8, 8) if conservative_mode else (6, 6)
        elif lighting_analysis['lighting_type'] == 'bright':
            clip_limit = 1.5 if conservative_mode else 2.0
            grid_size = (12, 12) if conservative_mode else (10, 10)
        else:
            clip_limit = 2.5 if conservative_mode else 3.0
            grid_size = (8, 8)
        
        # 根据对比度调整
        if lighting_analysis['contrast_type'] == 'low':
            clip_limit += 0.5 if conservative_mode else 1.0
        elif lighting_analysis['contrast_type'] == 'high':
            clip_limit -= 0.3 if conservative_mode else 0.5
        
        # 应用CUDA CLAHE
        normalized = self.cuda_utils.clahe_cuda(image, clip_limit, grid_size)
        
        return normalized
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        应用对比度限制自适应直方图均衡化
        
        Args:
            image: 输入图像
            
        Returns:
            CLAHE处理后的图像
        """
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.clahe_cuda(
                image,
                self.config.clahe_clip_limit,
                self.config.clahe_grid_size
            )
        else:
            return self._apply_clahe_cpu(image)
    
    def _apply_clahe_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU版本的CLAHE"""
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        
        if len(image.shape) == 3:
            # 彩色图像：在LAB空间的L通道应用CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像：直接应用CLAHE
            return clahe.apply(image)
    
    def preprocess_two_face_sets(self, faces1: Dict[str, np.ndarray], 
                                faces2: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], 
                                                                       Dict[str, np.ndarray]]:
        """
        同时预处理两组立方体面
        
        Args:
            faces1: 第一组立方体面
            faces2: 第二组立方体面
            
        Returns:
            两组预处理后的立方体面
        """
        logging.info("=" * 50)
        logging.info("图像预处理详细参数:")
        logging.info(f"  去噪方法: {self.config.denoise_method}")
        logging.info(f"  高斯核大小: {self.config.gaussian_kernel_size}")
        logging.info(f"  双边滤波参数: d={self.config.bilateral_d}, sigma_color={self.config.bilateral_sigma_color}, sigma_space={self.config.bilateral_sigma_space}")
        logging.info(f"  中值滤波核: {self.config.median_kernel_size}")
        logging.info(f"  CLAHE启用: {self.config.enable_clahe}")
        logging.info(f"  CLAHE限制: {self.config.clahe_clip_limit}")
        logging.info(f"  CLAHE网格: {self.config.clahe_grid_size}")
        logging.info(f"  光照归一化: {self.config.enable_lighting_normalization}")
        logging.info(f"  CUDA使用: {self.cuda_utils.use_cuda}")
        
        processed_faces1 = self.preprocess_cube_faces(faces1)
        processed_faces2 = self.preprocess_cube_faces(faces2)
        
        logging.info(f"预处理完成: 第一组 {len(processed_faces1)} 个面, 第二组 {len(processed_faces2)} 个面")
        logging.info("=" * 50)
        
        return processed_faces1, processed_faces2
    
    def compare_preprocessing_effects(self, original_image: np.ndarray, 
                                    processed_image: np.ndarray) -> Dict[str, Any]:
        """
        比较预处理前后的效果
        
        Args:
            original_image: 原始图像
            processed_image: 处理后图像
            
        Returns:
            比较结果
        """
        # 分析原始图像和处理后图像的光照特性
        original_analysis = self.analyze_image_lighting(original_image)
        processed_analysis = self.analyze_image_lighting(processed_image)
        
        # 计算改进指标
        brightness_improvement = (processed_analysis['mean_brightness'] - 
                                original_analysis['mean_brightness'])
        contrast_improvement = (processed_analysis['std_brightness'] - 
                              original_analysis['std_brightness'])
        dynamic_range_improvement = (processed_analysis['dynamic_range'] - 
                                   original_analysis['dynamic_range'])
        
        return {
            'original_analysis': original_analysis,
            'processed_analysis': processed_analysis,
            'improvements': {
                'brightness': float(brightness_improvement),
                'contrast': float(contrast_improvement),
                'dynamic_range': int(dynamic_range_improvement)
            },
            'quality_score': self._calculate_quality_score(processed_analysis)
        }
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """
        计算图像质量评分
        
        Args:
            analysis: 图像分析结果
            
        Returns:
            质量评分（0-100）
        """
        score = 0.0
        
        # 亮度评分（理想亮度范围：100-150）
        brightness = analysis['mean_brightness']
        if 100 <= brightness <= 150:
            brightness_score = 100
        else:
            brightness_score = max(0, 100 - abs(brightness - 125) * 0.8)
        score += brightness_score * 0.3
        
        # 对比度评分（理想标准差：40-70）
        contrast = analysis['std_brightness']
        if 40 <= contrast <= 70:
            contrast_score = 100
        else:
            contrast_score = max(0, 100 - abs(contrast - 55) * 1.5)
        score += contrast_score * 0.3
        
        # 动态范围评分（理想范围：150-255）
        dynamic_range = analysis['dynamic_range']
        if dynamic_range >= 150:
            range_score = 100
        else:
            range_score = max(0, dynamic_range / 150 * 100)
        score += range_score * 0.4
        
        return min(100.0, max(0.0, score))
    
    def get_preprocessing_parameters(self) -> Dict[str, Any]:
        """获取当前预处理参数"""
        return {
            'denoise_method': self.config.denoise_method,
            'bilateral_params': {
                'd': self.config.bilateral_d,
                'sigma_color': self.config.bilateral_sigma_color,
                'sigma_space': self.config.bilateral_sigma_space
            },
            'gaussian_kernel_size': self.config.gaussian_kernel_size,
            'median_kernel_size': self.config.median_kernel_size,
            'clahe_enabled': self.config.enable_clahe,
            'clahe_params': {
                'clip_limit': self.config.clahe_clip_limit,
                'grid_size': self.config.clahe_grid_size
            },
            'lighting_normalization': self.config.enable_lighting_normalization,
            'conservative_mode': self.config.conservative_mode,
            'cuda_enabled': self.cuda_utils.use_cuda
        }
    
    def cleanup_memory(self):
        """清理内存"""
        if self.cuda_utils.use_cuda:
            self.cuda_utils.cleanup_memory()
