#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图像变化检测系统
合并全景图处理和图像变化检测功能
实现完整的全景图像变化检测流程

主要功能：
1. 全景图立方体分割
2. 图像预处理（去噪、直方图均衡化）
3. AKAZE特征点提取和匹配
4. 图像配准和变换
5. 图像差分计算
6. 阈值分割和形态学操作
7. 轮廓提取和过滤
8. 目标检测框生成
9. 结果还原至全景图

作者：AI Assistant
日期：2025年09月18日
"""

import cv2
import numpy as np
import os
import json
import math
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# CUDA支持检测
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("✅ CUDA支持已启用")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA不可用，将使用CPU版本")
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


class PanoramaChangeDetectionSystem:
    """全景图像变化检测系统"""
    
    def __init__(self, output_dir="panorama_change_detection_results", use_cuda=True):
        """
        初始化系统
        
        Args:
            output_dir (str): 输出目录
            use_cuda (bool): 是否使用CUDA加速
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 立方体面名称和描述
        self.face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        self.face_descriptions = {
            'front': '前面', 'right': '右面', 'back': '后面',
            'left': '左面', 'top': '上面', 'bottom': '下面'
        }
        
        # CUDA设置
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        if self.use_cuda:
            try:
                cp.cuda.Device(0).use()
                print("🚀 GPU加速已启用")
                print("  ├─ 立方体分割: CUDA加速")
                print("  ├─ 图像预处理: CUDA加速")
                print("  ├─ 图像差分: CUDA加速")
                print("  ├─ 形态学操作: CUDA加速")
                print("  ├─ 全景图重建: CUDA加速")
                print("  └─ 双线性插值: CUDA加速")
            except Exception:
                self.use_cuda = False
                print("⚠️ GPU初始化失败，使用CPU版本")
        
        # 系统参数配置
        self.config = {
            'cube_size': None,                    # 立方体面尺寸（动态计算）
            'diff_threshold': 30,                 # 差异阈值
            'min_contour_area': 500,              # 最小轮廓面积
            'max_contour_area': 50000,            # 最大轮廓面积
            'min_aspect_ratio': 0.2,              # 最小长宽比
            'max_aspect_ratio': 5.0,              # 最大长宽比
            'morphology_kernel_size': (5, 5),     # 形态学核大小
            'gaussian_blur_kernel': (3, 3),       # 高斯模糊核大小
            'clahe_clip_limit': 2.0,              # CLAHE限制值
            'clahe_tile_grid_size': (8, 8),       # CLAHE网格大小
            'skip_faces': ['top'],                # 跳过的面
        }
        
        print(f"🔧 系统初始化完成")
        print(f"📁 输出目录: {output_dir}")
        print(f"⚙️ 系统配置: {self.config}")
    
    def load_image_with_chinese_path(self, path):
        """
        加载包含中文路径的图像
        
        Args:
            path (str): 图像路径
            
        Returns:
            ndarray: 图像数据
        """
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            return img
        except Exception as e:
            print(f"❌ 读取图像失败 {path}: {str(e)}")
            return None
    
    def panorama_to_cubemap(self, panorama_img, cube_size=None):
        """
        全景图转换为立方体贴图
        
        Args:
            panorama_img (ndarray): 全景图像
            cube_size (int): 立方体面尺寸，如果为None则动态计算
            
        Returns:
            dict: 立方体面字典
        """
        print("🔄 开始全景图立方体分割...")
        height, width = panorama_img.shape[:2]
        
        # 动态计算最佳立方体面尺寸
        if cube_size is None:
            # 基于全景图尺寸计算，保持高分辨率
            cube_size = min(width // 4, height // 2)  # 确保不会太大导致内存问题
            cube_size = max(cube_size, 1024)  # 最小1024
            cube_size = min(cube_size, 4096)  # 最大4096
        
        print(f"📐 使用立方体面尺寸: {cube_size}×{cube_size}")
        print(f"📊 原图尺寸: {width}×{height}")
        
        # 更新配置中的cube_size
        self.config['cube_size'] = cube_size
        
        faces = {}
        
        if self.use_cuda:
            faces = self._panorama_to_cubemap_cuda(panorama_img, cube_size, height, width)
        else:
            faces = self._panorama_to_cubemap_cpu(panorama_img, cube_size, height, width)
        
        print(f"✅ 立方体分割完成，生成 {len(faces)} 个面")
        return faces
    
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
            
            # 根据面类型计算3D坐标
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
            face_img_gpu = self._cuda_bilinear_sample(panorama_gpu, u, v, valid_mask, cube_size)
            
            # 传输回CPU
            faces[face_name] = cp.asnumpy(face_img_gpu)
        
        return faces
    
    def _panorama_to_cubemap_cpu(self, panorama_img, cube_size, height, width):
        """CPU版本的全景图转换"""
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
        """CUDA双线性插值采样"""
        valid_indices = cp.where(valid_mask)
        if len(valid_indices[0]) == 0:
            return cp.zeros((cube_size, cube_size, 3), dtype=cp.uint8)
        
        face_img = cp.zeros((cube_size, cube_size, 3), dtype=cp.uint8)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        u_int = cp.floor(u_valid).astype(cp.int32)
        v_int = cp.floor(v_valid).astype(cp.int32)
        u_frac = u_valid - u_int
        v_frac = v_valid - v_int
        
        height, width = img_gpu.shape[:2]
        u_int = cp.clip(u_int, 0, width - 2)
        v_int = cp.clip(v_int, 0, height - 2)
        
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
        
        return face_img
    
    def preprocess_image(self, img):
        """
        图像预处理：去噪和直方图均衡化
        
        Args:
            img (ndarray): 输入图像
            
        Returns:
            ndarray: 预处理后的图像
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return self._preprocess_image_cuda(img)
        else:
            return self._preprocess_image_cpu(img)
    
    def _preprocess_image_cuda(self, img):
        """CUDA加速的图像预处理"""
        try:
            # 传输到GPU
            img_gpu = cp.asarray(img)
            
            # 1. GPU高斯模糊 (使用CuPy的filter实现)
            kernel_size = self.config['gaussian_blur_kernel'][0]
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # 简化的高斯模糊（使用均值滤波近似）
            from cupyx.scipy import ndimage
            denoised_gpu = ndimage.gaussian_filter(img_gpu.astype(cp.float32), sigma=1.0)
            denoised_gpu = cp.clip(denoised_gpu, 0, 255).astype(cp.uint8)
            
            # 2. 颜色空间转换到LAB（在GPU上）
            # CuPy没有直接的颜色空间转换，回退到CPU处理颜色空间
            denoised = cp.asnumpy(denoised_gpu)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 3. CLAHE处理（CPU，因为OpenCV的CLAHE没有GPU版本）
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'], 
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            l_clahe = clahe.apply(l)
            
            # 4. 合并通道并转回BGR
            lab_clahe = cv2.merge([l_clahe, a, b])
            processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            
            return processed
            
        except Exception as e:
            print(f"⚠️ CUDA预处理失败，回退到CPU: {e}")
            return self._preprocess_image_cpu(img)
    
    def _preprocess_image_cpu(self, img):
        """CPU版本的图像预处理"""
        # 1. 高斯模糊去噪
        denoised = cv2.GaussianBlur(img, self.config['gaussian_blur_kernel'], 0)
        
        # 2. 转换为LAB颜色空间进行CLAHE处理
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 3. 对L通道应用CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'], 
            tileGridSize=self.config['clahe_tile_grid_size']
        )
        l_clahe = clahe.apply(l)
        
        # 4. 合并通道并转回BGR
        lab_clahe = cv2.merge([l_clahe, a, b])
        processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def extract_akaze_features(self, img):
        """
        使用AKAZE算法提取图像特征点
        
        Args:
            img (ndarray): 输入图像
            
        Returns:
            tuple: (关键点, 描述符)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 创建AKAZE检测器
        akaze = cv2.AKAZE_create()
        
        # 检测关键点和计算描述符
        keypoints, descriptors = akaze.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features_and_register(self, img1, img2, kp1, des1, kp2, des2):
        """
        特征匹配和图像配准
        
        Args:
            img1, img2: 输入图像
            kp1, des1: 第一张图像的关键点和描述符
            kp2, des2: 第二张图像的关键点和描述符
            
        Returns:
            tuple: (配准后的图像2, 单应性矩阵, 匹配信息)
        """
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("⚠️ 特征点不足，跳过配准")
            return img2, None, {"matches": 0, "inliers": 0, "inlier_ratio": 0.0}
        
        # 使用BFMatcher进行特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            print("⚠️ 匹配点不足，跳过配准")
            return img2, None, {"matches": len(matches), "inliers": 0, "inlier_ratio": 0.0}
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        homography, mask = cv2.findHomography(
            dst_pts, src_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0
        )
        
        if homography is None:
            print("⚠️ 无法计算单应性矩阵")
            return img2, None, {"matches": len(matches), "inliers": 0, "inlier_ratio": 0.0}
        
        # 计算内点数量
        inliers = np.sum(mask) if mask is not None else 0
        
        # 应用单应性变换
        h, w = img1.shape[:2]
        registered_img2 = cv2.warpPerspective(img2, homography, (w, h))
        
        match_info = {
            "matches": len(matches),
            "inliers": int(inliers),
            "inlier_ratio": float(inliers / len(matches)) if len(matches) > 0 else 0.0,
            "homography": homography.tolist() if homography is not None else None
        }
        
        print(f"✅ 特征匹配完成：{len(matches)} 个匹配点，{inliers} 个内点")
        
        return registered_img2, homography, match_info
    
    def compute_image_difference(self, img1, img2):
        """
        计算图像差分
        
        Args:
            img1, img2: 输入图像
            
        Returns:
            ndarray: 差分图像
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return self._compute_image_difference_cuda(img1, img2)
        else:
            return self._compute_image_difference_cpu(img1, img2)
    
    def _compute_image_difference_cuda(self, img1, img2):
        """CUDA加速的图像差分计算"""
        try:
            # 传输到GPU
            img1_gpu = cp.asarray(img1)
            img2_gpu = cp.asarray(img2)
            
            # 转换为灰度图（在GPU上）
            if len(img1_gpu.shape) == 3:
                # RGB到灰度的权重 (0.299, 0.587, 0.114)
                weights = cp.array([0.299, 0.587, 0.114])
                gray1_gpu = cp.dot(img1_gpu[...,:3], weights).astype(cp.uint8)
            else:
                gray1_gpu = img1_gpu
                
            if len(img2_gpu.shape) == 3:
                weights = cp.array([0.299, 0.587, 0.114])
                gray2_gpu = cp.dot(img2_gpu[...,:3], weights).astype(cp.uint8)
            else:
                gray2_gpu = img2_gpu
            
            # 确保尺寸一致（在GPU上resize）
            if gray1_gpu.shape != gray2_gpu.shape:
                # 简单的最近邻插值resize（GPU版本）
                h1, w1 = gray1_gpu.shape
                h2, w2 = gray2_gpu.shape
                if h1 != h2 or w1 != w2:
                    # 使用GPU的resize功能
                    y_scale = h1 / h2
                    x_scale = w1 / w2
                    y_indices = cp.arange(h1)[:, None] / y_scale
                    x_indices = cp.arange(w1)[None, :] / x_scale
                    y_indices = cp.clip(y_indices, 0, h2-1).astype(cp.int32)
                    x_indices = cp.clip(x_indices, 0, w2-1).astype(cp.int32)
                    gray2_gpu = gray2_gpu[y_indices, x_indices]
            
            # 计算绝对差值（在GPU上）
            diff_gpu = cp.abs(gray1_gpu.astype(cp.int16) - gray2_gpu.astype(cp.int16)).astype(cp.uint8)
            
            # 传输回CPU
            return cp.asnumpy(diff_gpu)
            
        except Exception as e:
            print(f"⚠️ CUDA差分计算失败，回退到CPU: {e}")
            return self._compute_image_difference_cpu(img1, img2)
    
    def _compute_image_difference_cpu(self, img1, img2):
        """CPU版本的图像差分计算"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 确保尺寸一致
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # 计算绝对差值
        diff = cv2.absdiff(gray1, gray2)
        
        return diff
    
    def threshold_and_morphology(self, diff_img):
        """
        阈值分割和形态学操作
        
        Args:
            diff_img: 差分图像
            
        Returns:
            ndarray: 处理后的二值图像
        """
        if self.use_cuda and CUDA_AVAILABLE:
            return self._threshold_and_morphology_cuda(diff_img)
        else:
            return self._threshold_and_morphology_cpu(diff_img)
    
    def _threshold_and_morphology_cuda(self, diff_img):
        """CUDA加速的阈值分割和形态学操作"""
        try:
            # 传输到GPU
            diff_gpu = cp.asarray(diff_img)
            
            # 阈值分割（在GPU上）
            binary_gpu = (diff_gpu > self.config['diff_threshold']).astype(cp.uint8) * 255
            
            # 形态学操作（使用CuPy的ndimage）
            from cupyx.scipy import ndimage
            
            # 创建椭圆形结构元素
            kernel_size = self.config['morphology_kernel_size']
            y, x = cp.ogrid[-kernel_size[0]//2:kernel_size[0]//2+1, -kernel_size[1]//2:kernel_size[1]//2+1]
            kernel = ((x*x)/(kernel_size[1]//2)**2 + (y*y)/(kernel_size[0]//2)**2) <= 1
            kernel = kernel.astype(cp.uint8)
            
            # 闭运算：先膨胀后腐蚀
            dilated = ndimage.binary_dilation(binary_gpu > 0, structure=kernel).astype(cp.uint8) * 255
            closed = ndimage.binary_erosion(dilated > 0, structure=kernel).astype(cp.uint8) * 255
            
            # 开运算：先腐蚀后膨胀
            eroded = ndimage.binary_erosion(closed > 0, structure=kernel).astype(cp.uint8) * 255
            opened = ndimage.binary_dilation(eroded > 0, structure=kernel).astype(cp.uint8) * 255
            
            # 传输回CPU
            return cp.asnumpy(opened)
            
        except Exception as e:
            print(f"⚠️ CUDA形态学操作失败，回退到CPU: {e}")
            return self._threshold_and_morphology_cpu(diff_img)
    
    def _threshold_and_morphology_cpu(self, diff_img):
        """CPU版本的阈值分割和形态学操作"""
        # 阈值分割
        _, binary = cv2.threshold(diff_img, self.config['diff_threshold'], 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config['morphology_kernel_size'])
        
        # 先闭运算连接邻近区域
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 再开运算去除噪声
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def extract_contours_and_bboxes(self, binary_img, original_img):
        """
        轮廓提取和边界框生成
        
        Args:
            binary_img: 二值图像
            original_img: 原始图像（用于可视化）
            
        Returns:
            tuple: (边界框列表, 可视化图像)
        """
        # 查找轮廓
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        vis_img = original_img.copy()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.config['min_contour_area'] or area > self.config['max_contour_area']:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 长宽比过滤
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < self.config['min_aspect_ratio'] or aspect_ratio > self.config['max_aspect_ratio']:
                continue
            
            # 计算置信度（基于面积和形状）
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            confidence = min(1.0, area / 10000.0 * circularity)
            
            bbox_info = {
                'id': i + 1,
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': float(area),
                'aspect_ratio': float(aspect_ratio),
                'circularity': float(circularity),
                'confidence': float(confidence),
                'center': [int(x + w/2), int(y + h/2)]
            }
            
            bboxes.append(bbox_info)
            
            # 在可视化图像上绘制边界框
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (255, 0, 0)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis_img, f"ID:{i+1} ({confidence:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 按置信度排序
        bboxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"✅ 检测到 {len(bboxes)} 个有效变化区域")
        
        return bboxes, vis_img
    
    def map_bboxes_to_panorama(self, face_bboxes, face_name, cube_size, panorama_width, panorama_height):
        """
        将立方体面的边界框映射回全景图
        
        Args:
            face_bboxes: 立方体面的边界框列表
            face_name: 面名称
            cube_size: 立方体面尺寸
            panorama_width: 全景图宽度
            panorama_height: 全景图高度
            
        Returns:
            list: 映射到全景图的边界框列表
        """
        if not face_bboxes:
            return []
        
        panorama_bboxes = []
        face_index = self.face_names.index(face_name)
        
        for bbox_info in face_bboxes:
            x, y, w, h = bbox_info['bbox']
            
            # 计算边界框的四个角点
            corners = [
                [x, y], [x+w, y], [x+w, y+h], [x, y+h]
            ]
            
            # 将每个角点映射到全景图
            panorama_corners = []
            for corner_x, corner_y in corners:
                pano_x, pano_y = self._face_coord_to_panorama(
                    corner_x, corner_y, face_index, cube_size, panorama_width, panorama_height
                )
                panorama_corners.append([pano_x, pano_y])
            
            # 计算全景图中的边界框
            xs = [c[0] for c in panorama_corners]
            ys = [c[1] for c in panorama_corners]
            
            # 处理跨越边界的情况
            if max(xs) - min(xs) > panorama_width * 0.5:
                # 跨越左右边界，分成两个框
                left_xs = [x if x < panorama_width/2 else x - panorama_width for x in xs]
                right_xs = [x if x > panorama_width/2 else x + panorama_width for x in xs]
                
                # 左侧框
                left_x1, left_x2 = max(0, min(left_xs)), min(panorama_width-1, max(left_xs))
                if left_x2 > left_x1:
                    y1, y2 = max(0, min(ys)), min(panorama_height-1, max(ys))
                    panorama_bboxes.append({
                        **bbox_info,
                        'face_name': face_name,
                        'panorama_bbox': [int(left_x1), int(y1), int(left_x2-left_x1), int(y2-y1)],
                        'corners': panorama_corners,
                        'split_part': 'left'
                    })
                
                # 右侧框
                right_x1, right_x2 = max(0, min(right_xs)), min(panorama_width-1, max(right_xs))
                if right_x2 > right_x1:
                    y1, y2 = max(0, min(ys)), min(panorama_height-1, max(ys))
                    panorama_bboxes.append({
                        **bbox_info,
                        'face_name': face_name,
                        'panorama_bbox': [int(right_x1), int(y1), int(right_x2-right_x1), int(y2-y1)],
                        'corners': panorama_corners,
                        'split_part': 'right'
                    })
            else:
                # 正常情况，不跨越边界
                x1, x2 = max(0, min(xs)), min(panorama_width-1, max(xs))
                y1, y2 = max(0, min(ys)), min(panorama_height-1, max(ys))
                
                if x2 > x1 and y2 > y1:
                    panorama_bboxes.append({
                        **bbox_info,
                        'face_name': face_name,
                        'panorama_bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'corners': panorama_corners,
                        'split_part': None
                    })
        
        return panorama_bboxes
    
    def _face_coord_to_panorama(self, face_x, face_y, face_index, cube_size, panorama_width, panorama_height):
        """将立方体面坐标转换为全景图坐标"""
        # 标准化到[-1, 1]
        x = (2.0 * face_x / cube_size) - 1.0
        y = (2.0 * face_y / cube_size) - 1.0
        
        # 根据面类型计算3D坐标
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
        
        return u, v
    
    def reconstruct_panorama_with_detections(self, faces_with_detections, panorama_width, panorama_height):
        """
        重建带有检测结果的全景图
        
        Args:
            faces_with_detections: 带有检测结果的立方体面
            panorama_width: 全景图宽度
            panorama_height: 全景图高度
            
        Returns:
            ndarray: 重建的全景图
        """
        print("🔄 重建带检测结果的全景图...")
        
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # 获取立方体面尺寸
        cube_size = self.config['cube_size']
        if cube_size is None:
            first_face = list(faces_with_detections.values())[0]
            cube_size = first_face.shape[0]
        
        print(f"📐 重建使用立方体面尺寸: {cube_size}")
        
        # 使用更精确的重建算法，确保没有黑色区域
        self._reconstruct_panorama_improved(panorama, faces_with_detections, panorama_width, panorama_height, cube_size)
        
        print("✅ 全景图重建完成")
        return panorama
    
    def _reconstruct_panorama_improved(self, panorama, faces_with_detections, panorama_width, panorama_height, cube_size):
        """改进的全景图重建算法，消除黑色区域"""
        
        print(f"🔍 CUDA状态检查: use_cuda={self.use_cuda}, CUDA_AVAILABLE={CUDA_AVAILABLE}")
        if self.use_cuda and CUDA_AVAILABLE:
            print("🚀 使用CUDA加速重建全景图...")
            self._reconstruct_panorama_cuda(panorama, faces_with_detections, panorama_width, panorama_height, cube_size)
        else:
            print("💻 使用CPU重建全景图...")
            self._reconstruct_panorama_cpu(panorama, faces_with_detections, panorama_width, panorama_height, cube_size)
    
    def _reconstruct_panorama_cuda(self, panorama, faces_with_detections, panorama_width, panorama_height, cube_size):
        """CUDA加速的全景图重建"""
        print("🚀 启动CUDA重建模式...")
        
        try:
            # 将立方体面传输到GPU
            faces_gpu = {}
            for face_name, face_img in faces_with_detections.items():
                faces_gpu[face_name] = cp.asarray(face_img)
            print(f"✅ {len(faces_gpu)} 个立方体面已传输到GPU")
            
            # 使用较小的批次以避免内存问题
            batch_size = 128  # 减小批次大小
            total_batches = (panorama_height + batch_size - 1) // batch_size
            
            print(f"📊 CUDA分批处理: {total_batches} 批次，每批 {batch_size} 行")
            
            for batch_idx in tqdm(range(total_batches), desc="🚀 CUDA重建全景图"):
                start_row = batch_idx * batch_size
                end_row = min(start_row + batch_size, panorama_height)
                current_height = end_row - start_row
                
                # 为当前批次创建坐标网格
                v_coords, u_coords = cp.meshgrid(
                    cp.arange(start_row, end_row),
                    cp.arange(panorama_width),
                    indexing='ij'
                )
                
                # 批量计算球面坐标
                theta = (u_coords / panorama_width) * 2 * cp.pi - cp.pi
                phi = (v_coords / panorama_height) * cp.pi
                
                # 批量计算3D坐标
                x = cp.sin(phi) * cp.sin(theta)
                y = cp.cos(phi)
                z = cp.sin(phi) * cp.cos(theta)
                
                # 批量处理当前批次
                batch_panorama = self._process_batch_cuda_simple(x, y, z, faces_gpu, cube_size)
                
                # 传输回CPU
                panorama[start_row:end_row, :, :] = cp.asnumpy(batch_panorama)
                
                # 清理GPU内存
                del v_coords, u_coords, theta, phi, x, y, z, batch_panorama
                cp.get_default_memory_pool().free_all_blocks()
            
            # 清理GPU立方体面
            for face_name in list(faces_gpu.keys()):
                del faces_gpu[face_name]
            cp.get_default_memory_pool().free_all_blocks()
            
            print("✅ CUDA重建完成")
            
        except Exception as e:
            print(f"❌ CUDA重建失败: {e}")
            print("🔄 回退到CPU重建...")
            self._reconstruct_panorama_cpu(panorama, faces_with_detections, panorama_width, panorama_height, cube_size)
    
    def _process_batch_cuda(self, batch_x, batch_y, batch_z, faces_gpu, cube_size):
        """CUDA批量处理"""
        batch_h, batch_w = batch_x.shape
        batch_panorama = cp.zeros((batch_h, batch_w, 3), dtype=cp.uint8)
        
        # 计算每个面的映射
        face_mappings = self._compute_face_mappings_cuda(batch_x, batch_y, batch_z, cube_size)
        
        # 为每个面应用映射
        for face_name, (face_mask, face_u, face_v) in face_mappings.items():
            if face_name in faces_gpu and cp.any(face_mask):
                face_img = faces_gpu[face_name]
                face_h, face_w = face_img.shape[:2]
                
                # 确保坐标在有效范围内
                face_u = cp.clip(face_u, 0, face_w - 1)
                face_v = cp.clip(face_v, 0, face_h - 1)
                
                # 双线性插值
                sampled_pixels = self._cuda_bilinear_sample_3d(face_img, face_u, face_v, face_mask)
                batch_panorama[face_mask] = sampled_pixels
        
        return batch_panorama
    
    def _compute_face_mappings_cuda(self, x, y, z, cube_size):
        """CUDA计算面映射"""
        abs_x, abs_y, abs_z = cp.abs(x), cp.abs(y), cp.abs(z)
        
        face_mappings = {}
        
        # Front face (z > 0 and abs_z >= abs_x and abs_z >= abs_y)
        front_mask = (z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        if cp.any(front_mask):
            face_u = (x[front_mask] / z[front_mask] + 1) * 0.5 * cube_size
            face_v = (-y[front_mask] / z[front_mask] + 1) * 0.5 * cube_size
            face_mappings['front'] = (front_mask, face_u, face_v)
        
        # Back face (z < 0 and abs_z >= abs_x and abs_z >= abs_y)
        back_mask = (z < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
        if cp.any(back_mask):
            face_u = (-x[back_mask] / (-z[back_mask]) + 1) * 0.5 * cube_size
            face_v = (-y[back_mask] / (-z[back_mask]) + 1) * 0.5 * cube_size
            face_mappings['back'] = (back_mask, face_u, face_v)
        
        # Right face (x > 0 and abs_x >= abs_y and abs_x >= abs_z)
        right_mask = (x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        if cp.any(right_mask):
            face_u = (-z[right_mask] / x[right_mask] + 1) * 0.5 * cube_size
            face_v = (-y[right_mask] / x[right_mask] + 1) * 0.5 * cube_size
            face_mappings['right'] = (right_mask, face_u, face_v)
        
        # Left face (x < 0 and abs_x >= abs_y and abs_x >= abs_z)
        left_mask = (x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
        if cp.any(left_mask):
            face_u = (z[left_mask] / (-x[left_mask]) + 1) * 0.5 * cube_size
            face_v = (-y[left_mask] / (-x[left_mask]) + 1) * 0.5 * cube_size
            face_mappings['left'] = (left_mask, face_u, face_v)
        
        # Top face (y > 0 and abs_y >= abs_x and abs_y >= abs_z)
        top_mask = (y > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
        if cp.any(top_mask):
            face_u = (x[top_mask] / y[top_mask] + 1) * 0.5 * cube_size
            face_v = (z[top_mask] / y[top_mask] + 1) * 0.5 * cube_size
            face_mappings['top'] = (top_mask, face_u, face_v)
        
        # Bottom face (y < 0 and abs_y >= abs_x and abs_y >= abs_z)
        bottom_mask = (y < 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
        if cp.any(bottom_mask):
            face_u = (x[bottom_mask] / (-y[bottom_mask]) + 1) * 0.5 * cube_size
            face_v = (z[bottom_mask] / (-y[bottom_mask]) + 1) * 0.5 * cube_size
            face_mappings['bottom'] = (bottom_mask, face_u, face_v)
        
        return face_mappings
    
    def _cuda_bilinear_sample_3d(self, img_gpu, u, v, mask):
        """CUDA三维双线性插值采样"""
        h, w = img_gpu.shape[:2]
        
        # 获取整数坐标
        u_int = cp.floor(u).astype(cp.int32)
        v_int = cp.floor(v).astype(cp.int32)
        u_frac = u - u_int
        v_frac = v - v_int
        
        # 边界检查
        u_int = cp.clip(u_int, 0, w - 2)
        v_int = cp.clip(v_int, 0, h - 2)
        
        # 获取四个邻近像素
        pixels = cp.zeros((len(u), 3), dtype=cp.uint8)
        
        for c in range(3):
            p00 = img_gpu[v_int, u_int, c]
            p01 = img_gpu[v_int, u_int + 1, c]
            p10 = img_gpu[v_int + 1, u_int, c]
            p11 = img_gpu[v_int + 1, u_int + 1, c]
            
            # 双线性插值
            interpolated = (p00 * (1 - u_frac) * (1 - v_frac) +
                          p01 * u_frac * (1 - v_frac) +
                          p10 * (1 - u_frac) * v_frac +
                          p11 * u_frac * v_frac)
            
            pixels[:, c] = interpolated.astype(cp.uint8)
        
        return pixels
    
    def _process_batch_cuda_simple(self, x, y, z, faces_gpu, cube_size):
        """简化的CUDA批处理"""
        batch_h, batch_w = x.shape
        batch_panorama = cp.zeros((batch_h, batch_w, 3), dtype=cp.uint8)
        
        # 计算绝对值
        abs_x, abs_y, abs_z = cp.abs(x), cp.abs(y), cp.abs(z)
        
        # 处理每个面
        for face_name, face_img in faces_gpu.items():
            face_h, face_w = face_img.shape[:2]
            
            # 确定当前面的掩码和坐标
            if face_name == 'front':
                mask = (z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
                if cp.any(mask):
                    face_u = (x[mask] / z[mask] + 1) * 0.5 * cube_size
                    face_v = (-y[mask] / z[mask] + 1) * 0.5 * cube_size
            elif face_name == 'back':
                mask = (z < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
                if cp.any(mask):
                    face_u = (-x[mask] / (-z[mask]) + 1) * 0.5 * cube_size
                    face_v = (-y[mask] / (-z[mask]) + 1) * 0.5 * cube_size
            elif face_name == 'right':
                mask = (x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
                if cp.any(mask):
                    face_u = (-z[mask] / x[mask] + 1) * 0.5 * cube_size
                    face_v = (-y[mask] / x[mask] + 1) * 0.5 * cube_size
            elif face_name == 'left':
                mask = (x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
                if cp.any(mask):
                    face_u = (z[mask] / (-x[mask]) + 1) * 0.5 * cube_size
                    face_v = (-y[mask] / (-x[mask]) + 1) * 0.5 * cube_size
            elif face_name == 'top':
                mask = (y > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
                if cp.any(mask):
                    face_u = (x[mask] / y[mask] + 1) * 0.5 * cube_size
                    face_v = (z[mask] / y[mask] + 1) * 0.5 * cube_size
            elif face_name == 'bottom':
                mask = (y < 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
                if cp.any(mask):
                    face_u = (x[mask] / (-y[mask]) + 1) * 0.5 * cube_size
                    face_v = (z[mask] / (-y[mask]) + 1) * 0.5 * cube_size
            else:
                continue
            
            if cp.any(mask):
                # 边界检查
                face_u = cp.clip(face_u, 0, face_w - 1)
                face_v = cp.clip(face_v, 0, face_h - 1)
                
                # 简单的最近邻插值
                u_int = cp.round(face_u).astype(cp.int32)
                v_int = cp.round(face_v).astype(cp.int32)
                
                # 采样像素
                sampled_pixels = face_img[v_int, u_int]
                batch_panorama[mask] = sampled_pixels
        
        return batch_panorama
    
    def _reconstruct_panorama_cpu(self, panorama, faces_with_detections, panorama_width, panorama_height, cube_size):
        """CPU版本的全景图重建"""
        print("💻 CPU重建模式 - 预计算坐标映射...")
        
        for v in tqdm(range(panorama_height), desc="CPU重建全景图行"):
            for u in range(panorama_width):
                # 全景图坐标转换为球面坐标
                theta = (u / panorama_width) * 2 * math.pi - math.pi
                phi = (v / panorama_height) * math.pi
                
                # 球面坐标转换为3D坐标
                x = math.sin(phi) * math.sin(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.cos(theta)
                
                # 确定属于哪个立方体面并获取精确坐标
                face_name, face_u, face_v = self._xyz_to_cube_face_precise(x, y, z, cube_size)
                
                if face_name and face_name in faces_with_detections:
                    face_img = faces_with_detections[face_name]
                    face_h, face_w = face_img.shape[:2]
                    
                    # 确保坐标在有效范围内
                    face_u = max(0, min(face_w - 1, face_u))
                    face_v = max(0, min(face_h - 1, face_v))
                    
                    # 使用双线性插值获取更平滑的结果
                    pixel_value = self._bilinear_interpolation(face_img, face_u, face_v)
                    panorama[v, u] = pixel_value
    
    def _xyz_to_cube_face_precise(self, x, y, z, cube_size):
        """精确的3D坐标到立方体面映射"""
        abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
        
        # 找到最大的分量来确定面
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
    
    def _bilinear_interpolation(self, img, x, y):
        """双线性插值获取像素值"""
        h, w = img.shape[:2]
        
        # 获取整数坐标
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        # 计算权重
        wx = x - x1
        wy = y - y1
        
        # 双线性插值
        if len(img.shape) == 3:  # 彩色图像
            pixel = (1 - wy) * ((1 - wx) * img[y1, x1] + wx * img[y1, x2]) + \
                    wy * ((1 - wx) * img[y2, x1] + wx * img[y2, x2])
        else:  # 灰度图像
            pixel = (1 - wy) * ((1 - wx) * img[y1, x1] + wx * img[y1, x2]) + \
                    wy * ((1 - wx) * img[y2, x1] + wx * img[y2, x2])
        
        return pixel.astype(np.uint8)
    
    def _save_feature_images(self, img1, img2, kp1, kp2, output_dir, face_name):
        """保存特征点图像"""
        # 绘制第一张图的特征点
        img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(os.path.join(output_dir, f"03_{face_name}_features_face1.jpg"), img1_kp)
        
        # 绘制第二张图的特征点
        img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(os.path.join(output_dir, f"03_{face_name}_features_face2.jpg"), img2_kp)
    
    def _save_feature_matching_image(self, img1, img2, kp1, kp2, des1, des2, output_dir, face_name):
        """保存特征匹配图像"""
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # 如果没有足够的特征点，创建空白匹配图
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            match_img[:h1, :w1] = img1
            match_img[:h2, w1:w1+w2] = img2
            cv2.putText(match_img, "Insufficient features for matching", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # 使用BFMatcher进行特征匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 只显示前50个最佳匹配
            match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv2.imwrite(os.path.join(output_dir, f"03_{face_name}_feature_matching.jpg"), match_img)
    
    def _save_morphology_steps(self, diff_img, final_binary, output_dir, face_name):
        """保存形态学操作的各个步骤"""
        # 1. 保存原始阈值分割结果
        _, binary_thresh = cv2.threshold(diff_img, self.config['diff_threshold'], 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_dir, f"06a_{face_name}_threshold.jpg"), binary_thresh)
        
        # 2. 创建形态学核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config['morphology_kernel_size'])
        
        # 3. 闭运算（先膨胀后腐蚀）
        closed = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(output_dir, f"06b_{face_name}_morphology_close.jpg"), closed)
        
        # 4. 开运算（先腐蚀后膨胀）
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(output_dir, f"06c_{face_name}_morphology_open.jpg"), opened)
        
        # 5. 保存最终二值图
        cv2.imwrite(os.path.join(output_dir, f"06d_{face_name}_final_binary.jpg"), final_binary)
        
        # 6. 创建形态学操作对比图
        self._create_morphology_comparison(binary_thresh, closed, opened, final_binary, output_dir, face_name)
    
    def _create_morphology_comparison(self, thresh, closed, opened, final, output_dir, face_name):
        """创建形态学操作对比图"""
        # 创建2x2网格显示
        h, w = thresh.shape
        comparison = np.zeros((2*h, 2*w), dtype=np.uint8)
        
        # 左上：原始阈值
        comparison[:h, :w] = thresh
        # 右上：闭运算
        comparison[:h, w:] = closed
        # 左下：开运算  
        comparison[h:, :w] = opened
        # 右下：最终结果
        comparison[h:, w:] = final
        
        # 添加标签
        comparison_color = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
        cv2.putText(comparison_color, "Threshold", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_color, "Close", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_color, "Open", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_color, "Final", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, f"06e_{face_name}_morphology_comparison.jpg"), comparison_color)
    
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
    
    def process_face_pair(self, face1, face2, face_name):
        """
        处理单个立方体面对
        
        Args:
            face1, face2: 两期立方体面图像
            face_name: 面名称
            
        Returns:
            dict: 处理结果
        """
        print(f"🔄 处理 {face_name} 面 ({self.face_descriptions[face_name]})...")
        
        # 创建面专用的输出目录
        face_output_dir = os.path.join(self.output_dir, f"face_{face_name}_steps")
        os.makedirs(face_output_dir, exist_ok=True)
        
        # 1. 保存原始图像
        cv2.imwrite(os.path.join(face_output_dir, f"01_{face_name}_original_face1.jpg"), face1)
        cv2.imwrite(os.path.join(face_output_dir, f"01_{face_name}_original_face2.jpg"), face2)
        
        # 2. 图像预处理
        acceleration = "🚀 CUDA" if self.use_cuda and CUDA_AVAILABLE else "💻 CPU"
        print(f"   预处理加速: {acceleration}")
        processed_face1 = self.preprocess_image(face1)
        processed_face2 = self.preprocess_image(face2)
        
        # 保存预处理结果
        cv2.imwrite(os.path.join(face_output_dir, f"02_{face_name}_preprocessed_face1.jpg"), processed_face1)
        cv2.imwrite(os.path.join(face_output_dir, f"02_{face_name}_preprocessed_face2.jpg"), processed_face2)
        
        # 3. AKAZE特征提取
        kp1, des1 = self.extract_akaze_features(processed_face1)
        kp2, des2 = self.extract_akaze_features(processed_face2)
        
        print(f"   特征点: {len(kp1)} vs {len(kp2)}")
        
        # 保存特征点图像
        self._save_feature_images(processed_face1, processed_face2, kp1, kp2, face_output_dir, face_name)
        
        # 4. 特征匹配和配准
        registered_face2, homography, match_info = self.match_features_and_register(
            processed_face1, processed_face2, kp1, des1, kp2, des2
        )
        
        # 保存配准结果
        cv2.imwrite(os.path.join(face_output_dir, f"04_{face_name}_registered_face2.jpg"), registered_face2)
        
        # 保存特征匹配图像
        self._save_feature_matching_image(processed_face1, processed_face2, kp1, kp2, des1, des2, face_output_dir, face_name)
        
        # 5. 图像差分
        print(f"   差分计算: {acceleration}")
        diff_img = self.compute_image_difference(processed_face1, registered_face2)
        
        # 保存差分图像
        cv2.imwrite(os.path.join(face_output_dir, f"05_{face_name}_difference.jpg"), diff_img)
        
        # 6. 阈值分割和形态学操作
        print(f"   形态学操作: {acceleration}")
        binary_img = self.threshold_and_morphology(diff_img)
        
        # 保存二值化和形态学操作结果
        self._save_morphology_steps(diff_img, binary_img, face_output_dir, face_name)
        
        # 7. 轮廓提取和边界框生成
        face_bboxes, vis_img = self.extract_contours_and_bboxes(binary_img, registered_face2)
        
        # 保存最终检测结果
        cv2.imwrite(os.path.join(face_output_dir, f"07_{face_name}_final_detection.jpg"), vis_img)
        
        result = {
            'face_name': face_name,
            'original_face1': face1,
            'original_face2': face2,
            'processed_face1': processed_face1,
            'processed_face2': processed_face2,
            'registered_face2': registered_face2,
            'diff_image': diff_img,
            'binary_image': binary_img,
            'visualization': vis_img,
            'bboxes': face_bboxes,
            'match_info': match_info,
            'feature_points': {
                'face1_kp_count': len(kp1),
                'face2_kp_count': len(kp2),
                'matches': match_info['matches'],
                'inliers': match_info['inliers']
            },
            'step_images_saved': face_output_dir
        }
        
        print(f"✅ {face_name} 面处理完成，检测到 {len(face_bboxes)} 个变化区域")
        print(f"📁 处理步骤图像已保存至: {face_output_dir}")
        
        return result
    
    def create_comprehensive_visualization(self, all_results, panorama1, panorama2, final_panorama, all_panorama_bboxes):
        """
        创建综合可视化结果
        
        Args:
            all_results: 所有立方体面的处理结果
            panorama1: 第一期全景图
            panorama2: 第二期全景图
            final_panorama: 最终重建的全景图
            all_panorama_bboxes: 所有全景图边界框
            
        Returns:
            str: 保存的可视化图像路径
        """
        print("🎨 创建综合可视化...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建大型图形
        fig = plt.figure(figsize=(24, 18))
        
        # 主标题
        fig.suptitle('全景图像变化检测系统 - 综合分析结果', fontsize=20, fontweight='bold', y=0.98)
        
        # 创建网格布局
        gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
        
        # 第一行：全景图对比
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(cv2.cvtColor(panorama1, cv2.COLOR_BGR2RGB))
        ax1.set_title('第一期全景图（基准）', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(cv2.cvtColor(panorama2, cv2.COLOR_BGR2RGB))
        ax2.set_title('第二期全景图（待检测）', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 带检测结果的全景图
        panorama_with_detections = final_panorama.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, bbox_info in enumerate(all_panorama_bboxes):
            if 'panorama_bbox' in bbox_info:
                x, y, w, h = bbox_info['panorama_bbox']
                color = colors[i % len(colors)]
                cv2.rectangle(panorama_with_detections, (x, y), (x+w, y+h), color, 3)
                cv2.putText(panorama_with_detections, f"变化{i+1}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        ax3 = fig.add_subplot(gs[0, 4:])
        ax3.imshow(cv2.cvtColor(panorama_with_detections, cv2.COLOR_BGR2RGB))
        ax3.set_title(f'检测结果全景图 (发现 {len(all_panorama_bboxes)} 个变化区域)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 第二行和第三行：每个立方体面的详细结果
        valid_results = [r for r in all_results if len(r['bboxes']) > 0]
        
        row_idx = 1
        col_idx = 0
        for result in valid_results[:6]:  # 最多显示6个有检测结果的面
            if row_idx >= 3:
                break
                
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB))
            ax.set_title(f"{result['face_name']} 面\n检测: {len(result['bboxes'])}个", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            col_idx += 1
            if col_idx >= 6:
                col_idx = 0
                row_idx += 1
        
        # 第四行：统计信息
        # 总体统计
        total_faces_processed = len(all_results)
        total_detections = sum(len(r['bboxes']) for r in all_results)
        faces_with_changes = len([r for r in all_results if len(r['bboxes']) > 0])
        
        ax_stats1 = fig.add_subplot(gs[3, :2])
        stats_text = f"""📊 检测统计摘要
        
总处理面数: {total_faces_processed} / {len(self.face_names)}
发现变化的面: {faces_with_changes}
总变化区域数: {total_detections}
全景图检测框: {len(all_panorama_bboxes)}

系统配置:
• 立方体尺寸: {self.config['cube_size']}×{self.config['cube_size']} (动态)
• 差异阈值: {self.config['diff_threshold']}
• 最小区域面积: {self.config['min_contour_area']} px²
• 跳过面: {', '.join(self.config['skip_faces'])}
• 使用GPU加速: {'是' if self.use_cuda else '否'}
"""
        
        ax_stats1.text(0.05, 0.95, stats_text, transform=ax_stats1.transAxes, fontsize=11,
                      verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax_stats1.set_xlim(0, 1)
        ax_stats1.set_ylim(0, 1)
        ax_stats1.axis('off')
        ax_stats1.set_title('系统统计', fontsize=12, fontweight='bold')
        
        # 面详细信息
        ax_stats2 = fig.add_subplot(gs[3, 2:4])
        face_details = "🔍 各面检测详情\n\n"
        
        # 显示处理的面
        for result in all_results:
            status = f"✅ {len(result['bboxes'])}个" if len(result['bboxes']) > 0 else "⭕ 无变化"
            face_details += f"{result['face_name']} ({self.face_descriptions[result['face_name']]}): {status}\n"
            if len(result['bboxes']) > 0:
                face_details += f"  特征匹配: {result['match_info']['matches']}个\n"
                face_details += f"  内点比例: {result['match_info']['inlier_ratio']:.2%}\n"
        
        # 显示跳过的面
        for face_name in self.config['skip_faces']:
            face_details += f"{face_name} ({self.face_descriptions[face_name]}): ⏭️ 已跳过\n"
        
        ax_stats2.text(0.05, 0.95, face_details, transform=ax_stats2.transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        ax_stats2.set_xlim(0, 1)
        ax_stats2.set_ylim(0, 1)
        ax_stats2.axis('off')
        ax_stats2.set_title('面处理详情', fontsize=12, fontweight='bold')
        
        # 技术流程说明
        ax_tech = fig.add_subplot(gs[3, 4:])
        tech_text = """🔬 技术处理流程
        
1️⃣ 全景图立方体分割
   • 6个立方体面提取
   • GPU/CPU自适应处理

2️⃣ 图像预处理
   • 高斯模糊去噪
   • CLAHE直方图均衡化

3️⃣ 特征提取与配准
   • AKAZE特征点检测
   • BF匹配器特征匹配
   • RANSAC单应性变换

4️⃣ 变化检测
   • 图像差分计算
   • 自适应阈值分割
   • 形态学操作优化

5️⃣ 目标识别
   • 轮廓提取分析
   • 几何特征过滤
   • 置信度评估

6️⃣ 结果映射
   • 坐标系逆变换
   • 全景图重建
   • 检测框可视化
"""
        
        ax_tech.text(0.05, 0.95, tech_text, transform=ax_tech.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        ax_tech.set_xlim(0, 1)
        ax_tech.set_ylim(0, 1)
        ax_tech.axis('off')
        ax_tech.set_title('技术流程', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.output_dir, f"panorama_change_detection_comprehensive_{timestamp}.jpg")
        plt.savefig(image_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 综合可视化已保存: {image_path}")
        return image_path
    
    def save_results(self, all_results, all_panorama_bboxes):
        """
        保存检测结果
        
        Args:
            all_results: 所有处理结果
            all_panorama_bboxes: 全景图边界框
            
        Returns:
            str: 保存的JSON文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备可序列化的结果
        serializable_results = []
        for result in all_results:
            serializable_result = {
                'face_name': result['face_name'],
                'bboxes': result['bboxes'],
                'match_info': result['match_info'],
                'feature_points': result['feature_points'],
                'detection_count': len(result['bboxes'])
            }
            serializable_results.append(serializable_result)
        
        # 创建完整结果
        final_results = {
            'timestamp': timestamp,
            'system_config': self.config,
            'processing_summary': {
                'total_faces_processed': len(all_results),
                'faces_with_detections': len([r for r in all_results if len(r['bboxes']) > 0]),
                'total_detections': sum(len(r['bboxes']) for r in all_results),
                'panorama_bboxes_count': len(all_panorama_bboxes),
                'gpu_acceleration_used': self.use_cuda
            },
            'face_results': serializable_results,
            'panorama_bboxes': all_panorama_bboxes
        }
        
        # 保存JSON结果
        json_path = os.path.join(self.output_dir, f"detection_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 检测结果已保存: {json_path}")
        return json_path
    
    def process_panorama_pair(self, panorama1_path, panorama2_path):
        """
        处理全景图对的完整流程
        
        Args:
            panorama1_path (str): 第一期全景图路径
            panorama2_path (str): 第二期全景图路径
            
        Returns:
            dict: 完整的处理结果
        """
        print("🚀 开始全景图像变化检测系统处理...")
        print(f"📂 第一期图像: {os.path.basename(panorama1_path)}")
        print(f"📂 第二期图像: {os.path.basename(panorama2_path)}")
        
        # 1. 加载全景图
        print("\n📖 Step 1: 加载全景图像...")
        panorama1 = self.load_image_with_chinese_path(panorama1_path)
        panorama2 = self.load_image_with_chinese_path(panorama2_path)
        
        if panorama1 is None or panorama2 is None:
            print("❌ 图像加载失败")
            return None
        
        print(f"   第一期尺寸: {panorama1.shape}")
        print(f"   第二期尺寸: {panorama2.shape}")
        
        # 2. 立方体分割
        print("\n🔄 Step 2: 全景图立方体分割...")
        faces1 = self.panorama_to_cubemap(panorama1)
        faces2 = self.panorama_to_cubemap(panorama2)
        
        # 3. 处理每个立方体面对
        print("\n🔍 Step 3: 处理各立方体面...")
        all_results = []
        faces_with_detections = {}
        
        for face_name in self.face_names:
            # 跳过指定的面（如top面）
            if face_name in self.config['skip_faces']:
                print(f"⏭️ 跳过 {face_name} 面 ({self.face_descriptions[face_name]})")
                # 为跳过的面使用原始图像
                faces_with_detections[face_name] = faces2[face_name] if face_name in faces2 else faces1[face_name]
                continue
                
            if face_name in faces1 and face_name in faces2:
                face_result = self.process_face_pair(faces1[face_name], faces2[face_name], face_name)
                all_results.append(face_result)
                
                # 保存带有检测结果的面图像
                faces_with_detections[face_name] = face_result['visualization']
        
        # 4. 映射边界框到全景图
        print("\n🗺️ Step 4: 映射检测结果到全景图...")
        all_panorama_bboxes = []
        panorama_height, panorama_width = panorama1.shape[:2]
        
        for result in all_results:
            if result['bboxes']:
                face_panorama_bboxes = self.map_bboxes_to_panorama(
                    result['bboxes'], 
                    result['face_name'],
                    self.config['cube_size'],
                    panorama_width,
                    panorama_height
                )
                all_panorama_bboxes.extend(face_panorama_bboxes)
        
        print(f"   映射了 {len(all_panorama_bboxes)} 个检测框到全景图")
        
        # 5. 重建全景图
        print("\n🔄 Step 5: 重建带检测结果的全景图...")
        final_panorama = self.reconstruct_panorama_with_detections(
            faces_with_detections, panorama_width, panorama_height
        )
        
        # 6. 创建可视化
        print("\n🎨 Step 6: 创建综合可视化...")
        visualization_path = self.create_comprehensive_visualization(
            all_results, panorama1, panorama2, final_panorama, all_panorama_bboxes
        )
        
        # 7. 保存结果
        print("\n💾 Step 7: 保存检测结果...")
        results_path = self.save_results(all_results, all_panorama_bboxes)
        
        # 保存重建的全景图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_panorama_path = os.path.join(self.output_dir, f"final_panorama_with_detections_{timestamp}.jpg")
        cv2.imwrite(final_panorama_path, final_panorama)
        
        # 汇总结果
        summary = {
            'processing_successful': True,
            'total_faces_processed': len(all_results),
            'faces_with_detections': len([r for r in all_results if len(r['bboxes']) > 0]),
            'total_detection_count': sum(len(r['bboxes']) for r in all_results),
            'panorama_bboxes_count': len(all_panorama_bboxes),
            'output_files': {
                'comprehensive_visualization': visualization_path,
                'detection_results_json': results_path,
                'final_panorama_image': final_panorama_path
            },
            'face_results': all_results,
            'panorama_bboxes': all_panorama_bboxes
        }
        
        print("\n🎉 全景图像变化检测完成！")
        print(f"📊 处理结果摘要:")
        print(f"   • 处理立方体面: {summary['total_faces_processed']}")
        print(f"   • 有检测结果的面: {summary['faces_with_detections']}")
        print(f"   • 总检测区域: {summary['total_detection_count']}")
        print(f"   • 全景图检测框: {summary['panorama_bboxes_count']}")
        print(f"📁 主要输出文件:")
        print(f"   • 综合可视化: {os.path.basename(visualization_path)}")
        print(f"   • 结果数据: {os.path.basename(results_path)}")
        print(f"   • 最终全景图: {os.path.basename(final_panorama_path)}")
        print(f"📁 详细处理步骤:")
        for result in all_results:
            if 'step_images_saved' in result:
                print(f"   • {result['face_name']} 面步骤图像: {os.path.basename(result['step_images_saved'])}/")
        
        return summary


def main():
    """主函数 - 演示系统使用"""
    # 测试图像路径
    panorama1_path = os.path.join("test", "20250910164040_0002_V.jpeg")  # 第一期全景图
    panorama2_path = os.path.join("test", "20250910164151_0003_V.jpeg")  # 第二期全景图
    
    # 检查测试图像是否存在
    if not os.path.exists(panorama1_path) or not os.path.exists(panorama2_path):
        print("❌ 测试图像不存在，请将图像放入 test/ 目录")
        print(f"需要的图像:")
        print(f"  - {panorama1_path}")
        print(f"  - {panorama2_path}")
        return
    
    try:
        # 创建检测系统
        system = PanoramaChangeDetectionSystem(
            output_dir="panorama_change_detection_results",
            use_cuda=True  # 尝试使用GPU加速
        )
        
        # 执行完整的检测流程
        results = system.process_panorama_pair(panorama1_path, panorama2_path)
        
        if results and results['processing_successful']:
            print("\n✅ 系统运行成功！")
            
            # 可以继续进行其他分析...
            if results['total_detection_count'] > 0:
                print("🔍 发现图像变化，建议进一步人工审核")
            else:
                print("📝 未发现显著变化，图像基本一致")
        else:
            print("❌ 系统运行失败")
            
    except Exception as e:
        print(f"❌ 系统运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 