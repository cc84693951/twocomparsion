#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA工具函数
提供CUDA加速的图像处理操作和内存管理
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Union, Any
from functools import wraps

# CUDA支持检测
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    logging.info("CuPy已加载，CUDA加速可用")
except ImportError:
    CUDA_AVAILABLE = False
    logging.warning("CuPy未安装，将使用CPU处理")
    # 创建CuPy的替代实现
    class cp:
        @staticmethod
        def array(x): return np.array(x)
        @staticmethod
        def asnumpy(x): return np.array(x)
        @staticmethod
        def asarray(x): return np.array(x)
        @staticmethod
        def zeros_like(x): return np.zeros_like(x)
        @staticmethod
        def zeros(shape, dtype=None): return np.zeros(shape, dtype=dtype)
        @staticmethod
        def ones_like(x): return np.ones_like(x)
        @staticmethod
        def sqrt(x): return np.sqrt(x)
        @staticmethod
        def sin(x): return np.sin(x)
        @staticmethod
        def cos(x): return np.cos(x)
        @staticmethod
        def arctan2(x, y): return np.arctan2(x, y)
        @staticmethod
        def arccos(x): return np.arccos(x)
        @staticmethod
        def meshgrid(*args, **kwargs): return np.meshgrid(*args, **kwargs)
        @staticmethod
        def arange(*args, **kwargs): return np.arange(*args, **kwargs)
        @staticmethod
        def stack(*args, **kwargs): return np.stack(*args, **kwargs)
        @staticmethod
        def abs(x): return np.abs(x)
        @staticmethod
        def any(x): return np.any(x)
        @staticmethod
        def floor(x): return np.floor(x)
        @staticmethod
        def clip(x, a_min, a_max): return np.clip(x, a_min, a_max)
        @staticmethod
        def where(condition, x, y): return np.where(condition, x, y)
        @staticmethod
        def sum(x, axis=None): return np.sum(x, axis=axis)
        @staticmethod
        def mean(x, axis=None): return np.mean(x, axis=axis)
        @staticmethod
        def std(x, axis=None): return np.std(x, axis=axis)
        @staticmethod
        def logical_and(x, y): return np.logical_and(x, y)
        @staticmethod
        def logical_or(x, y): return np.logical_or(x, y)
        @staticmethod
        def maximum(x, y): return np.maximum(x, y)
        @staticmethod
        def minimum(x, y): return np.minimum(x, y)
        @staticmethod
        def exp(x): return np.exp(x)
        @staticmethod
        def log(x): return np.log(x)
        @staticmethod
        def power(x, y): return np.power(x, y)
        pi = np.pi
        uint8 = np.uint8
        float32 = np.float32
        int32 = np.int32


def ensure_cuda_available(func):
    """装饰器：确保CUDA可用，否则使用CPU版本"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.use_cuda or not CUDA_AVAILABLE:
            # 如果有对应的CPU版本方法，调用它
            cpu_method_name = func.__name__.replace('_cuda', '_cpu')
            if hasattr(self, cpu_method_name):
                cpu_method = getattr(self, cpu_method_name)
                return cpu_method(*args, **kwargs)
            else:
                # 没有CPU版本，强制使用CPU处理
                old_use_cuda = self.use_cuda
                self.use_cuda = False
                try:
                    result = func(self, *args, **kwargs)
                finally:
                    self.use_cuda = old_use_cuda
                return result
        return func(self, *args, **kwargs)
    return wrapper


class CUDAUtils:
    """CUDA工具类"""
    
    def __init__(self, use_cuda: bool = True, device_id: int = 0):
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.device_id = device_id
        
        if self.use_cuda:
            try:
                cp.cuda.Device(device_id).use()
                self.device = cp.cuda.Device(device_id)
                logging.info(f"CUDA设备 {device_id} 已激活")
            except Exception as e:
                logging.warning(f"CUDA初始化失败: {e}, 将使用CPU")
                self.use_cuda = False
    
    def cleanup_memory(self):
        """清理GPU内存"""
        if self.use_cuda:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                logging.warning(f"GPU内存清理失败: {e}")
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """将numpy数组转换到GPU"""
        if self.use_cuda:
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """将数组转换到CPU"""
        if self.use_cuda and hasattr(array, '__array_interface__'):
            return cp.asnumpy(array)
        return np.array(array)
    
    @ensure_cuda_available
    def bilateral_filter_cuda(self, image: np.ndarray, d: int, sigma_color: float, 
                             sigma_space: float) -> np.ndarray:
        """CUDA加速的双边滤波"""
        if not self.use_cuda:
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # GPU双边滤波实现
        img_gpu = self.to_gpu(image)
        
        # 使用CUDA内核实现双边滤波（简化版）
        # 这里可以根据需要实现更复杂的CUDA内核
        result_gpu = self._bilateral_filter_kernel(img_gpu, d, sigma_color, sigma_space)
        
        return self.to_cpu(result_gpu)
    
    def _bilateral_filter_kernel(self, img_gpu, d, sigma_color, sigma_space):
        """双边滤波CUDA内核（简化实现）"""
        # 这是一个简化的实现，实际中可以用更优化的CUDA内核
        height, width = img_gpu.shape[:2]
        result = cp.zeros_like(img_gpu)
        
        # 创建空间权重矩阵
        half_d = d // 2
        spatial_weights = cp.zeros((d, d), dtype=cp.float32)
        for i in range(d):
            for j in range(d):
                spatial_dist = ((i - half_d) ** 2 + (j - half_d) ** 2) ** 0.5
                spatial_weights[i, j] = cp.exp(-(spatial_dist ** 2) / (2 * sigma_space ** 2))
        
        # 对每个像素进行双边滤波
        padded_img = cp.pad(img_gpu, ((half_d, half_d), (half_d, half_d), (0, 0)), mode='reflect')
        
        for y in range(height):
            for x in range(width):
                center_pixel = padded_img[y + half_d, x + half_d]
                
                weights_sum = 0
                filtered_pixel = cp.zeros_like(center_pixel, dtype=cp.float32)
                
                for ky in range(d):
                    for kx in range(d):
                        neighbor_pixel = padded_img[y + ky, x + kx]
                        
                        # 计算颜色权重
                        color_diff = cp.sum((center_pixel - neighbor_pixel) ** 2) ** 0.5
                        color_weight = cp.exp(-(color_diff ** 2) / (2 * sigma_color ** 2))
                        
                        # 综合权重
                        weight = spatial_weights[ky, kx] * color_weight
                        weights_sum += weight
                        filtered_pixel += weight * neighbor_pixel.astype(cp.float32)
                
                result[y, x] = (filtered_pixel / weights_sum).astype(img_gpu.dtype)
        
        return result
    
    @ensure_cuda_available
    def gaussian_blur_cuda(self, image: np.ndarray, kernel_size: int, sigma: float = 0) -> np.ndarray:
        """CUDA加速的高斯滤波"""
        if not self.use_cuda:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        img_gpu = self.to_gpu(image)
        
        # 生成高斯核
        if sigma == 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        
        kernel = self._create_gaussian_kernel_cuda(kernel_size, sigma)
        
        # 进行卷积
        result_gpu = self._convolve2d_cuda(img_gpu, kernel)
        
        return self.to_cpu(result_gpu)
    
    def _create_gaussian_kernel_cuda(self, size: int, sigma: float):
        """创建高斯核"""
        kernel = cp.zeros((size, size), dtype=cp.float32)
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = cp.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        return kernel / cp.sum(kernel)
    
    def _convolve2d_cuda(self, image, kernel):
        """2D卷积"""
        if len(image.shape) == 3:
            height, width, channels = image.shape
            result = cp.zeros_like(image)
            for c in range(channels):
                result[:, :, c] = self._convolve2d_single_channel(image[:, :, c], kernel)
        else:
            result = self._convolve2d_single_channel(image, kernel)
        
        return result
    
    def _convolve2d_single_channel(self, image, kernel):
        """单通道2D卷积"""
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2
        
        # 填充图像
        padded_image = cp.pad(image, padding, mode='reflect')
        
        height, width = image.shape
        result = cp.zeros_like(image)
        
        for y in range(height):
            for x in range(width):
                patch = padded_image[y:y+kernel_size, x:x+kernel_size]
                result[y, x] = cp.sum(patch * kernel)
        
        return result
    
    @ensure_cuda_available
    def clahe_cuda(self, image: np.ndarray, clip_limit: float = 3.0, 
                   grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """CUDA加速的CLAHE"""
        if not self.use_cuda:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            if len(image.shape) == 3:
                # 对于彩色图像，转换到LAB空间处理L通道
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return clahe.apply(image)
        
        img_gpu = self.to_gpu(image)
        
        if len(image.shape) == 3:
            # 彩色图像：转换到LAB空间
            lab_gpu = self._bgr_to_lab_cuda(img_gpu)
            lab_gpu[:, :, 0] = self._apply_clahe_cuda(lab_gpu[:, :, 0], clip_limit, grid_size)
            result_gpu = self._lab_to_bgr_cuda(lab_gpu)
        else:
            # 灰度图像
            result_gpu = self._apply_clahe_cuda(img_gpu, clip_limit, grid_size)
        
        return self.to_cpu(result_gpu)
    
    def _bgr_to_lab_cuda(self, bgr_img):
        """BGR到LAB颜色空间转换（简化版）"""
        # 这是一个简化实现，实际可以使用更精确的颜色空间转换
        bgr_float = bgr_img.astype(cp.float32) / 255.0
        
        # 简化的RGB到XYZ转换
        b, g, r = bgr_float[:, :, 0], bgr_float[:, :, 1], bgr_float[:, :, 2]
        
        # XYZ
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        
        # LAB（简化版）
        l = 116 * cp.power(y, 1/3) - 16
        a = 500 * (cp.power(x, 1/3) - cp.power(y, 1/3))
        b = 200 * (cp.power(y, 1/3) - cp.power(z, 1/3))
        
        lab_img = cp.stack([l, a, b], axis=2)
        return lab_img
    
    def _lab_to_bgr_cuda(self, lab_img):
        """LAB到BGR颜色空间转换（简化版）"""
        # 这是一个简化实现，与_bgr_to_lab_cuda对应
        l, a, b = lab_img[:, :, 0], lab_img[:, :, 1], lab_img[:, :, 2]
        
        # LAB到XYZ
        fy = (l + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        x = cp.power(fx, 3)
        y = cp.power(fy, 3)
        z = cp.power(fz, 3)
        
        # XYZ到RGB
        r = 3.240479 * x - 1.537150 * y - 0.498535 * z
        g = -0.969256 * x + 1.875992 * y + 0.041556 * z
        b = 0.055648 * x - 0.204043 * y + 1.057311 * z
        
        # 转换回0-255范围
        rgb_img = cp.stack([b, g, r], axis=2)
        rgb_img = cp.clip(rgb_img * 255, 0, 255).astype(cp.uint8)
        
        return rgb_img
    
    def _apply_clahe_cuda(self, image, clip_limit, grid_size):
        """应用CLAHE"""
        # 简化的CLAHE实现
        height, width = image.shape
        tile_height = height // grid_size[0]
        tile_width = width // grid_size[1]
        
        result = cp.zeros_like(image)
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y_start = i * tile_height
                y_end = min((i + 1) * tile_height, height)
                x_start = j * tile_width
                x_end = min((j + 1) * tile_width, width)
                
                tile = image[y_start:y_end, x_start:x_end]
                
                # 计算直方图
                hist, _ = cp.histogram(tile, bins=256, range=(0, 256))
                
                # 应用剪切限制
                excess = cp.sum(cp.maximum(hist - clip_limit, 0))
                hist = cp.minimum(hist, clip_limit)
                hist += excess / 256
                
                # 计算累积分布函数
                cdf = cp.cumsum(hist)
                cdf = cdf / cdf[-1] * 255
                
                # 应用映射
                result[y_start:y_end, x_start:x_end] = cdf[tile]
        
        return result.astype(cp.uint8)
    
    @ensure_cuda_available
    def morphology_cuda(self, image: np.ndarray, operation: str, 
                       kernel_size: int, kernel_shape: str = 'ellipse') -> np.ndarray:
        """CUDA加速的形态学操作"""
        if not self.use_cuda:
            if kernel_shape == 'ellipse':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            elif kernel_shape == 'rect':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            else:  # cross
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            
            if operation == 'open':
                return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            elif operation == 'close':
                return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            elif operation == 'erode':
                return cv2.erode(image, kernel)
            elif operation == 'dilate':
                return cv2.dilate(image, kernel)
            else:
                return image
        
        img_gpu = self.to_gpu(image)
        kernel_gpu = self._create_morphology_kernel_cuda(kernel_size, kernel_shape)
        
        if operation == 'erode':
            result_gpu = self._erode_cuda(img_gpu, kernel_gpu)
        elif operation == 'dilate':
            result_gpu = self._dilate_cuda(img_gpu, kernel_gpu)
        elif operation == 'open':
            eroded = self._erode_cuda(img_gpu, kernel_gpu)
            result_gpu = self._dilate_cuda(eroded, kernel_gpu)
        elif operation == 'close':
            dilated = self._dilate_cuda(img_gpu, kernel_gpu)
            result_gpu = self._erode_cuda(dilated, kernel_gpu)
        else:
            result_gpu = img_gpu
        
        return self.to_cpu(result_gpu)
    
    def _create_morphology_kernel_cuda(self, size: int, shape: str):
        """创建形态学核"""
        kernel = cp.zeros((size, size), dtype=cp.uint8)
        center = size // 2
        
        if shape == 'rect':
            kernel[:, :] = 1
        elif shape == 'ellipse':
            for i in range(size):
                for j in range(size):
                    if ((i - center) ** 2 + (j - center) ** 2) <= center ** 2:
                        kernel[i, j] = 1
        else:  # cross
            kernel[center, :] = 1
            kernel[:, center] = 1
        
        return kernel
    
    def _erode_cuda(self, image, kernel):
        """腐蚀操作"""
        return self._morphology_operation_cuda(image, kernel, 'min')
    
    def _dilate_cuda(self, image, kernel):
        """膨胀操作"""
        return self._morphology_operation_cuda(image, kernel, 'max')
    
    def _morphology_operation_cuda(self, image, kernel, op):
        """通用形态学操作"""
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2
        
        padded_image = cp.pad(image, padding, mode='constant', constant_values=0 if op == 'max' else 255)
        
        height, width = image.shape
        result = cp.zeros_like(image)
        
        for y in range(height):
            for x in range(width):
                patch = padded_image[y:y+kernel_size, x:x+kernel_size]
                masked_patch = patch[kernel == 1]
                
                if len(masked_patch) > 0:
                    if op == 'min':
                        result[y, x] = cp.min(masked_patch)
                    else:  # max
                        result[y, x] = cp.max(masked_patch)
                else:
                    result[y, x] = image[y, x]
        
        return result
    
    @ensure_cuda_available
    def threshold_cuda(self, image: np.ndarray, threshold: float, 
                      method: str = 'binary') -> Tuple[np.ndarray, float]:
        """CUDA加速的阈值处理"""
        if not self.use_cuda:
            if method == 'binary':
                _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
                return result, threshold
            elif method == 'otsu':
                threshold, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return result, threshold
            else:
                return image, threshold
        
        img_gpu = self.to_gpu(image)
        
        if method == 'otsu':
            threshold = self._otsu_threshold_cuda(img_gpu)
        
        if method in ['binary', 'otsu']:
            result_gpu = cp.where(img_gpu > threshold, 255, 0).astype(cp.uint8)
        else:
            result_gpu = img_gpu
        
        return self.to_cpu(result_gpu), float(threshold)
    
    def _otsu_threshold_cuda(self, image):
        """CUDA版本的Otsu阈值计算"""
        # 计算直方图
        hist, _ = cp.histogram(image, bins=256, range=(0, 256))
        hist = hist.astype(cp.float32)
        
        # 归一化直方图
        total_pixels = cp.sum(hist)
        hist_norm = hist / total_pixels
        
        # 计算最优阈值
        max_variance = 0
        optimal_threshold = 0
        
        for t in range(256):
            # 计算类间方差
            w0 = cp.sum(hist_norm[:t])
            w1 = cp.sum(hist_norm[t:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = cp.sum(cp.arange(t) * hist_norm[:t]) / w0
            mu1 = cp.sum(cp.arange(t, 256) * hist_norm[t:]) / w1
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = t
        
        return optimal_threshold
    
    def get_memory_info(self) -> dict:
        """获取GPU内存信息"""
        if not self.use_cuda:
            return {"available": False, "message": "CUDA不可用"}
        
        try:
            mempool = cp.get_default_memory_pool()
            return {
                "available": True,
                "used_bytes": mempool.used_bytes(),
                "total_bytes": mempool.total_bytes(),
                "device_id": self.device_id
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


def load_image_with_chinese_path(path):
    """
    加载包含中文路径的图像
    """
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return img
    except Exception as e:
        logging.error(f"读取图像失败 {path}: {str(e)}")
        return None


def cuda_bilinear_sample(img_gpu, u_coords, v_coords, valid_mask, output_shape):
    """
    CUDA双线性插值采样
    Args:
        img_gpu (cp.ndarray): 输入图像 (GPU)
        u_coords (cp.ndarray): 采样点的u坐标 (GPU)
        v_coords (cp.ndarray): 采样点的v坐标 (GPU)
        valid_mask (cp.ndarray): 有效采样点的掩码 (GPU)
        output_shape (tuple): 输出图像的形状 (H, W)
    Returns:
        cp.ndarray: 采样后的图像 (GPU)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available. Cannot use cuda_bilinear_sample.")

    height, width = img_gpu.shape[:2]
    output_img_gpu = cp.zeros(output_shape + (3,), dtype=cp.uint8)

    # 获取有效点的索引
    valid_rows, valid_cols = cp.where(valid_mask)
    if len(valid_rows) == 0:
        return output_img_gpu

    # 提取有效点的坐标
    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]

    # 获取整数和小数部分
    u_int = cp.floor(u_valid).astype(cp.int32)
    v_int = cp.floor(v_valid).astype(cp.int32)
    u_frac = u_valid - u_int
    v_frac = v_valid - v_int

    # 确保索引在边界内
    u_int = cp.clip(u_int, 0, width - 2)
    v_int = cp.clip(v_int, 0, height - 2)

    # 双线性插值
    for c in range(3):  # BGR通道
        p00 = img_gpu[v_int, u_int, c]
        p01 = img_gpu[v_int, u_int + 1, c]
        p10 = img_gpu[v_int + 1, u_int, c]
        p11 = img_gpu[v_int + 1, u_int + 1, c]

        interpolated = (p00 * (1 - u_frac) * (1 - v_frac) +
                        p01 * u_frac * (1 - v_frac) +
                        p10 * (1 - u_frac) * v_frac +
                        p11 * u_frac * v_frac)

        output_img_gpu[valid_rows, valid_cols, c] = interpolated.astype(cp.uint8)

    return output_img_gpu
