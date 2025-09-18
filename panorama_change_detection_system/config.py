#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图变化检测系统配置文件
配置所有模块的参数和CUDA设置
"""

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class CUDAConfig:
    """CUDA配置"""
    use_cuda: bool = True
    device_id: int = 0
    memory_pool_cleanup: bool = True
    
@dataclass
class PanoramaSplitterConfig:
    """全景图分割配置"""
    cube_size: int = 1024
    interpolation_method: str = "bilinear"
    face_names: Tuple[str, ...] = ('front', 'right', 'back', 'left', 'top', 'bottom')
    output_format: str = "jpg"
    
@dataclass
class ImagePreprocessorConfig:
    """图像预处理配置"""
    # 去噪参数
    denoise_method: str = "bilateral"  # "bilateral", "gaussian", "median"
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    gaussian_kernel_size: int = 5
    median_kernel_size: int = 5
    
    # 直方图均衡化参数
    enable_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # 光照归一化参数
    enable_lighting_normalization: bool = True
    conservative_mode: bool = True
    
@dataclass
class ImageRegistrationConfig:
    """图像配准配置"""
    # AKAZE特征检测参数
    akaze_descriptor_type: int = 5  # AKAZE.DESCRIPTOR_MLDB
    akaze_descriptor_size: int = 0
    akaze_descriptor_channels: int = 3
    akaze_threshold: float = 0.001
    akaze_nOctaves: int = 4
    akaze_nOctaveLayers: int = 4
    
    # 特征匹配参数
    matcher_type: str = "BF"  # "BF" or "FLANN"
    match_ratio_threshold: float = 0.7
    min_match_count: int = 10
    
    # RANSAC参数
    ransac_threshold: float = 5.0
    ransac_confidence: float = 0.99
    ransac_max_iterations: int = 2000
    
    # 掩码处理参数
    mask_threshold: int = 10
    mask_morphology_kernel_size: int = 5
    
@dataclass
class ChangeDetectorConfig:
    """变化检测配置"""
    # 差分计算参数
    diff_method: str = "absdiff"  # "absdiff", "background_subtraction"
    threshold_method: str = "otsu"  # "otsu", "adaptive", "fixed"
    fixed_threshold: int = 50
    
    # 形态学操作参数
    morphology_operations: list = None  # ["close", "open"]
    close_kernel_size: int = 5
    open_kernel_size: int = 3
    
    # 轮廓过滤参数
    min_contour_area: int = 500
    max_contour_area: int = 100000
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    min_extent: float = 0.1  # 轮廓面积与边界框面积的比值
    
    # 边界框参数
    bbox_padding: int = 5  # 边界框扩展像素
    
    def __post_init__(self):
        if self.morphology_operations is None:
            self.morphology_operations = ["close", "open"]

@dataclass
class ResultMapperConfig:
    """结果映射配置"""
    # 坐标变换参数
    enable_inverse_transform: bool = True
    interpolation_method: str = "bilinear"
    
    # 全景图重建参数
    reconstruction_method: str = "improved"  # "basic", "improved"
    enable_boundary_handling: bool = True
    
@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 输出图像参数
    output_dpi: int = 150
    figure_size: Tuple[int, int] = (20, 15)
    
    # 检测框颜色
    bbox_colors: list = None
    bbox_thickness: int = 3
    bbox_alpha: float = 0.3
    
    # 字体设置
    font_scale: float = 0.8
    font_thickness: int = 2
    
    # 中间结果保存
    save_intermediate: bool = True
    intermediate_format: str = "jpg"
    
    def __post_init__(self):
        if self.bbox_colors is None:
            self.bbox_colors = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
            ]

@dataclass
class SystemConfig:
    """系统总配置"""
    cuda: CUDAConfig = None
    panorama_splitter: PanoramaSplitterConfig = None
    image_preprocessor: ImagePreprocessorConfig = None
    image_registration: ImageRegistrationConfig = None
    change_detector: ChangeDetectorConfig = None
    result_mapper: ResultMapperConfig = None
    visualization: VisualizationConfig = None
    
    # 系统路径配置
    output_root: str = "results"
    temp_dir: str = "temp"
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    def __post_init__(self):
        if self.cuda is None:
            self.cuda = CUDAConfig()
        if self.panorama_splitter is None:
            self.panorama_splitter = PanoramaSplitterConfig()
        if self.image_preprocessor is None:
            self.image_preprocessor = ImagePreprocessorConfig()
        if self.image_registration is None:
            self.image_registration = ImageRegistrationConfig()
        if self.change_detector is None:
            self.change_detector = ChangeDetectorConfig()
        if self.result_mapper is None:
            self.result_mapper = ResultMapperConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()

def get_default_config() -> SystemConfig:
    """获取默认配置"""
    return SystemConfig()

def load_config_from_dict(config_dict: Dict[str, Any]) -> SystemConfig:
    """从字典加载配置"""
    return SystemConfig(**config_dict)

def save_config_to_file(config: SystemConfig, filepath: str):
    """保存配置到文件"""
    import json
    config_dict = {
        'cuda': config.cuda.__dict__,
        'panorama_splitter': config.panorama_splitter.__dict__,
        'image_preprocessor': config.image_preprocessor.__dict__,
        'image_registration': config.image_registration.__dict__,
        'change_detector': config.change_detector.__dict__,
        'result_mapper': config.result_mapper.__dict__,
        'visualization': config.visualization.__dict__,
        'output_root': config.output_root,
        'temp_dir': config.temp_dir,
        'log_level': config.log_level
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def load_config_from_file(filepath: str) -> SystemConfig:
    """从文件加载配置"""
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return SystemConfig(
        cuda=CUDAConfig(**config_dict.get('cuda', {})),
        panorama_splitter=PanoramaSplitterConfig(**config_dict.get('panorama_splitter', {})),
        image_preprocessor=ImagePreprocessorConfig(**config_dict.get('image_preprocessor', {})),
        image_registration=ImageRegistrationConfig(**config_dict.get('image_registration', {})),
        change_detector=ChangeDetectorConfig(**config_dict.get('change_detector', {})),
        result_mapper=ResultMapperConfig(**config_dict.get('result_mapper', {})),
        visualization=VisualizationConfig(**config_dict.get('visualization', {})),
        output_root=config_dict.get('output_root', 'results'),
        temp_dir=config_dict.get('temp_dir', 'temp'),
        log_level=config_dict.get('log_level', 'INFO')
    )
