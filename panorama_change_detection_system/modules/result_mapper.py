#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果映射与全景图还原模块
将变化检测结果映射回原始立方体面，然后重建全景图
支持CUDA加速
"""

import cv2
import numpy as np
import math
import logging
from typing import Dict, Tuple, Optional, Union, Any, List
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cuda_utils import CUDAUtils, ensure_cuda_available, cp, CUDA_AVAILABLE
from config import ResultMapperConfig, CUDAConfig


class ResultMapper:
    """结果映射与全景图还原器"""
    
    def __init__(self, config: ResultMapperConfig = None,
                 cuda_config: CUDAConfig = None):
        self.config = config or ResultMapperConfig()
        self.cuda_config = cuda_config or CUDAConfig()
        
        # 初始化CUDA工具
        self.cuda_utils = CUDAUtils(
            use_cuda=self.cuda_config.use_cuda,
            device_id=self.cuda_config.device_id
        )
        
        # 立方体面配置
        self.face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        
        logging.info(f"结果映射器初始化完成，CUDA: {'启用' if self.cuda_utils.use_cuda else '禁用'}")
    
    def map_detections_to_original_faces(self, 
                                       change_results: Dict[str, Dict[str, Any]],
                                       registration_info: Dict[str, Dict[str, Any]],
                                       original_faces: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        将检测结果映射回原始立方体面坐标
        
        Args:
            change_results: 变化检测结果
            registration_info: 配准信息（包含变换矩阵）
            original_faces: 原始立方体面
            
        Returns:
            映射到原始坐标的检测结果
        """
        logging.info("开始将检测结果映射回原始立方体面坐标")
        
        mapped_results = {}
        
        for face_name, change_result in change_results.items():
            if face_name in registration_info and face_name in original_faces:
                logging.debug(f"映射面 {face_name} 的检测结果")
                
                # 获取变换矩阵
                transform_matrix = registration_info[face_name].get('transform_matrix')
                
                if transform_matrix is not None and self.config.enable_inverse_transform:
                    # 计算逆变换矩阵
                    try:
                        inverse_matrix = cv2.invert(transform_matrix)[1]
                        
                        # 映射检测框
                        face_shape = original_faces[face_name].shape[:2]
                        mapped_detections = self._map_detections_with_inverse_transform(
                            change_result['detections'], inverse_matrix, face_shape
                        )
                    except Exception as e:
                        logging.warning(f"逆变换失败 {face_name}: {e}, 使用原始坐标")
                        mapped_detections = change_result['detections']
                else:
                    # 没有变换矩阵，直接使用原始坐标
                    mapped_detections = change_result['detections']
                
                # 创建映射后的可视化
                mapped_visualization = self._create_mapped_visualization(
                    original_faces[face_name], mapped_detections
                )
                
                mapped_results[face_name] = {
                    'detections': mapped_detections,
                    'visualization': mapped_visualization,
                    'original_detections': change_result['detections'],
                    'transform_applied': transform_matrix is not None and self.config.enable_inverse_transform
                }
            else:
                logging.warning(f"面 {face_name} 缺少配准信息或原始面数据")
        
        logging.info(f"检测结果映射完成，处理了 {len(mapped_results)} 个面")
        return mapped_results
    
    def _map_detections_with_inverse_transform(self, detections: List[Dict[str, Any]], 
                                             inverse_matrix: np.ndarray,
                                             face_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        使用逆变换矩阵映射检测结果
        
        Args:
            detections: 检测结果列表
            inverse_matrix: 逆变换矩阵
            face_shape: 立方体面的形状 (height, width)
            
        Returns:
            映射后的检测结果列表
        """
        mapped_detections = []
        
        for detection in detections:
            # 获取边界框四个角点
            x, y, w, h = detection['bbox']
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # 应用逆变换
            mapped_corners = cv2.perspectiveTransform(corners, inverse_matrix)
            mapped_corners = mapped_corners.reshape(-1, 2)
            
            # 计算映射后的边界框
            min_x = int(np.min(mapped_corners[:, 0]))
            max_x = int(np.max(mapped_corners[:, 0]))
            min_y = int(np.min(mapped_corners[:, 1]))
            max_y = int(np.max(mapped_corners[:, 1]))
            
            # 获取立方体面尺寸进行边界检查
            face_height, face_width = face_shape
            
            # 边界检查和裁剪
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(face_width, max_x)
            max_y = min(face_height, max_y)
            
            # 确保边界框有效
            if max_x <= min_x or max_y <= min_y:
                logging.warning(f"检测框映射后无效，跳过: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
                continue
            
            mapped_bbox = [
                min_x,
                min_y,
                max_x - min_x,
                max_y - min_y
            ]
            
            # 创建映射后的检测结果
            mapped_detection = detection.copy()
            mapped_detection['bbox'] = mapped_bbox
            mapped_detection['mapped_corners'] = mapped_corners.tolist()
            mapped_detection['center'] = [
                int(min_x + (max_x - min_x) // 2),
                int(min_y + (max_y - min_y) // 2)
            ]
            
            mapped_detections.append(mapped_detection)
        
        return mapped_detections
    
    def _create_mapped_visualization(self, original_face: np.ndarray, 
                                   mapped_detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        创建映射后的可视化图像
        
        Args:
            original_face: 原始立方体面
            mapped_detections: 映射后的检测结果
            
        Returns:
            可视化图像
        """
        vis_image = original_face.copy()
        overlay = vis_image.copy()
        
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        
        face_height, face_width = original_face.shape[:2]
        
        for i, detection in enumerate(mapped_detections):
            color = colors[i % len(colors)]
            confidence = detection['confidence']
            
            # 如果有映射的角点，绘制多边形；否则绘制矩形
            if 'mapped_corners' in detection:
                corners = np.array(detection['mapped_corners'], dtype=np.int32)
                
                # 边界检查：裁剪超出边界的角点
                corners[:, 0] = np.clip(corners[:, 0], 0, face_width - 1)
                corners[:, 1] = np.clip(corners[:, 1], 0, face_height - 1)
                
                # 绘制半透明填充
                cv2.fillPoly(overlay, [corners], color)
                
                # 绘制边框
                cv2.polylines(vis_image, [corners], True, color, 3)
            else:
                # 传统矩形绘制
                x, y, w, h = detection['bbox']
                
                # 边界检查和裁剪
                x = max(0, min(x, face_width - 1))
                y = max(0, min(y, face_height - 1))
                x2 = max(0, min(x + w, face_width))
                y2 = max(0, min(y + h, face_height))
                
                # 确保矩形有效
                if x2 > x and y2 > y:
                    cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)
                    cv2.rectangle(vis_image, (x, y), (x2, y2), color, 3)
            
            # 添加标签
            center = detection['center']
            
            # 边界检查中心点
            center_x = max(0, min(center[0], face_width - 1))
            center_y = max(0, min(center[1], face_height - 1))
            
            label = f'ID:{detection["id"]} ({confidence:.2f})'
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_bg_x1 = max(0, center_x - text_size[0] // 2 - 5)
            label_bg_y1 = max(0, center_y - text_size[1] // 2 - 5)
            label_bg_x2 = min(face_width, center_x + text_size[0] // 2 + 5)
            label_bg_y2 = min(face_height, center_y + text_size[1] // 2 + 5)
            
            # 确保标签背景有效
            if label_bg_x2 > label_bg_x1 and label_bg_y2 > label_bg_y1:
                cv2.rectangle(vis_image, 
                            (label_bg_x1, label_bg_y1), 
                            (label_bg_x2, label_bg_y2), 
                            (0, 0, 0), -1)
                cv2.rectangle(vis_image, 
                            (label_bg_x1, label_bg_y1), 
                            (label_bg_x2, label_bg_y2), 
                            color, 2)
            
                # 调整文本位置确保在边界内
                text_x = max(text_size[0] // 2, min(center_x, face_width - text_size[0] // 2))
                text_y = max(text_size[1] // 2, min(center_y + text_size[1] // 2, face_height - 5))
                
            cv2.putText(vis_image, label, 
                          (text_x - text_size[0] // 2, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 融合半透明效果
        alpha = 0.3
        cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0, vis_image)
        
        return vis_image
    
    def reconstruct_panorama_with_detections(self, 
                                           cube_faces: Dict[str, np.ndarray],
                                           mapped_results: Dict[str, Dict[str, Any]],
                                           output_width: int, output_height: int) -> np.ndarray:
        """
        重建包含检测结果的全景图
        
        Args:
            cube_faces: 立方体面字典
            mapped_results: 映射后的检测结果
            output_width: 输出全景图宽度
            output_height: 输出全景图高度
            
        Returns:
            重建的全景图
        """
        logging.info(f"开始重建全景图，尺寸: {output_width}x{output_height}")
        
        # 使用原始立方体面（不包含检测框和标签）
        faces_for_reconstruction = {}
        for face_name in self.face_names:
            if face_name in cube_faces:
                # 始终使用原始图像，不使用包含ID和置信度的可视化图像
                faces_for_reconstruction[face_name] = cube_faces[face_name]
        
        # 重建全景图
        if self.config.reconstruction_method == "improved":
            panorama = self._reconstruct_panorama_improved(
                faces_for_reconstruction, output_width, output_height
            )
        else:
            panorama = self._reconstruct_panorama_basic(
                faces_for_reconstruction, output_width, output_height
            )
        
        logging.info("全景图重建完成")
        return panorama
    
    def _reconstruct_panorama_improved(self, faces: Dict[str, np.ndarray], 
                                     output_width: int, output_height: int) -> np.ndarray:
        """改进的全景图重建（CUDA加速版本）"""
        cube_size = list(faces.values())[0].shape[0]
        
        if self.cuda_utils.use_cuda:
            return self._reconstruct_panorama_cuda(faces, output_width, output_height, cube_size)
        else:
            return self._reconstruct_panorama_cpu(faces, output_width, output_height, cube_size)
    
    @ensure_cuda_available
    def _reconstruct_panorama_cuda(self, faces: Dict[str, np.ndarray], 
                                  output_width: int, output_height: int, cube_size: int) -> np.ndarray:
        """CUDA加速的全景图重建"""
        # 将立方体面传输到GPU
        faces_gpu = {}
        for face_name, face_img in faces.items():
            faces_gpu[face_name] = self.cuda_utils.to_gpu(face_img)
        
        # 创建输出全景图
        panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 分块处理以节省GPU内存
        chunk_height = min(512, output_height)
        
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
            panorama_chunk_gpu = self._cuda_xyz_to_panorama_batch(
                x, y, z, faces_gpu, cube_size, panorama_chunk_gpu
            )
            
            # 将结果复制到CPU
            panorama[start_row:end_row, :, :] = self.cuda_utils.to_cpu(panorama_chunk_gpu)
            
            # 清理GPU内存
            del panorama_chunk_gpu, v_coords, u_coords, theta, phi, x, y, z
            self.cuda_utils.cleanup_memory()
        
        # 清理GPU立方体面
        for face_name in list(faces_gpu.keys()):
            del faces_gpu[face_name]
        self.cuda_utils.cleanup_memory()
        
        return panorama
    
    def _cuda_xyz_to_panorama_batch(self, x, y, z, faces_gpu, cube_size, panorama_gpu):
        """CUDA批量处理3D坐标到全景图的映射"""
        abs_x, abs_y, abs_z = cp.abs(x), cp.abs(y), cp.abs(z)
        
        # 为每个立方体面创建掩码和坐标
        face_mappings = [
            # (face_name, condition, u_formula, v_formula)
            ('front', (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0),
             lambda: (x / z + 1) * 0.5 * cube_size,
             lambda: (-y / z + 1) * 0.5 * cube_size),
            ('back', (abs_z >= abs_x) & (abs_z >= abs_y) & (z <= 0),
             lambda: (-x / (-z) + 1) * 0.5 * cube_size,
             lambda: (-y / (-z) + 1) * 0.5 * cube_size),
            ('right', (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0),
             lambda: (-z / x + 1) * 0.5 * cube_size,
             lambda: (-y / x + 1) * 0.5 * cube_size),
            ('left', (abs_x >= abs_y) & (abs_x >= abs_z) & (x <= 0),
             lambda: (z / (-x) + 1) * 0.5 * cube_size,
             lambda: (-y / (-x) + 1) * 0.5 * cube_size),
            ('top', (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0),
             lambda: (x / y + 1) * 0.5 * cube_size,
             lambda: (z / y + 1) * 0.5 * cube_size),
            ('bottom', (abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0),
             lambda: (x / (-y) + 1) * 0.5 * cube_size,
             lambda: (z / (-y) + 1) * 0.5 * cube_size),
        ]
        
        # 为每个面执行映射
        for face_name, condition, u_func, v_func in face_mappings:
            if face_name in faces_gpu:
                mask = condition
                if cp.any(mask):
                    face_u = u_func()
                    face_v = v_func()
                    
                    # 边界检查
                    valid_coords = (face_u >= 0) & (face_u < cube_size) & (face_v >= 0) & (face_v < cube_size)
                    combined_mask = mask & valid_coords
                    
                    if cp.any(combined_mask):
                        # GPU双线性插值
                        sampled_pixels = self._cuda_face_bilinear_sample(
                            faces_gpu[face_name], face_u, face_v, combined_mask
                        )
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
    
    def _reconstruct_panorama_cpu(self, faces: Dict[str, np.ndarray], 
                                 output_width: int, output_height: int, cube_size: int) -> np.ndarray:
        """CPU版本的全景图重建"""
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
                
                if (face_name and face_name in faces and 
                    0 <= face_u < cube_size and 0 <= face_v < cube_size):
                    
                    # 使用双线性插值
                    pixel_value = self._bilinear_interpolate(faces[face_name], face_u, face_v)
                    panorama[v, u] = pixel_value
        
        return panorama
    
    def _reconstruct_panorama_basic(self, faces: Dict[str, np.ndarray], 
                                   output_width: int, output_height: int) -> np.ndarray:
        """基础的全景图重建"""
        return self._reconstruct_panorama_cpu(faces, output_width, output_height, 
                                            list(faces.values())[0].shape[0])
    
    def _xyz_to_cube_face(self, x: float, y: float, z: float, cube_size: int):
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
    
    def _bilinear_interpolate(self, img: np.ndarray, x: float, y: float):
        """双线性插值"""
        h, w = img.shape[:2]
        
        x = max(0.0, min(float(x), w - 1.0))
        y = max(0.0, min(float(y), h - 1.0))
        
        x1, y1 = int(math.floor(x)), int(math.floor(y))
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
        
        dx = x - x1
        dy = y - y1
        
        pixel = ((1.0 - dx) * (1.0 - dy) * img[y1, x1] +
                dx * (1.0 - dy) * img[y1, x2] +
                (1.0 - dx) * dy * img[y2, x1] +
                dx * dy * img[y2, x2])
        
        return pixel.astype(np.uint8)
    
    def create_detection_summary_on_panorama(self, panorama: np.ndarray, 
                                           mapped_results: Dict[str, Dict[str, Any]],
                                           cube_size: int) -> np.ndarray:
        """
        在全景图上创建检测结果摘要标注（简洁的标记点代替复杂的检测框）
        
        Args:
            panorama: 重建的全景图
            mapped_results: 映射后的检测结果
            cube_size: 立方体尺寸
            
        Returns:
            带有检测标记的全景图
        """
        annotated_panorama = panorama.copy()
        panorama_height, panorama_width = panorama.shape[:2]
        
        # 计算总检测数量和记录详细信息
        total_detections = sum(len(result['detections']) for result in mapped_results.values())
        
        # 详细日志记录
        logging.info("=" * 60)
        logging.info("全景图检测结果摘要:")
        logging.info(f"  全景图尺寸: {panorama_width} x {panorama_height}")
        logging.info(f"  立方体面尺寸: {cube_size} x {cube_size}")
        logging.info(f"  总检测数量: {total_detections}")
        
        for face_name, result in mapped_results.items():
            if result['detections']:
                logging.info(f"  {face_name} 面: {len(result['detections'])} 个检测")
                for i, detection in enumerate(result['detections']):
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    area = detection.get('area', 0)
                    logging.info(f"    检测{i+1}: bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}], "
                               f"confidence={confidence:.3f}, area={area:.1f}")
        
        logging.info("=" * 60)
        
        if total_detections == 0:
            # 无变化时返回干净的全景图（不添加任何文字）
            return annotated_panorama
        
        # 颜色配置
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        
        # 为每个检测在全景图上绘制简洁的标记点
        detection_id = 0
        for face_name, result in mapped_results.items():
            face_index = self.face_names.index(face_name)
            
            for detection in result['detections']:
                color = colors[detection_id % len(colors)]
                
                # 将检测框中心点映射到全景图坐标
                center_point = self._map_detection_center_to_panorama(
                    detection['bbox'], face_index, cube_size, panorama_width, panorama_height
                )
                
                if center_point:
                    # 添加面索引和立方体尺寸到检测信息中
                    detection_with_meta = detection.copy()
                    detection_with_meta['face_index'] = face_index
                    detection_with_meta['cube_size'] = cube_size
                    
                    # 绘制真实比例的检测框
                    self._draw_detection_marker(
                        annotated_panorama, center_point, detection_with_meta, color, detection_id + 1
                    )
                
                detection_id += 1
        
        # 添加文字摘要到右上角
        self._add_summary_text(annotated_panorama, mapped_results, total_detections)
        
        return annotated_panorama
    
    def _map_detection_center_to_panorama(self, bbox: List[int], face_index: int, 
                                        cube_size: int, panorama_width: int, 
                                        panorama_height: int) -> Optional[Tuple[int, int]]:
        """
        将检测框中心点从立方体面坐标映射到全景图坐标
        
        Args:
            bbox: [x, y, w, h] 在立方体面上的坐标
            face_index: 面索引
            cube_size: 立方体面尺寸
            panorama_width, panorama_height: 全景图尺寸
            
        Returns:
            全景图上的中心点坐标 (x, y)
        """
        x, y, w, h = bbox
        
        # 计算检测框中心点
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        
        # 标准化立方体面坐标到[-1, 1]
        norm_x = (2.0 * center_x / cube_size) - 1.0
        norm_y = (2.0 * center_y / cube_size) - 1.0
        
        # 获取3D坐标
        x3d, y3d, z3d = self._get_face_3d_coords_cpu(face_index, norm_x, norm_y)
        
        # 转换为球面坐标
        r = math.sqrt(x3d * x3d + y3d * y3d + z3d * z3d)
        theta = math.atan2(x3d, z3d)
        phi = math.acos(y3d / r) if r > 0 else 0
        
        # 转换为全景图坐标
        u = (theta + math.pi) / (2 * math.pi) * panorama_width
        v = phi / math.pi * panorama_height
        
        # 边界检查
        u = max(0, min(u, panorama_width - 1))
        v = max(0, min(v, panorama_height - 1))
        
        return (int(u), int(v))
    
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
    
    def _draw_detection_marker(self, panorama: np.ndarray, 
                             center_point: Tuple[int, int], 
                             detection: Dict[str, Any], color: Tuple[int, int, int], 
                             detection_id: int):
        """在全景图上绘制真实比例的检测框（无文字信息）"""
        center_x, center_y = center_point
        
        # 获取检测框的实际尺寸和立方体面信息
        bbox = detection.get('bbox', [0, 0, 50, 50])
        face_index = detection.get('face_index', 0)
        cube_size = detection.get('cube_size', 1024)
        
        # 计算真实比例的检测框
        real_bbox = self._calculate_real_scale_bbox(
            bbox, face_index, cube_size, panorama.shape[1], panorama.shape[0]
        )
        
        if real_bbox is None:
            # 如果映射失败，回退到中心点标记
            cv2.circle(panorama, (center_x, center_y), 8, color, -1)
            cv2.circle(panorama, (center_x, center_y), 10, (255, 255, 255), 2)
            return
        
        x1, y1, x2, y2 = real_bbox
        
        # 边界检查
        panorama_height, panorama_width = panorama.shape[:2]
        x1 = max(0, min(x1, panorama_width - 1))
        y1 = max(0, min(y1, panorama_height - 1))
        x2 = max(0, min(x2, panorama_width - 1))
        y2 = max(0, min(y2, panorama_height - 1))
        
        # 确保框有效（避免退化的矩形）
        if x2 <= x1 or y2 <= y1:
            # 回退到中心点标记
            cv2.circle(panorama, (center_x, center_y), 8, color, -1)
            cv2.circle(panorama, (center_x, center_y), 10, (255, 255, 255), 2)
            return
        
        # 绘制检测框
        thickness = 3
        cv2.rectangle(panorama, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制白色外框增强可见性
        cv2.rectangle(panorama, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)
        
        # 绘制中心点标记
        cv2.circle(panorama, (center_x, center_y), 3, color, -1)
        cv2.circle(panorama, (center_x, center_y), 4, (255, 255, 255), 1)
    
    def _calculate_real_scale_bbox(self, bbox: List[int], face_index: int, 
                                  cube_size: int, panorama_width: int, 
                                  panorama_height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        计算检测框在全景图上的真实比例边界框
        
        Args:
            bbox: [x, y, w, h] 在立方体面上的坐标
            face_index: 面索引
            cube_size: 立方体面尺寸
            panorama_width, panorama_height: 全景图尺寸
            
        Returns:
            全景图上的边界框坐标 [x1, y1, x2, y2] 或 None（如果映射失败）
        """
        x, y, w, h = bbox
        
        # 定义检测框的四个角点
        corners = [
            (x, y),           # 左上角
            (x + w, y),       # 右上角
            (x, y + h),       # 左下角
            (x + w, y + h)    # 右下角
        ]
        
        # 映射所有角点到全景图
        mapped_points = []
        for corner_x, corner_y in corners:
            # 标准化立方体面坐标到[-1, 1]
            norm_x = (2.0 * corner_x / cube_size) - 1.0
            norm_y = (2.0 * corner_y / cube_size) - 1.0
            
            # 获取3D坐标
            x3d, y3d, z3d = self._get_face_3d_coords_cpu(face_index, norm_x, norm_y)
            
            # 转换为球面坐标
            r = math.sqrt(x3d * x3d + y3d * y3d + z3d * z3d)
            if r == 0:
                continue
                
            theta = math.atan2(x3d, z3d)
            phi = math.acos(y3d / r)
            
            # 转换为全景图坐标
            u = (theta + math.pi) / (2 * math.pi) * panorama_width
            v = phi / math.pi * panorama_height
            
            # 边界检查
            u = max(0, min(u, panorama_width - 1))
            v = max(0, min(v, panorama_height - 1))
            
            mapped_points.append((int(u), int(v)))
        
        if len(mapped_points) < 4:
            return None
        
        # 计算映射后的边界框
        u_coords = [p[0] for p in mapped_points]
        v_coords = [p[1] for p in mapped_points]
        
        # 处理全景图边界换行问题（经度0/360度边界）
        u_min, u_max = min(u_coords), max(u_coords)
        v_min, v_max = min(v_coords), max(v_coords)
        
        # 检查是否跨越全景图边界（经度换行）
        if u_max - u_min > panorama_width * 0.5:
            # 可能跨越了0/360度边界，需要特殊处理
            # 这种情况下使用原始的简化映射
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 计算一个合理的缩放比例
            # 基于立方体面在全景图中的相对大小
            scale_factor = panorama_width / (6 * cube_size)  # 6个面平均分布
            
            box_w = max(10, int(w * scale_factor))
            box_h = max(10, int(h * scale_factor))
            
            center_u, center_v = self._map_point_to_panorama(
                center_x, center_y, face_index, cube_size, panorama_width, panorama_height
            )
            
            if center_u is None or center_v is None:
                return None
                
            x1 = center_u - box_w // 2
            y1 = center_v - box_h // 2
            x2 = center_u + box_w // 2
            y2 = center_v + box_h // 2
            
            return (x1, y1, x2, y2)
        
        return (u_min, v_min, u_max, v_max)
    
    def _map_point_to_panorama(self, x: float, y: float, face_index: int, 
                              cube_size: int, panorama_width: int, 
                              panorama_height: int) -> Tuple[Optional[int], Optional[int]]:
        """映射单个点到全景图坐标"""
        # 标准化立方体面坐标到[-1, 1]
        norm_x = (2.0 * x / cube_size) - 1.0
        norm_y = (2.0 * y / cube_size) - 1.0
        
        # 获取3D坐标
        x3d, y3d, z3d = self._get_face_3d_coords_cpu(face_index, norm_x, norm_y)
        
        # 转换为球面坐标
        r = math.sqrt(x3d * x3d + y3d * y3d + z3d * z3d)
        if r == 0:
            return None, None
            
        theta = math.atan2(x3d, z3d)
        phi = math.acos(y3d / r)
        
        # 转换为全景图坐标
        u = (theta + math.pi) / (2 * math.pi) * panorama_width
        v = phi / math.pi * panorama_height
        
        # 边界检查
        u = max(0, min(u, panorama_width - 1))
        v = max(0, min(v, panorama_height - 1))
        
        return int(u), int(v)
    
    def _add_summary_text(self, panorama: np.ndarray, 
                         mapped_results: Dict[str, Dict[str, Any]], 
                         total_detections: int):
        """添加摘要信息（已禁用文字显示）"""
        # 不绘制任何文字信息，保持全景图干净
        pass
    
    def get_mapping_statistics(self, mapped_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取映射统计信息
        
        Args:
            mapped_results: 映射后的检测结果
            
        Returns:
            统计信息字典
        """
        total_detections = 0
        total_areas = []
        confidence_scores = []
        faces_with_detections = 0
        transform_applied_count = 0
        
        face_stats = {}
        
        for face_name, result in mapped_results.items():
            detections = result['detections']
            num_detections = len(detections)
            
            total_detections += num_detections
            
            if num_detections > 0:
                faces_with_detections += 1
                
                areas = [det['area'] for det in detections]
                confidences = [det['confidence'] for det in detections]
                
                total_areas.extend(areas)
                confidence_scores.extend(confidences)
                
                face_stats[face_name] = {
                    'detection_count': num_detections,
                    'avg_area': np.mean(areas),
                    'avg_confidence': np.mean(confidences),
                    'transform_applied': result.get('transform_applied', False)
                }
            
            if result.get('transform_applied', False):
                transform_applied_count += 1
        
        return {
            'total_faces': len(mapped_results),
            'faces_with_detections': faces_with_detections,
            'faces_without_detections': len(mapped_results) - faces_with_detections,
            'total_detections': total_detections,
            'faces_with_transform_applied': transform_applied_count,
            'overall_avg_area': np.mean(total_areas) if total_areas else 0,
            'overall_avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'face_statistics': face_stats,
            'detection_distribution': {
                face_name: len(result['detections']) 
                for face_name, result in mapped_results.items()
            }
        }
    
    def save_cube_faces_with_detections(self, mapped_results: Dict[str, Dict[str, Any]], 
                                       output_dir: str) -> Dict[str, str]:
        """
        保存包含检测结果的立方体面图像
        
        Args:
            mapped_results: 映射后的检测结果
            output_dir: 输出目录
            
        Returns:
            保存的文件路径字典
        """
        import os
        faces_dir = os.path.join(output_dir, 'cube_faces_with_mapped_detections')
        os.makedirs(faces_dir, exist_ok=True)
        
        saved_paths = {}
        
        for face_name, result in mapped_results.items():
            filename = f'{face_name}_with_detections.jpg'
            face_path = os.path.join(faces_dir, filename)
            
            success = cv2.imwrite(face_path, result['visualization'])
            if success:
                saved_paths[face_name] = face_path
                logging.debug(f"保存检测结果立方体面: {face_path}")
            else:
                logging.error(f"保存检测结果立方体面失败: {face_path}")
        
        logging.info(f"检测结果立方体面保存完成，目录: {faces_dir}")
        return saved_paths
    
    def cleanup_memory(self):
        """清理内存"""
        if self.cuda_utils.use_cuda:
            self.cuda_utils.cleanup_memory()
