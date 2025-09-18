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
                        mapped_detections = self._map_detections_with_inverse_transform(
                            change_result['detections'], inverse_matrix
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
                                             inverse_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用逆变换矩阵映射检测结果
        
        Args:
            detections: 检测结果列表
            inverse_matrix: 逆变换矩阵
            
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
            
            mapped_bbox = [
                max(0, min_x),
                max(0, min_y),
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
        
        for i, detection in enumerate(mapped_detections):
            color = colors[i % len(colors)]
            confidence = detection['confidence']
            
            # 如果有映射的角点，绘制多边形；否则绘制矩形
            if 'mapped_corners' in detection:
                corners = np.array(detection['mapped_corners'], dtype=np.int32)
                
                # 绘制半透明填充
                cv2.fillPoly(overlay, [corners], color)
                
                # 绘制边框
                cv2.polylines(vis_image, [corners], True, color, 3)
            else:
                # 传统矩形绘制
                x, y, w, h = detection['bbox']
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
            
            # 添加标签
            center = detection['center']
            label = f'ID:{detection["id"]} ({confidence:.2f})'
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_bg_x1 = center[0] - text_size[0] // 2 - 5
            label_bg_y1 = center[1] - text_size[1] // 2 - 5
            label_bg_x2 = center[0] + text_size[0] // 2 + 5
            label_bg_y2 = center[1] + text_size[1] // 2 + 5
            
            cv2.rectangle(vis_image, 
                        (label_bg_x1, label_bg_y1), 
                        (label_bg_x2, label_bg_y2), 
                        (0, 0, 0), -1)
            cv2.rectangle(vis_image, 
                        (label_bg_x1, label_bg_y1), 
                        (label_bg_x2, label_bg_y2), 
                        color, 2)
            
            cv2.putText(vis_image, label, 
                      (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2), 
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
        
        # 创建包含检测框的立方体面
        faces_with_detections = {}
        for face_name in self.face_names:
            if face_name in cube_faces:
                if face_name in mapped_results:
                    # 使用包含检测框的可视化图像
                    faces_with_detections[face_name] = mapped_results[face_name]['visualization']
                else:
                    # 使用原始图像
                    faces_with_detections[face_name] = cube_faces[face_name]
        
        # 重建全景图
        if self.config.reconstruction_method == "improved":
            panorama = self._reconstruct_panorama_improved(
                faces_with_detections, output_width, output_height
            )
        else:
            panorama = self._reconstruct_panorama_basic(
                faces_with_detections, output_width, output_height
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
        在全景图上创建检测结果摘要标注
        
        Args:
            panorama: 重建的全景图
            mapped_results: 映射后的检测结果
            cube_size: 立方体尺寸
            
        Returns:
            带有摘要标注的全景图
        """
        annotated_panorama = panorama.copy()
        
        # 计算总检测数量
        total_detections = sum(len(result['detections']) for result in mapped_results.values())
        
        if total_detections == 0:
            # 添加"无变化"标注
            cv2.putText(annotated_panorama, "No Changes Detected", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            return annotated_panorama
        
        # 在全景图上标注检测摘要
        y_offset = 50
        line_height = 40
        
        # 总体信息
        cv2.putText(annotated_panorama, f"Total Changes: {total_detections}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        y_offset += line_height
        
        # 各面的检测数量
        for face_name, result in mapped_results.items():
            if result['detections']:
                count = len(result['detections'])
                face_desc = {'front': 'Front', 'back': 'Back', 'left': 'Left', 
                           'right': 'Right', 'top': 'Top', 'bottom': 'Bottom'}.get(face_name, face_name)
                
                cv2.putText(annotated_panorama, f"{face_desc}: {count}", 
                           (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += int(line_height * 0.8)
        
        return annotated_panorama
    
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
