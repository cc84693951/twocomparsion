#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变化区域检测模块
进行图像差分计算、阈值分割、形态学操作、轮廓检测和边界框提取
支持CUDA加速
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cuda_utils import CUDAUtils, ensure_cuda_available, cp, CUDA_AVAILABLE
from config import ChangeDetectorConfig, CUDAConfig


class ChangeDetector:
    """变化区域检测器"""
    
    def __init__(self, config: ChangeDetectorConfig = None,
                 cuda_config: CUDAConfig = None):
        self.config = config or ChangeDetectorConfig()
        self.cuda_config = cuda_config or CUDAConfig()
        
        # 初始化CUDA工具
        self.cuda_utils = CUDAUtils(
            use_cuda=self.cuda_config.use_cuda,
            device_id=self.cuda_config.device_id
        )
        
        logging.info(f"变化区域检测器初始化完成，CUDA: {'启用' if self.cuda_utils.use_cuda else '禁用'}")
    
    def detect_changes_in_faces(self, faces1: Dict[str, np.ndarray], 
                               faces2_aligned: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        批量检测立方体面中的变化区域
        
        Args:
            faces1: 第一组立方体面（参考图像）
            faces2_aligned: 配准后的第二组立方体面
            
        Returns:
            每个面的变化检测结果字典
        """
        logging.info(f"开始批量检测 {len(faces1)} 个立方体面的变化")
        
        change_results = {}
        
        for face_name in faces1.keys():
            if face_name in faces2_aligned:
                logging.debug(f"检测立方体面变化: {face_name}")
                
                # 检测单个面的变化
                change_result = self.detect_changes_single_pair(
                    faces1[face_name], 
                    faces2_aligned[face_name],
                    face_name
                )
                
                change_results[face_name] = change_result
            else:
                logging.warning(f"立方体面 {face_name} 在配准后的第二组中不存在")
        
        total_detections = sum(len(result['detections']) for result in change_results.values())
        logging.info(f"立方体面变化检测完成，总共检测到 {total_detections} 个变化区域")
        
        return change_results
    
    def detect_changes_single_pair(self, img1: np.ndarray, img2: np.ndarray, 
                                  face_name: str = "unknown") -> Dict[str, Any]:
        """
        检测单对图像中的变化区域（改进版：边缘排除+置信度过滤）
        
        Args:
            img1: 参考图像
            img2: 比较图像
            face_name: 面的名称（用于日志）
            
        Returns:
            变化检测结果
        """
        # 1. 计算图像差分
        diff_image = self.compute_image_difference(img1, img2)
        
        # 2. 创建边缘排除掩码（排除立方体面边缘区域）
        edge_mask = self.create_edge_exclusion_mask(img1.shape[:2])
        
        # 3. 阈值分割
        binary_image, threshold_used = self.apply_threshold(diff_image)
        
        # 4. 应用边缘掩码，排除边缘区域
        binary_image = binary_image & edge_mask
        
        # 5. 形态学操作
        processed_binary = self.apply_morphological_operations(binary_image)
        
        # 6. 轮廓检测和过滤
        contours = self.detect_and_filter_contours(processed_binary)
        
        # 7. 提取边界框（使用改进的置信度计算）
        detections = self.extract_bounding_boxes_improved(contours, processed_binary, diff_image, edge_mask)
        
        # 8. 应用置信度过滤（0.6以上）
        original_count = len(detections)
        detections = self.filter_detections_by_confidence(detections, min_confidence=0.6)
        filtered_count = len(detections)
        
        # 9. 创建可视化结果
        detection_image = self.create_detection_visualization(img2, detections)
        
        # 10. 计算统计信息
        stats = self.compute_detection_statistics(diff_image, binary_image, detections)
        
        result = {
            'face_name': face_name,
            'difference_image': diff_image,
            'binary_image': binary_image,
            'processed_binary': processed_binary,
            'detection_result': detection_image,
            'detections': detections,
            'detection_stats': stats,
            'threshold_used': threshold_used,
            'num_contours_found': len(contours),
            'num_detections': len(detections),
            'original_detections': original_count,
            'filtered_detections': filtered_count,
            'edge_mask': edge_mask
        }
        
        logging.debug(f"{face_name}: 检测到 {original_count} 个区域，过滤后 {filtered_count} 个（置信度≥0.6），阈值: {threshold_used}")
        
        return result
    
    def compute_image_difference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        计算图像差分
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            差分图像
        """
        if self.config.diff_method == "absdiff":
            return self._compute_absdiff(img1, img2)
        else:
            # 可以扩展其他差分方法
            return self._compute_absdiff(img1, img2)
    
    def _compute_absdiff(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """计算绝对差分"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 确保图像尺寸一致
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        if self.cuda_utils.use_cuda:
            return self._compute_absdiff_cuda(gray1, gray2)
        else:
            return cv2.absdiff(gray1, gray2)
    
    @ensure_cuda_available
    def _compute_absdiff_cuda(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """CUDA版本的绝对差分"""
        gray1_gpu = self.cuda_utils.to_gpu(gray1)
        gray2_gpu = self.cuda_utils.to_gpu(gray2)
        
        diff_gpu = cp.abs(gray1_gpu.astype(cp.int16) - gray2_gpu.astype(cp.int16))
        diff_gpu = cp.clip(diff_gpu, 0, 255).astype(cp.uint8)
        
        return self.cuda_utils.to_cpu(diff_gpu)
    
    def apply_threshold(self, diff_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        应用阈值分割
        
        Args:
            diff_image: 差分图像
            
        Returns:
            (二值化图像, 使用的阈值)
        """
        if self.config.threshold_method == "otsu":
            return self._apply_otsu_threshold(diff_image)
        elif self.config.threshold_method == "adaptive":
            return self._apply_adaptive_threshold(diff_image)
        else:  # fixed
            return self._apply_fixed_threshold(diff_image)
    
    def _apply_otsu_threshold(self, diff_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Otsu阈值分割"""
        if self.cuda_utils.use_cuda:
            binary, threshold = self.cuda_utils.threshold_cuda(diff_image, 0, "otsu")
        else:
            threshold, binary = cv2.threshold(diff_image, 0, 255, 
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary, threshold
    
    def _apply_adaptive_threshold(self, diff_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """自适应阈值分割"""
        # 使用简单的统计方法计算自适应阈值
        mean_diff = np.mean(diff_image)
        std_diff = np.std(diff_image)
        threshold = mean_diff + 2 * std_diff
        threshold = max(20, min(threshold, 100))  # 限制阈值范围
        
        if self.cuda_utils.use_cuda:
            binary, _ = self.cuda_utils.threshold_cuda(diff_image, threshold, "binary")
        else:
            _, binary = cv2.threshold(diff_image, threshold, 255, cv2.THRESH_BINARY)
        
        return binary, threshold
    
    def _apply_fixed_threshold(self, diff_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """固定阈值分割"""
        threshold = self.config.fixed_threshold
        
        if self.cuda_utils.use_cuda:
            binary, _ = self.cuda_utils.threshold_cuda(diff_image, threshold, "binary")
        else:
            _, binary = cv2.threshold(diff_image, threshold, 255, cv2.THRESH_BINARY)
        
        return binary, threshold
    
    def apply_morphological_operations(self, binary_image: np.ndarray) -> np.ndarray:
        """
        应用形态学操作
        
        Args:
            binary_image: 二值化图像
            
        Returns:
            形态学处理后的图像
        """
        processed = binary_image.copy()
        
        for operation in self.config.morphology_operations:
            if operation == "close":
                processed = self._apply_morphology_close(processed)
            elif operation == "open":
                processed = self._apply_morphology_open(processed)
            elif operation == "erode":
                processed = self._apply_morphology_erode(processed)
            elif operation == "dilate":
                processed = self._apply_morphology_dilate(processed)
        
        return processed
    
    def _apply_morphology_close(self, image: np.ndarray) -> np.ndarray:
        """形态学闭运算"""
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.morphology_cuda(
                image, "close", self.config.close_kernel_size
            )
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.close_kernel_size, self.config.close_kernel_size)
            )
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    def _apply_morphology_open(self, image: np.ndarray) -> np.ndarray:
        """形态学开运算"""
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.morphology_cuda(
                image, "open", self.config.open_kernel_size
            )
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.open_kernel_size, self.config.open_kernel_size)
            )
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def _apply_morphology_erode(self, image: np.ndarray) -> np.ndarray:
        """形态学腐蚀"""
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.morphology_cuda(image, "erode", 3)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.erode(image, kernel)
    
    def _apply_morphology_dilate(self, image: np.ndarray) -> np.ndarray:
        """形态学膨胀"""
        if self.cuda_utils.use_cuda:
            return self.cuda_utils.morphology_cuda(image, "dilate", 3)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.dilate(image, kernel)
    
    def detect_and_filter_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        检测和过滤轮廓
        
        Args:
            binary_image: 二值化图像
            
        Returns:
            过滤后的轮廓列表
        """
        # 检测轮廓
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 过滤轮廓
        filtered_contours = []
        
        for contour in contours:
            if self._is_valid_contour(contour):
                filtered_contours.append(contour)
        
        logging.debug(f"轮廓检测: 总数={len(contours)}, 过滤后={len(filtered_contours)}")
        
        return filtered_contours
    
    def _is_valid_contour(self, contour: np.ndarray) -> bool:
        """
        判断轮廓是否有效
        
        Args:
            contour: 轮廓
            
        Returns:
            是否有效
        """
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 面积过滤
        if area < self.config.min_contour_area or area > self.config.max_contour_area:
            return False
        
        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 长宽比过滤
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
            return False
        
        # 轮廓密集度过滤（extent = 轮廓面积 / 边界框面积）
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        if extent < self.config.min_extent:
            return False
        
        return True
    
    def extract_bounding_boxes(self, contours: List[np.ndarray], 
                             binary_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        从轮廓提取边界框
        
        Args:
            contours: 轮廓列表
            binary_image: 二值化图像
            
        Returns:
            边界框信息列表
        """
        detections = []
        
        for i, contour in enumerate(contours):
            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 添加填充
            padding = self.config.bbox_padding
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(binary_image.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(binary_image.shape[0] - y_padded, h + 2 * padding)
            
            # 计算轮廓属性
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 计算置信度（基于面积和轮廓完整性）
            bbox_area = w * h
            extent = area / bbox_area if bbox_area > 0 else 0
            solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
            
            # 综合置信度计算
            confidence = min(1.0, (extent * 0.4 + solidity * 0.3 + 
                                 min(1.0, area / self.config.min_contour_area) * 0.3))
            
            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2
            
            detection = {
                'id': i + 1,
                'bbox': [int(x_padded), int(y_padded), int(w_padded), int(h_padded)],
                'bbox_original': [int(x), int(y), int(w), int(h)],
                'area': float(area),
                'perimeter': float(perimeter),
                'extent': float(extent),
                'solidity': float(solidity),
                'confidence': float(confidence),
                'center': [int(center_x), int(center_y)],
                'contour': contour
            }
            
            detections.append(detection)
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 重新分配ID
        for i, detection in enumerate(detections):
            detection['id'] = i + 1
        
        return detections
    
    def create_detection_visualization(self, image: np.ndarray, 
                                     detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        创建检测结果可视化
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            可视化图像
        """
        vis_image = image.copy()
        overlay = vis_image.copy()
        
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            color = colors[i % len(colors)]
            confidence = detection['confidence']
            
            # 绘制半透明填充
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            
            # 绘制边框
            thickness = 3 if confidence > 0.7 else 2
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
            
            # 绘制角点标记
            corner_size = 10
            cv2.line(vis_image, (x, y), (x + corner_size, y), color, 4)
            cv2.line(vis_image, (x, y), (x, y + corner_size), color, 4)
            cv2.line(vis_image, (x + w, y), (x + w - corner_size, y), color, 4)
            cv2.line(vis_image, (x + w, y), (x + w, y + corner_size), color, 4)
            cv2.line(vis_image, (x, y + h), (x + corner_size, y + h), color, 4)
            cv2.line(vis_image, (x, y + h), (x, y + h - corner_size), color, 4)
            cv2.line(vis_image, (x + w, y + h), (x + w - corner_size, y + h), color, 4)
            cv2.line(vis_image, (x + w, y + h), (x + w, y + h - corner_size), color, 4)
            
            # 添加标签
            label = f'ID:{detection["id"]} ({confidence:.2f})'
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 标签背景
            label_bg_x1 = max(0, x - 5)
            label_bg_y1 = max(0, y - text_size[1] - 15)
            label_bg_x2 = min(vis_image.shape[1], x + text_size[0] + 5)
            label_bg_y2 = y - 5
            
            cv2.rectangle(vis_image, 
                        (label_bg_x1, label_bg_y1), 
                        (label_bg_x2, label_bg_y2), 
                        (0, 0, 0), -1)
            cv2.rectangle(vis_image, 
                        (label_bg_x1, label_bg_y1), 
                        (label_bg_x2, label_bg_y2), 
                        color, 2)
            
            # 绘制标签文字
            cv2.putText(vis_image, label, (x, y - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 融合半透明效果
        alpha = 0.3
        cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0, vis_image)
        
        return vis_image
    
    def compute_detection_statistics(self, diff_image: np.ndarray, 
                                   binary_image: np.ndarray,
                                   detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算检测统计信息
        
        Args:
            diff_image: 差分图像
            binary_image: 二值化图像
            detections: 检测结果列表
            
        Returns:
            统计信息字典
        """
        if not detections:
            return {
                'num_detections': 0,
                'total_changed_pixels': int(np.sum(binary_image > 0)),
                'change_ratio': float(np.sum(binary_image > 0) / binary_image.size),
                'avg_area': 0.0,
                'max_area': 0.0,
                'min_area': 0.0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0,
                'med_confidence_count': 0,
                'low_confidence_count': 0
            }
        
        areas = [det['area'] for det in detections]
        confidences = [det['confidence'] for det in detections]
        
        # 按置信度分类
        high_conf = len([c for c in confidences if c > 0.8])
        med_conf = len([c for c in confidences if 0.6 <= c <= 0.8])
        low_conf = len([c for c in confidences if c < 0.6])
        
        stats = {
            'num_detections': len(detections),
            'total_changed_pixels': int(np.sum(binary_image > 0)),
            'change_ratio': float(np.sum(binary_image > 0) / binary_image.size),
            'avg_area': float(np.mean(areas)),
            'max_area': float(np.max(areas)),
            'min_area': float(np.min(areas)),
            'avg_confidence': float(np.mean(confidences)),
            'high_confidence_count': high_conf,
            'med_confidence_count': med_conf,
            'low_confidence_count': low_conf,
            'avg_diff_intensity': float(np.mean(diff_image)),
            'max_diff_intensity': float(np.max(diff_image))
        }
        
        return stats
    
    def filter_detections_by_confidence(self, detections: List[Dict[str, Any]], 
                                      min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        根据置信度过滤检测结果
        
        Args:
            detections: 检测结果列表
            min_confidence: 最小置信度阈值
            
        Returns:
            过滤后的检测结果列表
        """
        filtered = [det for det in detections if det['confidence'] >= min_confidence]
        
        # 重新分配ID
        for i, detection in enumerate(filtered):
            detection['id'] = i + 1
        
        return filtered
    
    def non_maximum_suppression(self, detections: List[Dict[str, Any]], 
                               iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        非极大值抑制去除重叠检测框
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            NMS后的检测结果列表
        """
        if not detections:
            return detections
        
        # 按置信度排序
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while detections_sorted:
            # 保留置信度最高的
            current = detections_sorted.pop(0)
            keep.append(current)
            
            # 计算与剩余框的IoU
            remaining = []
            for det in detections_sorted:
                iou = self._compute_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections_sorted = remaining
        
        # 重新分配ID
        for i, detection in enumerate(keep):
            detection['id'] = i + 1
        
        return keep
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            bbox1: 边界框1 [x, y, w, h]
            bbox2: 边界框2 [x, y, w, h]
            
        Returns:
            IoU值
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 并集面积
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def get_detection_summary(self, change_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取整体检测总结
        
        Args:
            change_results: 所有面的变化检测结果
            
        Returns:
            总结信息
        """
        total_detections = 0
        total_changed_pixels = 0
        total_pixels = 0
        all_confidences = []
        face_detection_counts = {}
        
        for face_name, result in change_results.items():
            detections = result['detections']
            stats = result['detection_stats']
            
            total_detections += len(detections)
            total_changed_pixels += stats['total_changed_pixels']
            total_pixels += result['binary_image'].size
            
            face_detection_counts[face_name] = len(detections)
            
            if detections:
                all_confidences.extend([det['confidence'] for det in detections])
        
        # 按检测数量排序的面
        sorted_faces = sorted(face_detection_counts.items(), 
                             key=lambda x: x[1], reverse=True)
        
        summary = {
            'total_faces_processed': len(change_results),
            'total_detections': total_detections,
            'total_changed_pixels': total_changed_pixels,
            'overall_change_ratio': total_changed_pixels / total_pixels if total_pixels > 0 else 0,
            'faces_with_changes': len([f for f, c in face_detection_counts.items() if c > 0]),
            'faces_without_changes': len([f for f, c in face_detection_counts.items() if c == 0]),
            'face_detection_counts': face_detection_counts,
            'most_changed_faces': sorted_faces[:3],  # 前3个变化最多的面
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'high_confidence_detections': len([c for c in all_confidences if c > 0.8]),
            'medium_confidence_detections': len([c for c in all_confidences if 0.6 <= c <= 0.8]),
            'low_confidence_detections': len([c for c in all_confidences if c < 0.6])
        }
        
        return summary
    
    def create_edge_exclusion_mask(self, image_shape: Tuple[int, int], border_size: int = 30) -> np.ndarray:
        """
        创建边缘排除掩码，排除立方体面边缘区域的检测
        
        Args:
            image_shape: 图像尺寸 (height, width)
            border_size: 边缘排除的像素宽度
            
        Returns:
            边缘排除掩码（True=有效区域，False=边缘区域）
        """
        height, width = image_shape
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 排除边缘区域
        mask[:border_size, :] = 0      # 上边缘
        mask[-border_size:, :] = 0     # 下边缘  
        mask[:, :border_size] = 0      # 左边缘
        mask[:, -border_size:] = 0     # 右边缘
        
        # 对角落区域进行额外的圆角处理，减少更多边缘伪影
        corner_size = min(border_size * 2, min(height, width) // 4)
        if corner_size > border_size:
            # 左上角
            mask[:corner_size, :corner_size] = 0
            # 右上角
            mask[:corner_size, -corner_size:] = 0
            # 左下角
            mask[-corner_size:, :corner_size] = 0
            # 右下角
            mask[-corner_size:, -corner_size:] = 0
        
        return mask.astype(bool)
    
    def extract_bounding_boxes_improved(self, contours: List, binary_image: np.ndarray, 
                                       diff_image: np.ndarray, edge_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        改进的边界框提取方法，使用更准确的置信度计算
        
        Args:
            contours: 轮廓列表
            binary_image: 二值化图像
            diff_image: 差分图像
            edge_mask: 边缘掩码
            
        Returns:
            边界框信息列表
        """
        detections = []
        
        for i, contour in enumerate(contours):
            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查是否距离边缘太近（额外的边缘检查）
            margin = 15
            if (x < margin or y < margin or 
                x + w > binary_image.shape[1] - margin or 
                y + h > binary_image.shape[0] - margin):
                continue  # 跳过太靠近边缘的检测
            
            # 添加填充
            padding = self.config.bbox_padding
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(binary_image.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(binary_image.shape[0] - y_padded, h + 2 * padding)
            
            # 计算轮廓属性
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 改进的置信度计算
            confidence = self._calculate_improved_confidence(
                contour, diff_image, edge_mask, x, y, w, h
            )
            
            # 如果置信度太低，直接跳过
            if confidence < 0.3:  # 预过滤极低置信度
                continue
            
            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2
            
            detection = {
                'id': i + 1,
                'bbox': [int(x_padded), int(y_padded), int(w_padded), int(h_padded)],
                'bbox_original': [int(x), int(y), int(w), int(h)],
                'area': float(area),
                'perimeter': float(perimeter),
                'confidence': float(confidence),
                'center': [int(center_x), int(center_y)],
                'contour': contour
            }
            
            detections.append(detection)
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 重新分配ID
        for i, detection in enumerate(detections):
            detection['id'] = i + 1
        
        return detections
    
    def _calculate_improved_confidence(self, contour: np.ndarray, diff_image: np.ndarray, 
                                     edge_mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """
        计算改进的置信度分数
        
        Args:
            contour: 轮廓
            diff_image: 差分图像
            edge_mask: 边缘掩码
            x, y, w, h: 边界框坐标
            
        Returns:
            置信度分数 (0-1)
        """
        # 1. 基本几何特征
        area = cv2.contourArea(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 2. 差异强度特征
        roi_diff = diff_image[y:y+h, x:x+w]
        roi_mask = edge_mask[y:y+h, x:x+w]
        
        # 只考虑有效区域内的像素
        valid_diff = roi_diff[roi_mask[y:y+h, x:x+w]] if roi_mask[y:y+h, x:x+w].any() else roi_diff
        
        if len(valid_diff) == 0:
            return 0.0
        
        avg_diff = np.mean(valid_diff)
        max_diff = np.max(valid_diff)
        diff_std = np.std(valid_diff)
        
        # 3. 边缘距离惩罚
        image_h, image_w = diff_image.shape
        center_x, center_y = x + w//2, y + h//2
        
        # 距离边缘的最小距离
        edge_dist = min(center_x, center_y, image_w - center_x, image_h - center_y)
        edge_penalty = min(1.0, edge_dist / 50.0)  # 距离边缘50像素内开始惩罚
        
        # 4. 尺寸合理性
        size_score = min(1.0, area / self.config.min_contour_area)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        aspect_penalty = 1.0 / (1.0 + max(0, aspect_ratio - 3.0))  # 长宽比超过3:1开始惩罚
        
        # 5. 综合置信度计算
        geometric_score = (extent * 0.3 + solidity * 0.2) * aspect_penalty
        intensity_score = min(1.0, (avg_diff / 100.0) * 0.3 + (max_diff / 255.0) * 0.2)
        consistency_score = min(1.0, diff_std / 30.0) * 0.1  # 差异的一致性
        
        confidence = (geometric_score + intensity_score + consistency_score + size_score * 0.2) * edge_penalty
        
        return min(1.0, max(0.0, confidence))
    
    def cleanup_memory(self):
        """清理内存"""
        if self.cuda_utils.use_cuda:
            self.cuda_utils.cleanup_memory()
