#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
违章建筑检测系统
基于轮廓检测和区域像素差比较的两期图像违章建筑识别
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd


class IllegalBuildingDetector:
    """违章建筑检测系统"""
    
    def __init__(self, output_dir="illegal_building_results"):
        """
        初始化违章建筑检测系统
        
        Args:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 系统参数
        self.min_building_size = (50, 50)  # 最小建筑物尺寸 (50x50px/1080p)
        self.pixel_diff_threshold = 50     # 像素差异阈值
        self.contour_area_threshold = 2500  # 轮廓面积阈值 (50x50)
        self.illumination_range = (100, 1200)  # 光照范围 (lux)
        self.adaptive_threshold = True     # 是否使用自适应阈值
        
        print(f"违章建筑检测系统初始化完成")
        print(f"输出目录: {output_dir}")
        print(f"系统参数:")
        print(f"  最小建筑物尺寸: {self.min_building_size}")
        print(f"  像素差异阈值: {self.pixel_diff_threshold}")
        print(f"  轮廓面积阈值: {self.contour_area_threshold}")
    
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
            print(f"读取图像失败 {path}: {str(e)}")
            return None
    
    def perspective_correction(self, img1, img2):
        """
        3.2.3.1 透视变换矩阵进行角度偏差校正
        以第一期图像作为基准（标定图），将第二期图像配准到第一期图像坐标系
        
        Args:
            img1 (ndarray): 第一期图像（基准图像）
            img2 (ndarray): 第二期图像（待配准图像）
            
        Returns:
            tuple: (原始img1, 校正后的img2, 变换矩阵, 特征点信息)
        """
        print("执行透视变换矩阵角度偏差校正...")
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 使用SIFT检测器找到特征点
        sift = cv2.SIFT_create(nfeatures=1000)
        
        # 检测关键点和描述符
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        print(f"  第一期图像检测到 {len(kp1)} 个特征点")
        print(f"  第二期图像检测到 {len(kp2)} 个特征点")
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("  特征点不足，返回原图像")
            return img1, img2, None, {"keypoints1": 0, "keypoints2": 0, "matches": 0}
        
        # 使用FLANN匹配器进行特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用比值测试筛选好的匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"  筛选后的良好匹配点: {len(good_matches)}")
        
        if len(good_matches) < 4:
            print("  良好匹配点不足4个，返回原图像")
            return img1, img2, None, {"keypoints1": len(kp1), "keypoints2": len(kp2), "matches": len(good_matches)}
        
        # 提取匹配点的坐标（交换src和dst，让第二期图像配准到第一期图像）
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 第二期图像的点
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 第一期图像的点
        
        # 使用RANSAC计算透视变换矩阵（第二期→第一期）
        M, mask = cv2.findHomography(src_pts, dst_pts, 
                                   cv2.RANSAC, 
                                   ransacReprojThreshold=5.0)
        
        if M is None:
            print("  无法计算透视变换矩阵，返回原图像")
            return img1, img2, None, {"keypoints1": len(kp1), "keypoints2": len(kp2), "matches": len(good_matches)}
        
        # 计算内点比例
        inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
        print(f"  透视变换内点比例: {inlier_ratio:.2%}")
        
        # 应用透视变换校正第二期图像到第一期图像坐标系
        h, w = img1.shape[:2]  # 使用第一期图像尺寸作为基准
        corrected_img2 = cv2.warpPerspective(img2, M, (w, h))
        
        transform_info = {
            "keypoints1": len(kp1),
            "keypoints2": len(kp2),
            "matches": len(good_matches),
            "inlier_ratio": float(inlier_ratio),
            "transform_matrix": M.tolist() if M is not None else None
        }
        
        print("  透视变换校正完成")
        return img1, corrected_img2, M, transform_info
    
    def analyze_image_lighting(self, img):
        """
        分析图像光照特性
        
        Args:
            img (ndarray): 输入图像
            
        Returns:
            dict: 光照分析结果
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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
            lighting_type = "dark"  # 暗光
        elif mean_brightness > 180:
            lighting_type = "bright"  # 亮光
        else:
            lighting_type = "normal"  # 正常光照
            
        # 判断对比度
        if std_brightness < 30:
            contrast_type = "low"  # 低对比度
        elif std_brightness > 80:
            contrast_type = "high"  # 高对比度
        else:
            contrast_type = "normal"  # 正常对比度
        
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
    
    def adaptive_normalize_image(self, img, lighting_analysis=None, conservative_mode=False):
        """
        自适应归一化图像处理
        根据图像光照特性调整处理参数
        
        Args:
            img (ndarray): 输入图像
            lighting_analysis (dict): 光照分析结果，如果为None则自动分析
            conservative_mode (bool): 保守模式，减少过度处理
            
        Returns:
            tuple: (归一化后的图像, 光照分析结果)
        """
        if lighting_analysis is None:
            lighting_analysis = self.analyze_image_lighting(img)
        
        print(f"    光照类型: {lighting_analysis['lighting_type']}, "
              f"对比度: {lighting_analysis['contrast_type']}, "
              f"平均亮度: {lighting_analysis['mean_brightness']:.1f}")
        
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 根据光照类型调整CLAHE参数 - 保守模式减少增强强度
        if lighting_analysis['lighting_type'] == 'dark':
            # 暗光图像：适度增强，避免过度处理
            clip_limit = 3.0 if conservative_mode else 4.0
            tile_grid_size = (8, 8) if conservative_mode else (6, 6)
        elif lighting_analysis['lighting_type'] == 'bright':
            # 亮光图像：轻度增强
            clip_limit = 1.5 if conservative_mode else 2.0
            tile_grid_size = (12, 12) if conservative_mode else (10, 10)
        else:
            # 正常光照：标准参数
            clip_limit = 2.5 if conservative_mode else 3.0
            tile_grid_size = (8, 8)
        
        # 根据对比度类型进一步调整 - 保守模式减少调整幅度
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
        if not conservative_mode and (lighting_analysis['contrast_type'] == 'low' or lighting_analysis['dynamic_range'] < 100):
            # 只有低对比度或动态范围小的图像才应用全局直方图均衡化
            yuv = cv2.cvtColor(normalized, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return normalized, lighting_analysis
    
    def normalize_image(self, img):
        """
        3.2.3.2 归一化图像处理
        减少光照、天气等因素对图像的影响
        
        Args:
            img (ndarray): 输入图像
            
        Returns:
            ndarray: 归一化后的图像
        """
        normalized, _ = self.adaptive_normalize_image(img)
        return normalized
    
    def detect_building_contours(self, img):
        """
        3.2.4 建筑物区域识别（使用轮廓检测替代模型检测）
        
        Args:
            img (ndarray): 输入图像
            
        Returns:
            tuple: (轮廓列表, 建筑物区域列表, 处理过程图像)
        """
        print("  执行建筑物轮廓检测...")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 多层次边缘检测
        # 1. Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 2. 形态学操作连接断裂的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        
        # 3. 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选轮廓（建筑物特征）
        building_contours = []
        building_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积筛选：大于最小建筑物尺寸
            if area > self.contour_area_threshold:
                # 计算轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 尺寸筛选：符合最小建筑物尺寸要求
                if w >= self.min_building_size[0] and h >= self.min_building_size[1]:
                    # 长宽比筛选：排除过于细长的区域
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio <= 5.0:  # 长宽比不超过5:1
                        # 轮廓复杂度筛选：建筑物通常有一定的复杂度
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if 0.1 <= circularity <= 0.9:  # 排除过于圆形或过于复杂的形状
                                building_contours.append(contour)
                                building_regions.append({
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'perimeter': perimeter,
                                    'aspect_ratio': aspect_ratio,
                                    'circularity': circularity,
                                    'contour': contour
                                })
        
        print(f"    检测到 {len(contours)} 个总轮廓")
        print(f"    筛选出 {len(building_contours)} 个建筑物轮廓")
        
        # 创建处理过程可视化
        process_img = img.copy()
        
        # 绘制所有轮廓（灰色）
        cv2.drawContours(process_img, contours, -1, (128, 128, 128), 1)
        
        # 绘制建筑物轮廓（红色）
        cv2.drawContours(process_img, building_contours, -1, (0, 0, 255), 2)
        
        # 绘制边界框
        for region in building_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(process_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # 添加面积标签
            cv2.putText(process_img, f"{int(region['area'])}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return building_contours, building_regions, process_img
    
    def calculate_adaptive_threshold(self, diff_img, conservative_factor=1.5):
        """
        计算自适应阈值
        
        Args:
            diff_img (ndarray): 差异图像
            conservative_factor (float): 保守系数，越大越保守
            
        Returns:
            int: 自适应阈值
        """
        mean_diff = np.mean(diff_img)
        std_diff = np.std(diff_img)
        
        # 使用统计方法计算阈值：均值 + 保守系数 * 标准差
        adaptive_threshold = int(mean_diff + conservative_factor * std_diff)
        
        # 限制阈值范围
        adaptive_threshold = max(30, min(adaptive_threshold, 120))
        
        return adaptive_threshold
    
    def region_pixel_comparison(self, img1, img2, regions1, regions2):
        """
        3.2.5 区域像素差比较
        
        Args:
            img1 (ndarray): 第一期图像
            img2 (ndarray): 第二期图像
            regions1 (list): 第一期建筑物区域
            regions2 (list): 第二期建筑物区域
            
        Returns:
            dict: 像素差比较结果
        """
        print("  执行区域像素差比较...")
        
        # 转换为灰度图进行像素差计算
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 确保图像尺寸一致
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # 计算整体像素差
        diff_img = cv2.absdiff(gray1, gray2)
        
        # 计算自适应阈值
        if self.adaptive_threshold:
            current_threshold = self.calculate_adaptive_threshold(diff_img)
            print(f"    自适应阈值: {current_threshold} (原始阈值: {self.pixel_diff_threshold})")
        else:
            current_threshold = self.pixel_diff_threshold
        
        # 区域匹配和比较
        region_comparisons = []
        suspicious_regions = []
        
        # 简化的区域匹配：基于位置距离
        matched_pairs = self._match_regions(regions1, regions2)
        
        print(f"    匹配到 {len(matched_pairs)} 对区域")
        
        for i, (region1, region2) in enumerate(matched_pairs):
            # 提取区域像素
            x1, y1, w1, h1 = region1['bbox']
            x2, y2, w2, h2 = region2['bbox']
            
            # 使用较大的边界框确保覆盖完整区域
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            
            # 确保区域在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, gray1.shape[1] - x)
            h = min(h, gray1.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # 提取对应区域
            region_gray1 = gray1[y:y+h, x:x+w]
            region_gray2 = gray2[y:y+h, x:x+w]
            region_diff = diff_img[y:y+h, x:x+w]
            
            # 计算区域统计信息
            mean_diff = np.mean(region_diff)
            max_diff = np.max(region_diff)
            std_diff = np.std(region_diff)
            
            # 计算超过阈值的像素比例
            threshold_pixels = np.sum(region_diff > current_threshold)
            total_pixels = region_diff.size
            threshold_ratio = threshold_pixels / total_pixels if total_pixels > 0 else 0
            
            region_comparison = {
                'region_id': i + 1,
                'bbox': (x, y, w, h),
                'region1_info': region1,
                'region2_info': region2,
                'mean_diff': float(mean_diff),
                'max_diff': float(max_diff),
                'std_diff': float(std_diff),
                'threshold_pixels': int(threshold_pixels),
                'total_pixels': int(total_pixels),
                'threshold_ratio': float(threshold_ratio),
                'is_suspicious': threshold_ratio > 0.3  # 30%以上像素差异超过阈值则标记为疑似
            }
            
            region_comparisons.append(region_comparison)
            
            # 标记疑似违章建筑区域
            if region_comparison['is_suspicious']:
                suspicious_regions.append(region_comparison)
        
        # 检查新增区域（只在第二期出现的建筑物）
        unmatched_regions2 = self._find_unmatched_regions(regions2, matched_pairs, is_second_period=True)
        for i, region in enumerate(unmatched_regions2):
            x, y, w, h = region['bbox']
            
            # 提取区域差异
            region_diff = diff_img[y:y+h, x:x+w]
            mean_diff = np.mean(region_diff)
            threshold_pixels = np.sum(region_diff > current_threshold)
            total_pixels = region_diff.size
            threshold_ratio = threshold_pixels / total_pixels if total_pixels > 0 else 0
            
            new_building = {
                'region_id': f"new_{i + 1}",
                'bbox': (x, y, w, h),
                'region1_info': None,
                'region2_info': region,
                'mean_diff': float(mean_diff),
                'max_diff': 255.0,  # 新建筑物，最大差异
                'std_diff': float(np.std(region_diff)),
                'threshold_pixels': int(threshold_pixels),
                'total_pixels': int(total_pixels),
                'threshold_ratio': float(threshold_ratio),
                'is_suspicious': True,  # 新建筑物直接标记为疑似
                'is_new_building': True
            }
            
            region_comparisons.append(new_building)
            suspicious_regions.append(new_building)
        
        print(f"    发现 {len(suspicious_regions)} 个疑似违章建筑区域")
        
        return {
            'diff_image': diff_img,
            'region_comparisons': region_comparisons,
            'suspicious_regions': suspicious_regions,
            'total_regions_compared': len(matched_pairs),
            'new_buildings_detected': len(unmatched_regions2),
            'suspicious_count': len(suspicious_regions)
        }
    
    def _match_regions(self, regions1, regions2, distance_threshold=100):
        """
        基于位置距离匹配两期图像中的建筑物区域
        
        Args:
            regions1 (list): 第一期区域
            regions2 (list): 第二期区域
            distance_threshold (float): 距离阈值
            
        Returns:
            list: 匹配的区域对列表
        """
        matched_pairs = []
        used_indices2 = set()
        
        for region1 in regions1:
            x1, y1, w1, h1 = region1['bbox']
            center1 = (x1 + w1/2, y1 + h1/2)
            
            best_match = None
            best_distance = float('inf')
            best_index = -1
            
            for i, region2 in enumerate(regions2):
                if i in used_indices2:
                    continue
                
                x2, y2, w2, h2 = region2['bbox']
                center2 = (x2 + w2/2, y2 + h2/2)
                
                # 计算中心点距离
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = region2
                    best_index = i
            
            if best_match is not None:
                matched_pairs.append((region1, best_match))
                used_indices2.add(best_index)
        
        return matched_pairs
    
    def _find_unmatched_regions(self, regions, matched_pairs, is_second_period=False):
        """
        找到未匹配的区域
        
        Args:
            regions (list): 区域列表
            matched_pairs (list): 匹配对列表
            is_second_period (bool): 是否为第二期图像
            
        Returns:
            list: 未匹配的区域列表
        """
        if is_second_period:
            matched_regions = [pair[1] for pair in matched_pairs]
        else:
            matched_regions = [pair[0] for pair in matched_pairs]
        
        unmatched = []
        for region in regions:
            if region not in matched_regions:
                unmatched.append(region)
        
        return unmatched
    
    def create_detection_visualization(self, img1, img2, img1_processed, img2_processed, 
                                     comparison_result, transform_info, comparison_name):
        """
        创建违章建筑检测可视化图像
        
        Args:
            img1, img2: 原始图像
            img1_processed, img2_processed: 处理后图像
            comparison_result: 比较结果
            transform_info: 变换信息
            comparison_name: 比较名称
            
        Returns:
            str: 保存的图像路径
        """
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle(f'违章建筑检测分析 - {comparison_name}', fontsize=16, fontweight='bold')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 第一行：原始图像和透视校正
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Base Image\n第一期图像（基准）', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Original Image\n第二期图像（原始）', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(img2_processed, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'Aligned Image\n透视校正后\n(内点率: {transform_info.get("inlier_ratio", 0):.2%})', fontsize=11)
        axes[0, 2].axis('off')
        
        # 像素差异图（热图风格）
        diff_img = comparison_result['diff_image']
        axes[0, 3].imshow(diff_img, cmap='hot')
        axes[0, 3].set_title(f'热图差异\n(阈值: {self.pixel_diff_threshold})', fontsize=11)
        axes[0, 3].axis('off')
        
        # 添加 Difference Map（蓝色风格）
        # 创建增强的difference map
        enhanced_diff = self._create_enhanced_difference_map(diff_img)
        im = axes[0, 4].imshow(enhanced_diff, cmap='Blues')
        axes[0, 4].set_title('Difference Map\n差异映射图', fontsize=11)
        axes[0, 4].axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[0, 4], fraction=0.046, pad=0.04)
        
        # 第二行：建筑物检测结果
        # 绘制第一期建筑物检测
        img1_buildings = img1_processed.copy()
        for region in comparison_result['region_comparisons']:
            if region['region1_info'] is not None:
                x, y, w, h = region['region1_info']['bbox']
                cv2.rectangle(img1_buildings, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img1_buildings, f"B{region['region_id']}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        axes[1, 0].imshow(cv2.cvtColor(img1_buildings, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'第一期建筑物检测\n(检测数量: {len([r for r in comparison_result["region_comparisons"] if r["region1_info"] is not None])})', fontsize=11)
        axes[1, 0].axis('off')
        
        # 绘制第二期建筑物检测
        img2_buildings = img2_processed.copy()
        for region in comparison_result['region_comparisons']:
            if region['region2_info'] is not None:
                x, y, w, h = region['region2_info']['bbox']
                color = (0, 0, 255) if region['is_suspicious'] else (0, 255, 0)
                cv2.rectangle(img2_buildings, (x, y), (x+w, y+h), color, 2)
                label = f"S{region['region_id']}" if region['is_suspicious'] else f"B{region['region_id']}"
                cv2.putText(img2_buildings, label, 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        axes[1, 1].imshow(cv2.cvtColor(img2_buildings, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'第二期建筑物检测\n(疑似: {comparison_result["suspicious_count"]})', fontsize=11)
        axes[1, 1].axis('off')
        
        # 疑似违章建筑区域放大显示
        if comparison_result['suspicious_regions']:
            # 选择第一个疑似区域进行放大显示
            suspicious = comparison_result['suspicious_regions'][0]
            x, y, w, h = suspicious['bbox']
            
            # 添加边距
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img2_processed.shape[1], x + w + margin)
            y2 = min(img2_processed.shape[0], y + h + margin)
            
            cropped_region = img2_processed[y1:y2, x1:x2]
            axes[1, 2].imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title(f'疑似区域放大\n(差异率: {suspicious["threshold_ratio"]:.2%})', fontsize=11)
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, '无疑似区域', ha='center', va='center', 
                           transform=axes[1, 2].transAxes, fontsize=14)
            axes[1, 2].set_title('疑似区域', fontsize=11)
            axes[1, 2].axis('off')
        
        # 差异统计直方图
        diff_flat = diff_img.flatten()
        axes[1, 3].hist(diff_flat, bins=50, alpha=0.7, color='red')
        axes[1, 3].axvline(x=self.pixel_diff_threshold, color='black', linestyle='--', 
                          label=f'阈值: {self.pixel_diff_threshold}')
        axes[1, 3].set_title('像素差异分布', fontsize=11)
        axes[1, 3].set_xlabel('像素差值')
        axes[1, 3].set_ylabel('频次')
        axes[1, 3].legend()
        
        # 添加边缘轮廓对比图
        contour_comparison = self._create_contour_comparison(img1_processed, img2_processed, diff_img)
        axes[1, 4].imshow(cv2.cvtColor(contour_comparison, cv2.COLOR_BGR2RGB))
        axes[1, 4].set_title('轮廓对比\nContour Comparison', fontsize=11)
        axes[1, 4].axis('off')
        
        # 第三行：统计信息和系统参数
        self._draw_detection_statistics(axes[2, 0], comparison_result)
        self._draw_system_parameters(axes[2, 1])
        self._draw_region_analysis(axes[2, 2], comparison_result)
        self._draw_technical_process(axes[2, 3])
        self._draw_difference_analysis(axes[2, 4], comparison_result)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in comparison_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        image_path = os.path.join(self.output_dir, f"illegal_building_detection_{safe_name}_{timestamp}.jpg")
        
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return image_path
    
    def _draw_detection_statistics(self, ax, result):
        """绘制检测统计信息"""
        stats_text = f"""🏢 检测统计信息

总区域对比: {result['total_regions_compared']}
新建筑检测: {result['new_buildings_detected']}
疑似违章: {result['suspicious_count']}

📊 像素差异统计:
平均差异: {np.mean(result['diff_image']):.2f}
最大差异: {np.max(result['diff_image']):.2f}
标准差: {np.std(result['diff_image']):.2f}

🎯 阈值设置:
像素差阈值: {self.pixel_diff_threshold}
疑似判定: 30%像素超阈值
"""
        
        ax.text(0.05, 0.95, stats_text, ha='left', va='top', transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('检测统计', fontsize=12, fontweight='bold')
    
    def _draw_system_parameters(self, ax):
        """绘制系统参数"""
        params_text = f"""⚙️ 系统参数配置

📷 图像要求:
分辨率: ≥1080P
目标尺寸: ≥50×50px
光照范围: {self.illumination_range[0]}-{self.illumination_range[1]} lux

🔧 检测参数:
最小建筑尺寸: {self.min_building_size}
轮廓面积阈值: {self.contour_area_threshold}
像素差阈值: {self.pixel_diff_threshold}
长宽比限制: ≤5:1
圆形度范围: 0.1-0.9

📝 技术标准:
符合违建检测规范
满足实时处理要求
"""
        
        ax.text(0.05, 0.95, params_text, ha='left', va='top', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('系统参数', fontsize=12, fontweight='bold')
    
    def _draw_region_analysis(self, ax, result):
        """绘制区域分析"""
        if not result['suspicious_regions']:
            ax.text(0.5, 0.5, '暂无疑似区域', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
        else:
            # 显示疑似区域详情
            suspicious = result['suspicious_regions'][0]  # 显示第一个疑似区域
            
            analysis_text = f"""🚨 疑似区域分析

区域ID: {suspicious['region_id']}
位置: ({suspicious['bbox'][0]}, {suspicious['bbox'][1]})
尺寸: {suspicious['bbox'][2]}×{suspicious['bbox'][3]}

像素差异分析:
平均差异: {suspicious['mean_diff']:.2f}
最大差异: {suspicious['max_diff']:.2f}
标准差: {suspicious['std_diff']:.2f}

阈值分析:
超阈值像素: {suspicious['threshold_pixels']}
总像素数: {suspicious['total_pixels']}
超阈值比例: {suspicious['threshold_ratio']:.2%}

判定结果: {'疑似违章建筑' if suspicious['is_suspicious'] else '正常建筑'}
"""
            
            color = 'lightcoral' if suspicious['is_suspicious'] else 'lightgreen'
            ax.text(0.05, 0.95, analysis_text, ha='left', va='top', transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('区域分析', fontsize=12, fontweight='bold')
    
    def _draw_technical_process(self, ax):
        """绘制技术流程"""
        process_text = """🔬 技术处理流程

1️⃣ 图像预处理
• 透视变换矩阵校正
• CLAHE光照归一化
• 直方图均衡化

2️⃣ 建筑物检测
• Canny边缘检测
• 形态学操作
• 轮廓筛选过滤

3️⃣ 区域匹配
• 基于位置距离匹配
• 几何特征验证
• 新建筑物识别

4️⃣ 像素差比较
• 逐像素差值计算
• 阈值比例分析
• 疑似区域标记

✅ 符合技术规范要求"""
        
        ax.text(0.05, 0.95, process_text, ha='left', va='top', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('技术流程', fontsize=12, fontweight='bold')
    
    def _create_enhanced_difference_map(self, diff_img):
        """
        创建增强的difference map
        
        Args:
            diff_img (ndarray): 原始差异图
            
        Returns:
            ndarray: 增强的差异图
        """
        # 归一化到0-1范围
        normalized_diff = diff_img.astype(np.float32) / 255.0
        
        # 应用非线性增强
        enhanced_diff = np.power(normalized_diff, 0.5)  # 开方增强对比度
        
        # 应用高斯滤波平滑处理
        enhanced_diff = cv2.GaussianBlur(enhanced_diff, (3, 3), 0)
        
        # 阈值化处理，突出显著差异区域
        threshold_mask = diff_img > self.pixel_diff_threshold
        enhanced_diff[threshold_mask] = enhanced_diff[threshold_mask] * 1.5
        enhanced_diff = np.clip(enhanced_diff, 0, 1)
        
        return enhanced_diff
    
    def _create_contour_comparison(self, img1, img2, diff_img):
        """
        创建轮廓对比图
        
        Args:
            img1 (ndarray): 第一期图像
            img2 (ndarray): 第二期图像
            diff_img (ndarray): 差异图像
            
        Returns:
            ndarray: 轮廓对比图
        """
        # 创建组合图像
        h, w = img1.shape[:2]
        comparison_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 将第一期图像转为红色通道
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        comparison_img[:, :, 2] = gray1  # 红色通道
        
        # 将第二期图像转为绿色通道
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        comparison_img[:, :, 1] = gray2  # 绿色通道
        
        # 将差异区域转为蓝色通道
        diff_threshold = diff_img > self.pixel_diff_threshold
        comparison_img[:, :, 0] = diff_img  # 蓝色通道
        
        # 增强差异区域的可见性
        comparison_img[diff_threshold] = [255, 100, 100]  # 浅蓝色标记差异区域
        
        return comparison_img
    
    def _draw_difference_analysis(self, ax, result):
        """绘制差异分析信息"""
        diff_img = result['diff_image']
        
        # 计算差异统计
        total_pixels = diff_img.size
        significant_diff_pixels = np.sum(diff_img > self.pixel_diff_threshold)
        diff_percentage = (significant_diff_pixels / total_pixels) * 100
        
        mean_diff = np.mean(diff_img)
        max_diff = np.max(diff_img)
        std_diff = np.std(diff_img)
        
        analysis_text = f"""📊 差异分析统计
        
🔍 整体差异分析:
像素总数: {total_pixels:,}
显著差异像素: {significant_diff_pixels:,}
差异比例: {diff_percentage:.2f}%

📈 统计指标:
平均差异: {mean_diff:.2f}
最大差异: {max_diff:.2f}
标准差: {std_diff:.2f}

🎯 阈值设置:
差异阈值: {self.pixel_diff_threshold}
疑似判定: 30%像素超阈值

📋 结果评估:
检测精度: {'高' if diff_percentage > 5 else '中' if diff_percentage > 1 else '低'}
变化程度: {'显著' if mean_diff > 30 else '中等' if mean_diff > 15 else '轻微'}
"""
        
        color = 'lightcoral' if diff_percentage > 5 else 'lightyellow' if diff_percentage > 1 else 'lightgreen'
        ax.text(0.05, 0.95, analysis_text, ha='left', va='top', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('差异分析', fontsize=12, fontweight='bold')
    
    def process_illegal_building_detection(self, img1_path, img2_path):
        """
        处理违章建筑检测的完整流程
        
        Args:
            img1_path (str): 第一期图像路径
            img2_path (str): 第二期图像路径
            
        Returns:
            dict: 检测结果
        """
        comparison_name = f"{os.path.basename(img1_path)}_vs_{os.path.basename(img2_path)}"
        print(f"\n=== 违章建筑检测分析: {comparison_name} ===")
        
        # 1. 加载图像
        print("1. 加载图像...")
        img1 = self.load_image_with_chinese_path(img1_path)
        img2 = self.load_image_with_chinese_path(img2_path)
        
        if img1 is None or img2 is None:
            return None
        
        print(f"   第一期图像尺寸: {img1.shape}")
        print(f"   第二期图像尺寸: {img2.shape}")
        
        # 2. 透视变换校正
        print("2. 透视变换校正...")
        base_img, aligned_img, transform_matrix, transform_info = self.perspective_correction(img1, img2)
        
        # 3. 自适应图像归一化
        print("3. 自适应图像归一化处理...")
        print("  分析第一期图像光照特性...")
        normalized_img1, lighting_info1 = self.adaptive_normalize_image(base_img, conservative_mode=True)
        
        print("  分析第二期图像光照特性...")
        normalized_img2, lighting_info2 = self.adaptive_normalize_image(aligned_img, conservative_mode=True)
        
        # 4. 建筑物轮廓检测
        print("4. 建筑物轮廓检测...")
        contours1, regions1, process_img1 = self.detect_building_contours(normalized_img1)
        contours2, regions2, process_img2 = self.detect_building_contours(normalized_img2)
        
        # 5. 区域像素差比较
        print("5. 区域像素差比较...")
        comparison_result = self.region_pixel_comparison(normalized_img1, normalized_img2, regions1, regions2)
        
        # 6. 生成可视化
        print("6. 生成检测可视化...")
        visualization_path = self.create_detection_visualization(
            img1, img2, normalized_img1, normalized_img2,
            comparison_result, transform_info, comparison_name
        )
        
        # 7. 整理结果
        result = {
            'analysis_info': {
                'image1_path': img1_path,
                'image2_path': img2_path,
                'comparison_name': comparison_name,
                'timestamp': datetime.now().isoformat(),
                'image1_size': img1.shape,
                'image2_size': img2.shape
            },
            'preprocessing': {
                'perspective_correction': transform_info,
                'normalization_applied': True,
                'lighting_analysis': {
                    'period1': lighting_info1,
                    'period2': lighting_info2
                }
            },
            'building_detection': {
                'period1_buildings': len(regions1),
                'period2_buildings': len(regions2),
                'period1_contours': len(contours1),
                'period2_contours': len(contours2)
            },
            'pixel_comparison': comparison_result,
            'system_parameters': {
                'min_building_size': self.min_building_size,
                'pixel_diff_threshold': self.pixel_diff_threshold,
                'contour_area_threshold': self.contour_area_threshold,
                'illumination_range': self.illumination_range
            },
            'visualization_path': visualization_path
        }
        
        print(f"✅ 检测完成！发现 {comparison_result['suspicious_count']} 个疑似违章建筑区域")
        return result
    
    def save_detection_results(self, results):
        """保存检测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_path = os.path.join(self.output_dir, f"illegal_building_detection_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已保存: {json_path}")
        
        # 生成统计报告
        self._generate_detection_report(results, timestamp)
        
        return json_path
    
    def _make_json_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        else:
            return obj
    
    def _generate_detection_report(self, results, timestamp):
        """生成检测报告"""
        report_path = os.path.join(self.output_dir, f"detection_report_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 违章建筑检测报告\n\n")
            f.write(f"**检测时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            
            # 基本信息
            analysis_info = results['analysis_info']
            f.write("## 📋 基本信息\n\n")
            f.write(f"- **第一期图像**: {os.path.basename(analysis_info['image1_path'])}\n")
            f.write(f"- **第二期图像**: {os.path.basename(analysis_info['image2_path'])}\n")
            f.write(f"- **图像尺寸**: {analysis_info['image1_size'][:2]}\n\n")
            
            # 检测结果摘要
            pixel_comparison = results['pixel_comparison']
            f.write("## 🎯 检测结果摘要\n\n")
            f.write(f"- **疑似违章建筑**: {pixel_comparison['suspicious_count']} 个\n")
            f.write(f"- **总区域对比**: {pixel_comparison['total_regions_compared']} 对\n")
            f.write(f"- **新建筑检测**: {pixel_comparison['new_buildings_detected']} 个\n\n")
            
            # 详细分析
            if pixel_comparison['suspicious_regions']:
                f.write("## 🚨 疑似区域详情\n\n")
                for i, region in enumerate(pixel_comparison['suspicious_regions'], 1):
                    f.write(f"### 疑似区域 {i}\n")
                    f.write(f"- **区域ID**: {region['region_id']}\n")
                    f.write(f"- **位置**: ({region['bbox'][0]}, {region['bbox'][1]})\n")
                    f.write(f"- **尺寸**: {region['bbox'][2]}×{region['bbox'][3]} 像素\n")
                    f.write(f"- **平均差异**: {region['mean_diff']:.2f}\n")
                    f.write(f"- **最大差异**: {region['max_diff']:.2f}\n")
                    f.write(f"- **超阈值比例**: {region['threshold_ratio']:.2%}\n")
                    f.write(f"- **是否新建筑**: {'是' if region.get('is_new_building', False) else '否'}\n\n")
            
            # 技术参数
            sys_params = results['system_parameters']
            f.write("## ⚙️ 技术参数\n\n")
            f.write(f"- **最小建筑尺寸**: {sys_params['min_building_size']}\n")
            f.write(f"- **像素差阈值**: {sys_params['pixel_diff_threshold']}\n")
            f.write(f"- **轮廓面积阈值**: {sys_params['contour_area_threshold']}\n")
            f.write(f"- **光照范围**: {sys_params['illumination_range']} lux\n\n")
            
            # 处理过程
            f.write("## 🔬 处理过程\n\n")
            f.write("1. **透视变换校正**: 使用SIFT特征点匹配和RANSAC算法\n")
            f.write("2. **图像归一化**: CLAHE光照均衡化和直方图均衡化\n")
            f.write("3. **建筑物检测**: Canny边缘检测 + 轮廓筛选\n")
            f.write("4. **区域匹配**: 基于位置距离的区域对应\n")
            f.write("5. **像素差比较**: 逐像素差值计算和阈值分析\n\n")
            
            f.write("---\n")
            f.write("*本报告由违章建筑检测系统自动生成*")
        
        print(f"检测报告已生成: {report_path}")


def analyze_false_positive_regions(img1_path, img2_path, region_coords=None):
    """
    分析可能的假阳性区域
    
    Args:
        img1_path (str): 第一期图像路径
        img2_path (str): 第二期图像路径
        region_coords (tuple): 可选的区域坐标 (x, y, w, h)
    """
    detector = IllegalBuildingDetector()
    
    print("=== 假阳性区域分析 ===")
    
    # 加载图像
    img1 = detector.load_image_with_chinese_path(img1_path)
    img2 = detector.load_image_with_chinese_path(img2_path)
    
    if img1 is None or img2 is None:
        return
    
    # 透视变换校正
    base_img, aligned_img, transform_matrix, transform_info = detector.perspective_correction(img1, img2)
    
    # 分别分析原始图像和处理后图像的差异
    print("\n1. 原始图像差异分析...")
    gray1_orig = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray2_orig = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    diff_orig = cv2.absdiff(gray1_orig, gray2_orig)
    
    print("\n2. 处理后图像差异分析...")
    normalized_img1, lighting_info1 = detector.adaptive_normalize_image(base_img)
    normalized_img2, lighting_info2 = detector.adaptive_normalize_image(aligned_img)
    gray1_proc = cv2.cvtColor(normalized_img1, cv2.COLOR_BGR2GRAY)
    gray2_proc = cv2.cvtColor(normalized_img2, cv2.COLOR_BGR2GRAY)
    diff_proc = cv2.absdiff(gray1_proc, gray2_proc)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建对比分析图
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('假阳性区域诊断分析', fontsize=16, fontweight='bold')
    
    # 第一行：原始图像对比
    axes[0, 0].imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('第一期图像（校正后）', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('第二期图像（校正后）', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(diff_orig, cmap='hot', vmin=0, vmax=255)
    axes[0, 2].set_title(f'原始差异图\n平均差异: {np.mean(diff_orig):.2f}', fontsize=11)
    axes[0, 2].axis('off')
    
    # 原始差异直方图
    axes[0, 3].hist(diff_orig.flatten(), bins=50, alpha=0.7, color='blue', label='原始差异')
    axes[0, 3].axvline(x=detector.pixel_diff_threshold, color='red', linestyle='--', label='阈值')
    axes[0, 3].set_title('原始差异分布', fontsize=11)
    axes[0, 3].legend()
    
    # 第二行：处理后图像对比
    axes[1, 0].imshow(cv2.cvtColor(normalized_img1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'第一期（处理后）\n{lighting_info1["lighting_type"]}', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(normalized_img2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'第二期（处理后）\n{lighting_info2["lighting_type"]}', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_proc, cmap='hot', vmin=0, vmax=255)
    axes[1, 2].set_title(f'处理后差异图\n平均差异: {np.mean(diff_proc):.2f}', fontsize=11)
    axes[1, 2].axis('off')
    
    # 处理后差异直方图
    axes[1, 3].hist(diff_proc.flatten(), bins=50, alpha=0.7, color='green', label='处理后差异')
    axes[1, 3].axvline(x=detector.pixel_diff_threshold, color='red', linestyle='--', label='阈值')
    axes[1, 3].set_title('处理后差异分布', fontsize=11)
    axes[1, 3].legend()
    
    # 第三行：差异对比和分析
    diff_comparison = diff_proc.astype(np.float32) - diff_orig.astype(np.float32)
    axes[2, 0].imshow(diff_comparison, cmap='RdBu', vmin=-100, vmax=100)
    axes[2, 0].set_title('处理增加的差异\n(蓝色=减少, 红色=增加)', fontsize=11)
    axes[2, 0].axis('off')
    
    # 阈值掩码对比
    mask_orig = diff_orig > detector.pixel_diff_threshold
    mask_proc = diff_proc > detector.pixel_diff_threshold
    mask_diff = mask_proc.astype(np.uint8) - mask_orig.astype(np.uint8)
    
    axes[2, 1].imshow(mask_diff, cmap='RdBu', vmin=-1, vmax=1)
    axes[2, 1].set_title('阈值掩码变化\n(红色=新增假阳性)', fontsize=11)
    axes[2, 1].axis('off')
    
    # 统计分析
    orig_above_threshold = np.sum(mask_orig)
    proc_above_threshold = np.sum(mask_proc)
    false_positives = np.sum(mask_diff > 0)
    
    stats_text = f"""差异统计分析:

原始超阈值像素: {orig_above_threshold:,}
处理后超阈值像素: {proc_above_threshold:,}
新增假阳性像素: {false_positives:,}

比例变化:
原始: {orig_above_threshold/diff_orig.size*100:.2f}%
处理后: {proc_above_threshold/diff_proc.size*100:.2f}%

处理影响:
{'增加了假阳性' if false_positives > orig_above_threshold*0.1 else '影响较小'}
"""
    
    axes[2, 2].text(0.05, 0.95, stats_text, ha='left', va='top', transform=axes[2, 2].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('统计分析', fontsize=12)
    
    # 建议优化方案
    suggestions_text = f"""优化建议:

1. 调整阈值:
   建议阈值: {int(np.mean(diff_orig) + 2*np.std(diff_orig))}
   (当前: {detector.pixel_diff_threshold})

2. 处理参数优化:
   {'减少CLAHE强度' if np.mean(diff_proc) > np.mean(diff_orig)*1.2 else '参数合适'}

3. 配准精度:
   内点率: {transform_info.get('inlier_ratio', 0):.1%}
   {'建议提高配准精度' if transform_info.get('inlier_ratio', 0) < 0.6 else '配准精度良好'}

4. 后处理建议:
   应用形态学滤波
   区域连通性分析
"""
    
    axes[2, 3].text(0.05, 0.95, suggestions_text, ha='left', va='top', transform=axes[2, 3].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    axes[2, 3].set_title('优化建议', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 输出详细分析
    print(f"\n📊 详细分析结果:")
    print(f"   原始平均差异: {np.mean(diff_orig):.2f}")
    print(f"   处理后平均差异: {np.mean(diff_proc):.2f}")
    print(f"   差异增幅: {(np.mean(diff_proc)/np.mean(diff_orig)-1)*100:.1f}%")
    print(f"   新增假阳性像素: {false_positives:,} ({false_positives/diff_orig.size*100:.3f}%)")
    
    return {
        'original_diff': diff_orig,
        'processed_diff': diff_proc,
        'false_positives': false_positives,
        'suggestions': {
            'recommended_threshold': int(np.mean(diff_orig) + 2*np.std(diff_orig)),
            'current_threshold': detector.pixel_diff_threshold
        }
    }


def show_processed_comparison(img1_path, img2_path):
    """
    显示处理后的两张图像对比
    
    Args:
        img1_path (str): 第一期图像路径
        img2_path (str): 第二期图像路径
    """
    # 创建检测器实例
    detector = IllegalBuildingDetector()
    
    print("加载图像...")
    # 加载图像
    img1 = detector.load_image_with_chinese_path(img1_path)
    img2 = detector.load_image_with_chinese_path(img2_path)
    
    if img1 is None or img2 is None:
        print("图像加载失败")
        return
    
    print("执行透视变换校正...")
    # 透视变换校正
    base_img, aligned_img, transform_matrix, transform_info = detector.perspective_correction(img1, img2)
    
    print("执行自适应图像归一化...")
    # 自适应图像归一化
    print("  分析第一期图像光照...")
    normalized_img1, lighting_info1 = detector.adaptive_normalize_image(base_img)
    print("  分析第二期图像光照...")
    normalized_img2, lighting_info2 = detector.adaptive_normalize_image(aligned_img)
    
    print("计算差异图...")
    # 计算差异图
    gray1 = cv2.cvtColor(normalized_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(normalized_img2, cv2.COLOR_BGR2GRAY)
    diff_img = cv2.absdiff(gray1, gray2)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建显示窗口
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('图像处理对比显示', fontsize=16, fontweight='bold')
    
    # 第一行：原始图像
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('第一期图像（原始）', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('第二期图像（原始）', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'透视校正后\n内点率: {transform_info.get("inlier_ratio", 0):.2%}', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行：处理后图像和差异图
    axes[1, 0].imshow(cv2.cvtColor(normalized_img1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('第一期图像（处理后）', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(normalized_img2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('第二期图像（处理后）', fontsize=12)
    axes[1, 1].axis('off')
    
    # 差异图
    im = axes[1, 2].imshow(diff_img, cmap='hot')
    axes[1, 2].set_title(f'差异图\n阈值: {detector.pixel_diff_threshold}', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # 显示统计信息
    total_pixels = diff_img.size
    significant_diff_pixels = np.sum(diff_img > detector.pixel_diff_threshold)
    diff_percentage = (significant_diff_pixels / total_pixels) * 100
    mean_diff = np.mean(diff_img)
    
    print(f"\n📊 处理结果统计:")
    print(f"   透视变换内点率: {transform_info.get('inlier_ratio', 0):.2%}")
    print(f"   平均像素差异: {mean_diff:.2f}")
    print(f"   显著差异比例: {diff_percentage:.2f}%")
    print(f"   匹配特征点: {transform_info.get('matches', 0)} 个")
    
    print(f"\n🔆 光照分析结果:")
    print(f"   第一期图像: {lighting_info1['lighting_type']} (亮度: {lighting_info1['mean_brightness']:.1f})")
    print(f"   第二期图像: {lighting_info2['lighting_type']} (亮度: {lighting_info2['mean_brightness']:.1f})")
    print(f"   光照一致性: {'良好' if abs(lighting_info1['mean_brightness'] - lighting_info2['mean_brightness']) < 30 else '需要注意'}")
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 测试图像路径
    img1_path = r'C:\Users\admin\Desktop\两期比对\两期比对\Snipaste_2025-09-08_15-39-45.png'  # 第一期图像
    img2_path = r'C:\Users\admin\Desktop\两期比对\两期比对\Snipaste_2025-09-08_15-39-16.png'  # 第二期图像
    
    try:
        # 创建违章建筑检测系统
        detector = IllegalBuildingDetector()
        
        # 执行检测
        result = detector.process_illegal_building_detection(img1_path, img2_path)
        
        if result:
            # 保存结果
            detector.save_detection_results(result)
            
            print(f"\n📊 检测结果统计:")
            print(f"   第一期建筑物: {result['building_detection']['period1_buildings']} 个")
            print(f"   第二期建筑物: {result['building_detection']['period2_buildings']} 个")
            print(f"   疑似违章建筑: {result['pixel_comparison']['suspicious_count']} 个")
            print(f"   新建筑物: {result['pixel_comparison']['new_buildings_detected']} 个")
            print(f"📁 可视化图像: {os.path.basename(result['visualization_path'])}")
        else:
            print("❌ 检测失败")
            
    except Exception as e:
        print(f"❌ 违章建筑检测时出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 

# ==================== 使用示例 ====================
# 如果只想显示处理后的图像对比，可以使用以下代码：
# 
# if __name__ == "__main__":
#     # 替换为您的图像路径
#     img1_path = r'C:\Users\admin\Desktop\两期比对\两期比对\image1.png'
#     img2_path = r'C:\Users\admin\Desktop\两期比对\两期比对\image2.png'
#     
#     # 显示处理对比（不保存文件）
#     show_processed_comparison(img1_path, img2_path) 

def show_alignment_check(img1_path, img2_path):
    """
    显示透视校正后的重叠检查图像
    用于检测配准偏移和校正质量
    
    Args:
        img1_path (str): 第一期图像路径
        img2_path (str): 第二期图像路径
    """
    detector = IllegalBuildingDetector()
    
    print("=== 透视校正重叠检查 ===")
    
    # 加载图像
    img1 = detector.load_image_with_chinese_path(img1_path)
    img2 = detector.load_image_with_chinese_path(img2_path)
    
    if img1 is None or img2 is None:
        print("图像加载失败")
        return
    
    print(f"原始图像尺寸: {img1.shape} vs {img2.shape}")
    
    # 透视变换校正
    base_img, aligned_img, transform_matrix, transform_info = detector.perspective_correction(img1, img2)
    
    print(f"校正后图像尺寸: {base_img.shape} vs {aligned_img.shape}")
    print(f"透视变换内点率: {transform_info.get('inlier_ratio', 0):.2%}")
    print(f"匹配特征点数: {transform_info.get('matches', 0)}")
    
    # 创建不同的重叠可视化
    h, w = base_img.shape[:2]
    
    # 1. 棋盘格重叠 - 用于检测配准精度
    def create_checkerboard_overlay(img1, img2, block_size=50):
        overlay = img1.copy()
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 1:
                    i_end = min(i + block_size, h)
                    j_end = min(j + block_size, w)
                    overlay[i:i_end, j:j_end] = img2[i:i_end, j:j_end]
        return overlay
    
    # 2. 半透明重叠 - 用于检测整体偏移
    def create_alpha_blend(img1, img2, alpha=0.5):
        return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    
    # 3. 红绿重叠 - 用于检测细微偏移
    def create_red_green_overlay(img1, img2):
        overlay = np.zeros_like(img1)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        overlay[:, :, 1] = gray1  # 第一期图像 -> 绿色通道
        overlay[:, :, 2] = gray2  # 第二期图像 -> 红色通道
        return overlay
    
    # 4. 边缘重叠 - 用于检测结构偏移
    def create_edge_overlay(img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 提取边缘
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # 创建彩色边缘重叠
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 1] = edges1  # 第一期边缘 -> 绿色
        overlay[:, :, 2] = edges2  # 第二期边缘 -> 红色
        
        # 重叠区域显示为黄色
        overlap = np.logical_and(edges1 > 0, edges2 > 0)
        overlay[overlap] = [0, 255, 255]
        
        return overlay
    
    # 创建各种重叠图像
    checkerboard = create_checkerboard_overlay(base_img, aligned_img)
    alpha_blend = create_alpha_blend(base_img, aligned_img)
    red_green = create_red_green_overlay(base_img, aligned_img)
    edge_overlay = create_edge_overlay(base_img, aligned_img)
    
    # 计算配准质量指标
    gray1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    
    # 结构相似性指数 (SSIM)
    from skimage.metrics import structural_similarity as ssim
    ssim_score = ssim(gray1, gray2)
    
    # 归一化互相关 (NCC)
    def normalized_cross_correlation(img1, img2):
        img1_norm = (img1 - np.mean(img1)) / np.std(img1)
        img2_norm = (img2 - np.mean(img2)) / np.std(img2)
        return np.mean(img1_norm * img2_norm)
    
    ncc_score = normalized_cross_correlation(gray1, gray2)
    
    # 均方误差 (MSE)
    mse_score = np.mean((gray1.astype(np.float32) - gray2.astype(np.float32)) ** 2)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建显示窗口
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('透视校正重叠检查分析', fontsize=16, fontweight='bold')
    
    # 第一行：原始图像和校正后图像
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('第一期图像（基准）', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('第二期图像（原始）', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'第二期图像（校正后）\n内点率: {transform_info.get("inlier_ratio", 0):.2%}', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行：重叠检查
    axes[1, 0].imshow(cv2.cvtColor(checkerboard, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('棋盘格重叠\n（检测配准精度）', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(alpha_blend, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('半透明重叠\n（检测整体偏移）', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(red_green, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('红绿重叠\n（红=期二，绿=期一）', fontsize=12)
    axes[1, 2].axis('off')
    
    # 第三行：边缘分析和质量指标
    axes[2, 0].imshow(cv2.cvtColor(edge_overlay, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('边缘重叠\n（黄=重合，红绿=偏移）', fontsize=12)
    axes[2, 0].axis('off')
    
    # 显示配准质量指标
    quality_text = f"""配准质量评估:

SSIM (结构相似性):
{ssim_score:.4f}
(1.0 = 完全相似)

NCC (归一化互相关):
{ncc_score:.4f}
(1.0 = 完全相关)

MSE (均方误差):
{mse_score:.2f}
(0 = 完全匹配)

总体评估:
{'优秀' if ssim_score > 0.9 else '良好' if ssim_score > 0.8 else '一般' if ssim_score > 0.7 else '较差'}
"""
    
    color = 'lightgreen' if ssim_score > 0.8 else 'lightyellow' if ssim_score > 0.7 else 'lightcoral'
    axes[2, 1].text(0.05, 0.95, quality_text, ha='left', va='top', transform=axes[2, 1].transAxes,
                    fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    axes[2, 1].set_title('质量指标', fontsize=12)
    
    # 显示偏移诊断建议
    diagnostic_text = f"""偏移诊断建议:

1. 特征点匹配:
   匹配点数: {transform_info.get('matches', 0)}
   内点率: {transform_info.get('inlier_ratio', 0):.1%}
   {'✅ 匹配良好' if transform_info.get('inlier_ratio', 0) > 0.6 else '⚠️ 匹配不佳'}

2. 配准精度:
   SSIM评分: {ssim_score:.3f}
   {'✅ 配准精确' if ssim_score > 0.85 else '⚠️ 存在偏移'}

3. 改进建议:
   {'增加特征点数量' if transform_info.get('matches', 0) < 100 else ''}
   {'提高图像质量' if ssim_score < 0.8 else ''}
   {'检查图像内容差异' if ncc_score < 0.7 else ''}

4. 偏移影响:
   {'可能影响差异检测准确性' if ssim_score < 0.8 else '对差异检测影响较小'}
"""
    
    axes[2, 2].text(0.05, 0.95, diagnostic_text, ha='left', va='top', transform=axes[2, 2].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('诊断建议', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 输出详细分析结果
    print(f"\n📊 配准质量分析:")
    print(f"   SSIM (结构相似性): {ssim_score:.4f}")
    print(f"   NCC (归一化互相关): {ncc_score:.4f}")
    print(f"   MSE (均方误差): {mse_score:.2f}")
    print(f"   配准质量: {'优秀' if ssim_score > 0.9 else '良好' if ssim_score > 0.8 else '一般' if ssim_score > 0.7 else '较差'}")
    
    if ssim_score < 0.8:
        print(f"\n⚠️  配准质量警告:")
        print(f"   SSIM评分 {ssim_score:.3f} 低于0.8，可能存在明显偏移")
        print(f"   这可能导致假阳性差异增加")
        print(f"   建议检查特征点分布和图像质量")
    
    return {
        'ssim': ssim_score,
        'ncc': ncc_score,
        'mse': mse_score,
        'transform_info': transform_info,
        'alignment_quality': 'excellent' if ssim_score > 0.9 else 'good' if ssim_score > 0.8 else 'fair' if ssim_score > 0.7 else 'poor'
    }


def analyze_false_positive_regions(img1_path, img2_path, region_coords=None):
    """
    分析可能的假阳性区域
    
    Args:
        img1_path (str): 第一期图像路径
        img2_path (str): 第二期图像路径
        region_coords (tuple): 可选的区域坐标 (x, y, w, h)
    """
    detector = IllegalBuildingDetector()
    
    print("=== 假阳性区域分析 ===")
    
    # 加载图像
    img1 = detector.load_image_with_chinese_path(img1_path)
    img2 = detector.load_image_with_chinese_path(img2_path)
    
    if img1 is None or img2 is None:
        return
    
    # 透视变换校正
    base_img, aligned_img, transform_matrix, transform_info = detector.perspective_correction(img1, img2)
    
    # 分别分析原始图像和处理后图像的差异
    print("\n1. 原始图像差异分析...")
    gray1_orig = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray2_orig = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    diff_orig = cv2.absdiff(gray1_orig, gray2_orig)
    
    print("\n2. 处理后图像差异分析...")
    normalized_img1, lighting_info1 = detector.adaptive_normalize_image(base_img)
    normalized_img2, lighting_info2 = detector.adaptive_normalize_image(aligned_img)
    gray1_proc = cv2.cvtColor(normalized_img1, cv2.COLOR_BGR2GRAY)
    gray2_proc = cv2.cvtColor(normalized_img2, cv2.COLOR_BGR2GRAY)
    diff_proc = cv2.absdiff(gray1_proc, gray2_proc)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建对比分析图
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('假阳性区域诊断分析', fontsize=16, fontweight='bold')
    
    # 第一行：原始图像对比
    axes[0, 0].imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('第一期图像（校正后）', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('第二期图像（校正后）', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(diff_orig, cmap='hot', vmin=0, vmax=255)
    axes[0, 2].set_title(f'原始差异图\n平均差异: {np.mean(diff_orig):.2f}', fontsize=11)
    axes[0, 2].axis('off')
    
    # 原始差异直方图
    axes[0, 3].hist(diff_orig.flatten(), bins=50, alpha=0.7, color='blue', label='原始差异')
    axes[0, 3].axvline(x=detector.pixel_diff_threshold, color='red', linestyle='--', label='阈值')
    axes[0, 3].set_title('原始差异分布', fontsize=11)
    axes[0, 3].legend()
    
    # 第二行：处理后图像对比
    axes[1, 0].imshow(cv2.cvtColor(normalized_img1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'第一期（处理后）\n{lighting_info1["lighting_type"]}', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(normalized_img2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'第二期（处理后）\n{lighting_info2["lighting_type"]}', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_proc, cmap='hot', vmin=0, vmax=255)
    axes[1, 2].set_title(f'处理后差异图\n平均差异: {np.mean(diff_proc):.2f}', fontsize=11)
    axes[1, 2].axis('off')
    
    # 处理后差异直方图
    axes[1, 3].hist(diff_proc.flatten(), bins=50, alpha=0.7, color='green', label='处理后差异')
    axes[1, 3].axvline(x=detector.pixel_diff_threshold, color='red', linestyle='--', label='阈值')
    axes[1, 3].set_title('处理后差异分布', fontsize=11)
    axes[1, 3].legend()
    
    # 第三行：差异对比和分析
    diff_comparison = diff_proc.astype(np.float32) - diff_orig.astype(np.float32)
    axes[2, 0].imshow(diff_comparison, cmap='RdBu', vmin=-100, vmax=100)
    axes[2, 0].set_title('处理增加的差异\n(蓝色=减少, 红色=增加)', fontsize=11)
    axes[2, 0].axis('off')
    
    # 阈值掩码对比
    mask_orig = diff_orig > detector.pixel_diff_threshold
    mask_proc = diff_proc > detector.pixel_diff_threshold
    mask_diff = mask_proc.astype(np.uint8) - mask_orig.astype(np.uint8)
    
    axes[2, 1].imshow(mask_diff, cmap='RdBu', vmin=-1, vmax=1)
    axes[2, 1].set_title('阈值掩码变化\n(红色=新增假阳性)', fontsize=11)
    axes[2, 1].axis('off')
    
    # 统计分析
    orig_above_threshold = np.sum(mask_orig)
    proc_above_threshold = np.sum(mask_proc)
    false_positives = np.sum(mask_diff > 0)
    
    stats_text = f"""差异统计分析:

原始超阈值像素: {orig_above_threshold:,}
处理后超阈值像素: {proc_above_threshold:,}
新增假阳性像素: {false_positives:,}

比例变化:
原始: {orig_above_threshold/diff_orig.size*100:.2f}%
处理后: {proc_above_threshold/diff_proc.size*100:.2f}%

处理影响:
{'增加了假阳性' if false_positives > orig_above_threshold*0.1 else '影响较小'}
"""
    
    axes[2, 2].text(0.05, 0.95, stats_text, ha='left', va='top', transform=axes[2, 2].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('统计分析', fontsize=12)
    
    # 建议优化方案
    suggestions_text = f"""优化建议:

1. 调整阈值:
   建议阈值: {int(np.mean(diff_orig) + 2*np.std(diff_orig))}
   (当前: {detector.pixel_diff_threshold})

2. 处理参数优化:
   {'减少CLAHE强度' if np.mean(diff_proc) > np.mean(diff_orig)*1.2 else '参数合适'}

3. 配准精度:
   内点率: {transform_info.get('inlier_ratio', 0):.1%}
   {'建议提高配准精度' if transform_info.get('inlier_ratio', 0) < 0.6 else '配准精度良好'}

4. 后处理建议:
   应用形态学滤波
   区域连通性分析
"""
    
    axes[2, 3].text(0.05, 0.95, suggestions_text, ha='left', va='top', transform=axes[2, 3].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    axes[2, 3].set_title('优化建议', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 输出详细分析
    print(f"\n📊 详细分析结果:")
    print(f"   原始平均差异: {np.mean(diff_orig):.2f}")
    print(f"   处理后平均差异: {np.mean(diff_proc):.2f}")
    print(f"   差异增幅: {(np.mean(diff_proc)/np.mean(diff_orig)-1)*100:.1f}%")
    print(f"   新增假阳性像素: {false_positives:,} ({false_positives/diff_orig.size*100:.3f}%)")
    
    return {
        'original_diff': diff_orig,
        'processed_diff': diff_proc,
        'false_positives': false_positives,
        'suggestions': {
            'recommended_threshold': int(np.mean(diff_orig) + 2*np.std(diff_orig)),
            'current_threshold': detector.pixel_diff_threshold
        }
    }


def elastic_registration_with_similarity_matching(img1_path, img2_path):
    """
    基于相似性匹配的弹性配准方法
    将大面积相似的区域自动对齐，解决透视校正无法完全重叠的问题
    
    Args:
        img1_path (str): 第一期图像路径
        img2_path (str): 第二期图像路径
    """
    detector = IllegalBuildingDetector()
    
    print("=== 弹性配准相似性匹配 ===")
    
    # 加载图像
    img1 = detector.load_image_with_chinese_path(img1_path)
    img2 = detector.load_image_with_chinese_path(img2_path)
    
    if img1 is None or img2 is None:
        return
    
    # 先进行基础透视校正
    base_img, aligned_img, _, transform_info = detector.perspective_correction(img1, img2)
    
    print(f"初始透视校正内点率: {transform_info.get('inlier_ratio', 0):.2%}")
    
    # 转换为灰度图
    gray1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    
    # 1. 基于滑动窗口的局部相似性匹配
    def local_similarity_matching(img1, img2, window_size=64, stride=32, threshold=0.7):
        """
        局部相似性匹配，找到大面积相似区域
        """
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        matches = []
        similarity_map = np.zeros_like(img1, dtype=np.float32)
        
        print(f"  执行局部相似性匹配 (窗口: {window_size}×{window_size}, 步长: {stride})...")
        
        for y1 in range(0, h1 - window_size, stride):
            for x1 in range(0, w1 - window_size, stride):
                # 提取第一期图像的窗口
                window1 = img1[y1:y1+window_size, x1:x1+window_size]
                
                # 在第二期图像中搜索最佳匹配
                best_similarity = -1
                best_match = None
                
                # 搜索范围（允许一定的偏移）
                search_range = 50
                y2_start = max(0, y1 - search_range)
                y2_end = min(h2 - window_size, y1 + search_range)
                x2_start = max(0, x1 - search_range)
                x2_end = min(w2 - window_size, x1 + search_range)
                
                for y2 in range(y2_start, y2_end, stride//2):
                    for x2 in range(x2_start, x2_end, stride//2):
                        window2 = img2[y2:y2+window_size, x2:x2+window_size]
                        
                        # 计算归一化互相关
                        correlation = cv2.matchTemplate(window1, window2, cv2.TM_CCOEFF_NORMED)[0, 0]
                        
                        if correlation > best_similarity:
                            best_similarity = correlation
                            best_match = (x2 + window_size//2, y2 + window_size//2)
                
                # 记录高质量匹配
                if best_similarity > threshold:
                    center1 = (x1 + window_size//2, y1 + window_size//2)
                    matches.append((center1, best_match, best_similarity))
                    
                    # 更新相似性地图
                    similarity_map[y1:y1+window_size, x1:x1+window_size] = best_similarity
        
        print(f"  找到 {len(matches)} 个高质量局部匹配")
        return matches, similarity_map
    
    # 执行局部相似性匹配
    local_matches, similarity_map = local_similarity_matching(gray1, gray2)
    
    # 2. 基于相似区域的稠密光流计算
    def compute_dense_flow_from_matches(img1, img2, matches):
        """
        基于匹配点计算稠密光流场
        """
        print("  计算稠密光流场...")
        
        if len(matches) < 10:
            print("  匹配点不足，使用全局光流")
            # 使用Farneback光流作为备选
            flow = cv2.calcOpticalFlowPyrLK(img1, img2, None, None)
            return flow
        
        # 创建稀疏光流场
        h, w = img1.shape
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        # 基于匹配点插值光流
        for (x1, y1), (x2, y2), similarity in matches:
            dx = x2 - x1
            dy = y2 - y1
            
            # 在匹配点周围应用光流
            radius = 32
            y_min = max(0, int(y1) - radius)
            y_max = min(h, int(y1) + radius)
            x_min = max(0, int(x1) - radius)
            x_max = min(w, int(x1) + radius)
            
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    distance = np.sqrt((x - x1)**2 + (y - y1)**2)
                    if distance < radius:
                        weight = similarity * np.exp(-distance / radius)
                        flow_x[y, x] += dx * weight
                        flow_y[y, x] += dy * weight
                        weights[y, x] += weight
        
        # 归一化
        mask = weights > 0
        flow_x[mask] /= weights[mask]
        flow_y[mask] /= weights[mask]
        
        # 平滑光流场
        flow_x = cv2.GaussianBlur(flow_x, (15, 15), 5)
        flow_y = cv2.GaussianBlur(flow_y, (15, 15), 5)
        
        return np.stack([flow_x, flow_y], axis=2)
    
    # 计算光流场
    flow_field = compute_dense_flow_from_matches(gray1, gray2, local_matches)
    
    # 3. 应用弹性变形
    def apply_elastic_deformation(img, flow_field):
        """
        应用弹性变形
        """
        print("  应用弹性变形...")
        
        h, w = img.shape[:2]
        
        # 创建变形网格
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        if len(flow_field.shape) == 3:
            x_coords += flow_field[:, :, 0]
            y_coords += flow_field[:, :, 1]
        
        # 应用重映射
        if len(img.shape) == 3:
            warped = cv2.remap(img, x_coords, y_coords, cv2.INTER_LINEAR)
        else:
            warped = cv2.remap(img, x_coords, y_coords, cv2.INTER_LINEAR)
        
        return warped
    
    # 对第二期图像应用弹性变形
    if isinstance(flow_field, np.ndarray) and flow_field.size > 0:
        elastically_aligned = apply_elastic_deformation(aligned_img, flow_field)
        elastically_aligned_gray = apply_elastic_deformation(gray2, flow_field)
    else:
        print("  光流计算失败，使用原始校正结果")
        elastically_aligned = aligned_img
        elastically_aligned_gray = gray2
    
    # 4. 质量评估
    from skimage.metrics import structural_similarity as ssim
    
    # 计算改进后的相似性指标
    ssim_before = ssim(gray1, gray2)
    ssim_after = ssim(gray1, elastically_aligned_gray)
    
    print(f"\n📊 弹性配准效果:")
    print(f"   透视校正后 SSIM: {ssim_before:.4f}")
    print(f"   弹性配准后 SSIM: {ssim_after:.4f}")
    print(f"   改进幅度: {(ssim_after - ssim_before):.4f}")
    
    # 5. 创建可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('弹性配准相似性匹配结果', fontsize=16, fontweight='bold')
    
    # 第一行：原始对比
    axes[0, 0].imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('第一期图像（基准）', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'透视校正后\nSSIM: {ssim_before:.3f}', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(elastically_aligned, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'弹性配准后\nSSIM: {ssim_after:.3f}', fontsize=12)
    axes[0, 2].axis('off')
    
    # 相似性地图
    axes[0, 3].imshow(similarity_map, cmap='viridis')
    axes[0, 3].set_title(f'相似性地图\n匹配点: {len(local_matches)}', fontsize=12)
    axes[0, 3].axis('off')
    
    # 第二行：重叠检查
    # 透视校正重叠
    alpha_blend_before = cv2.addWeighted(base_img, 0.5, aligned_img, 0.5, 0)
    axes[1, 0].imshow(cv2.cvtColor(alpha_blend_before, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('透视校正重叠', fontsize=12)
    axes[1, 0].axis('off')
    
    # 弹性配准重叠
    alpha_blend_after = cv2.addWeighted(base_img, 0.5, elastically_aligned, 0.5, 0)
    axes[1, 1].imshow(cv2.cvtColor(alpha_blend_after, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('弹性配准重叠', fontsize=12)
    axes[1, 1].axis('off')
    
    # 差异对比
    diff_before = cv2.absdiff(gray1, gray2)
    diff_after = cv2.absdiff(gray1, elastically_aligned_gray)
    
    axes[1, 2].imshow(diff_before, cmap='hot')
    axes[1, 2].set_title(f'透视校正差异\n平均: {np.mean(diff_before):.1f}', fontsize=12)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(diff_after, cmap='hot')
    axes[1, 3].set_title(f'弹性配准差异\n平均: {np.mean(diff_after):.1f}', fontsize=12)
    axes[1, 3].axis('off')
    
    # 第三行：光流场和统计
    if isinstance(flow_field, np.ndarray) and len(flow_field.shape) == 3:
        # 光流场可视化
        flow_magnitude = np.sqrt(flow_field[:, :, 0]**2 + flow_field[:, :, 1]**2)
        axes[2, 0].imshow(flow_magnitude, cmap='jet')
        axes[2, 0].set_title('光流场强度', fontsize=12)
        axes[2, 0].axis('off')
        
        # 光流方向
        flow_angle = np.arctan2(flow_field[:, :, 1], flow_field[:, :, 0])
        axes[2, 1].imshow(flow_angle, cmap='hsv')
        axes[2, 1].set_title('光流方向', fontsize=12)
        axes[2, 1].axis('off')
    else:
        axes[2, 0].text(0.5, 0.5, '光流计算失败', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].axis('off')
        axes[2, 1].text(0.5, 0.5, '无光流数据', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].axis('off')
    
    # 统计信息
    improvement_text = f"""配准改进统计:

SSIM改进:
  透视校正: {ssim_before:.4f}
  弹性配准: {ssim_after:.4f}
  改进幅度: {ssim_after - ssim_before:+.4f}

差异减少:
  校正前: {np.mean(diff_before):.1f}
  校正后: {np.mean(diff_after):.1f}
  减少: {np.mean(diff_before) - np.mean(diff_after):.1f}

匹配质量:
  局部匹配: {len(local_matches)}
  平均相似性: {np.mean([m[2] for m in local_matches]) if local_matches else 0:.3f}

总体评估:
{'显著改进' if ssim_after - ssim_before > 0.1 else '适度改进' if ssim_after - ssim_before > 0.05 else '轻微改进'}
"""
    
    color = 'lightgreen' if ssim_after - ssim_before > 0.1 else 'lightyellow' if ssim_after - ssim_before > 0.05 else 'lightcoral'
    axes[2, 2].text(0.05, 0.95, improvement_text, ha='left', va='top', transform=axes[2, 2].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('改进统计', fontsize=12)
    
    # 技术说明
    tech_text = """弹性配准技术:

1. 局部相似性匹配
   • 滑动窗口搜索
   • 归一化互相关
   • 自适应阈值筛选

2. 稠密光流计算
   • 基于匹配点插值
   • 高斯权重衰减
   • 光流场平滑

3. 弹性变形应用
   • 双线性插值
   • 保持局部连续性
   • 避免过度变形

优势:
• 处理几何变形
• 保持相似区域对齐
• 减少假阳性差异
"""
    
    axes[2, 3].text(0.05, 0.95, tech_text, ha='left', va='top', transform=axes[2, 3].transAxes,
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    axes[2, 3].set_title('技术说明', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_img1': base_img,
        'original_img2': aligned_img,
        'elastically_aligned': elastically_aligned,
        'ssim_before': ssim_before,
        'ssim_after': ssim_after,
        'improvement': ssim_after - ssim_before,
        'local_matches': len(local_matches),
        'flow_field': flow_field
    }


def improved_elastic_registration_with_bbox_detection(img1_path, img2_path):
    """
    改进的弹性配准方法，解决黑色掩码问题并提供bbox检测
    
    Args:
        img1_path (str): 第一期图像路径
        img2_path (str): 第二期图像路径
    
    Returns:
        dict: 包含配准结果和bbox检测信息
    """
    detector = IllegalBuildingDetector()
    
    print("=== 改进弹性配准与目标检测 ===")
    
    # 加载图像
    img1 = detector.load_image_with_chinese_path(img1_path)
    img2 = detector.load_image_with_chinese_path(img2_path)
    
    if img1 is None or img2 is None:
        return None
    
    # 透视校正
    base_img, aligned_img, _, transform_info = detector.perspective_correction(img1, img2)
    
    # 1. 创建有效区域掩码，排除黑色边缘
    def create_valid_mask(img, threshold=10):
        """
        创建有效区域掩码，排除透视校正产生的黑色边缘
        
        Args:
            img: 输入图像
            threshold: 黑色区域阈值
        
        Returns:
            mask: 有效区域掩码
        """
        if len(img.shape) == 3:
            # 彩色图像：检查所有通道
            mask = np.any(img > threshold, axis=2)
        else:
            # 灰度图像
            mask = img > threshold
        
        # 形态学操作，清理掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask.astype(bool)
    
    # 创建两张图像的有效区域掩码
    mask1 = create_valid_mask(base_img)
    mask2 = create_valid_mask(aligned_img)
    
    # 计算公共有效区域
    common_mask = np.logical_and(mask1, mask2)
    
    print(f"  有效区域比例: {np.sum(common_mask) / common_mask.size * 100:.1f}%")
    
    # 2. 在有效区域内进行弹性配准
    def masked_similarity_matching(img1, img2, mask, window_size=64, stride=32, threshold=0.7):
        """
        在掩码区域内进行相似性匹配
        """
        h, w = img1.shape[:2]
        matches = []
        similarity_map = np.zeros_like(img1, dtype=np.float32)
        
        print(f"  在有效区域内执行相似性匹配...")
        
        for y1 in range(0, h - window_size, stride):
            for x1 in range(0, w - window_size, stride):
                # 检查窗口是否在有效区域内
                window_mask = mask[y1:y1+window_size, x1:x1+window_size]
                if np.sum(window_mask) < window_size * window_size * 0.8:  # 80%有效像素
                    continue
                
                window1 = img1[y1:y1+window_size, x1:x1+window_size]
                best_similarity = -1
                best_match = None
                
                search_range = 50
                y2_start = max(0, y1 - search_range)
                y2_end = min(h - window_size, y1 + search_range)
                x2_start = max(0, x1 - search_range)
                x2_end = min(w - window_size, x1 + search_range)
                
                for y2 in range(y2_start, y2_end, stride//2):
                    for x2 in range(x2_start, x2_end, stride//2):
                        # 检查搜索窗口是否在有效区域内
                        search_mask = mask[y2:y2+window_size, x2:x2+window_size]
                        if np.sum(search_mask) < window_size * window_size * 0.8:
                            continue
                        
                        window2 = img2[y2:y2+window_size, x2:x2+window_size]
                        
                        # 只在有效区域计算相关性
                        masked_window1 = window1 * window_mask[:,:,np.newaxis] if len(window1.shape)==3 else window1 * window_mask
                        masked_window2 = window2 * search_mask[:,:,np.newaxis] if len(window2.shape)==3 else window2 * search_mask
                        
                        if len(masked_window1.shape) == 3:
                            masked_window1 = cv2.cvtColor(masked_window1, cv2.COLOR_BGR2GRAY)
                            masked_window2 = cv2.cvtColor(masked_window2, cv2.COLOR_BGR2GRAY)
                        
                        correlation = cv2.matchTemplate(masked_window1, masked_window2, cv2.TM_CCOEFF_NORMED)[0, 0]
                        
                        if correlation > best_similarity:
                            best_similarity = correlation
                            best_match = (x2 + window_size//2, y2 + window_size//2)
                
                if best_similarity > threshold:
                    center1 = (x1 + window_size//2, y1 + window_size//2)
                    matches.append((center1, best_match, best_similarity))
                    similarity_map[y1:y1+window_size, x1:x1+window_size] = best_similarity
        
        print(f"  找到 {len(matches)} 个有效区域内的匹配")
        return matches, similarity_map
    
    # 转换为灰度图
    gray1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    
    # 执行掩码内的相似性匹配
    local_matches, similarity_map = masked_similarity_matching(gray1, gray2, common_mask)
    
    # 3. 计算光流并应用弹性变形
    def compute_masked_dense_flow(img1, img2, matches, mask):
        """
        在掩码区域内计算稠密光流
        """
        if len(matches) < 5:
            print("  匹配点不足，跳过弹性变形")
            return None
        
        h, w = img1.shape
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        for (x1, y1), (x2, y2), similarity in matches:
            dx = x2 - x1
            dy = y2 - y1
            
            radius = 40
            y_min = max(0, int(y1) - radius)
            y_max = min(h, int(y1) + radius)
            x_min = max(0, int(x1) - radius)
            x_max = min(w, int(x1) + radius)
            
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if mask[y, x]:  # 只在有效区域内应用
                        distance = np.sqrt((x - x1)**2 + (y - y1)**2)
                        if distance < radius:
                            weight = similarity * np.exp(-distance / radius)
                            flow_x[y, x] += dx * weight
                            flow_y[y, x] += dy * weight
                            weights[y, x] += weight
        
        # 归一化
        valid_flow = weights > 0
        flow_x[valid_flow] /= weights[valid_flow]
        flow_y[valid_flow] /= weights[valid_flow]
        
        # 只在有效区域内平滑
        flow_x = cv2.GaussianBlur(flow_x, (15, 15), 5)
        flow_y = cv2.GaussianBlur(flow_y, (15, 15), 5)
        
        return np.stack([flow_x, flow_y], axis=2)
    
    # 计算光流场
    flow_field = compute_masked_dense_flow(gray1, gray2, local_matches, common_mask)
    
    # 应用弹性变形
    if flow_field is not None:
        h, w = aligned_img.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        x_coords += flow_field[:, :, 0]
        y_coords += flow_field[:, :, 1]
        
        elastically_aligned = cv2.remap(aligned_img, x_coords, y_coords, cv2.INTER_LINEAR)
        elastically_aligned_gray = cv2.remap(gray2, x_coords, y_coords, cv2.INTER_LINEAR)
    else:
        elastically_aligned = aligned_img
        elastically_aligned_gray = gray2
    
    # 4. 在有效区域内计算差异
    def compute_masked_difference(img1, img2, mask):
        """
        在掩码区域内计算差异
        """
        diff = cv2.absdiff(img1, img2)
        # 将无效区域设为0
        diff[~mask] = 0
        return diff
    
    # 计算掩码差异
    masked_diff = compute_masked_difference(gray1, elastically_aligned_gray, common_mask)
    
    # 5. 差异区域检测和bbox提取
    def detect_difference_bboxes(diff_img, mask, min_area=500, threshold=50):
        """
        检测差异区域并返回bbox信息
        
        Args:
            diff_img: 差异图像
            mask: 有效区域掩码
            min_area: 最小区域面积
            threshold: 差异阈值
        
        Returns:
            list: bbox信息列表，每个包含 [x, y, w, h, confidence, area]
        """
        print(f"  检测差异区域...")
        
        # 二值化差异图像
        binary_diff = (diff_img > threshold).astype(np.uint8)
        binary_diff = binary_diff * mask.astype(np.uint8)  # 只保留有效区域
        
        # 形态学操作，连接相近的差异区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, 
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算置信度（基于区域内的平均差异）
            roi_diff = diff_img[y:y+h, x:x+w]
            roi_mask = mask[y:y+h, x:x+w]
            
            if np.sum(roi_mask) == 0:
                continue
                
            avg_diff = np.mean(roi_diff[roi_mask])
            max_diff = np.max(roi_diff[roi_mask])
            
            # 置信度计算：结合平均差异和最大差异
            confidence = min(1.0, (avg_diff / 255.0) * 0.7 + (max_diff / 255.0) * 0.3)
            
            bboxes.append({
                'id': i + 1,
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(confidence),
                'area': float(area),
                'avg_difference': float(avg_diff),
                'max_difference': float(max_diff),
                'center': [int(x + w/2), int(y + h/2)]
            })
        
        # 按置信度排序
        bboxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"  检测到 {len(bboxes)} 个差异区域")
        return bboxes
    
    # 检测差异区域
    detected_bboxes = detect_difference_bboxes(masked_diff, common_mask)
    
    # 6. 质量评估
    from skimage.metrics import structural_similarity as ssim
    
    # 只在有效区域计算SSIM
    ssim_before = ssim(gray1 * common_mask, gray2 * common_mask)
    ssim_after = ssim(gray1 * common_mask, elastically_aligned_gray * common_mask)
    
    print(f"\n📊 改进配准效果:")
    print(f"   有效区域SSIM: {ssim_before:.4f} → {ssim_after:.4f}")
    print(f"   改进幅度: {ssim_after - ssim_before:+.4f}")
    print(f"   检测区域数: {len(detected_bboxes)}")
    
    # 7. 创建可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('改进弹性配准与目标检测结果', fontsize=16, fontweight='bold')
    
    # 第一行：掩码和配准对比
    axes[0, 0].imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('第一期图像（基准）', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('透视校正后（有黑边）', fontsize=12)
    axes[0, 1].axis('off')
    
    # 显示有效区域掩码
    axes[0, 2].imshow(common_mask, cmap='gray')
    axes[0, 2].set_title(f'有效区域掩码\n覆盖率: {np.sum(common_mask)/common_mask.size*100:.1f}%', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(cv2.cvtColor(elastically_aligned, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title(f'弹性配准后\nSSIM: {ssim_after:.3f}', fontsize=12)
    axes[0, 3].axis('off')
    
    # 第二行：差异检测
    axes[1, 0].imshow(masked_diff, cmap='hot')
    axes[1, 0].set_title(f'掩码差异图\n平均差异: {np.mean(masked_diff[common_mask]):.1f}', fontsize=12)
    axes[1, 0].axis('off')
    
    # 绘制检测结果
    detection_img = elastically_aligned.copy()
    for bbox_info in detected_bboxes:
        x, y, w, h = bbox_info['bbox']
        confidence = bbox_info['confidence']
        
        # 根据置信度选择颜色
        if confidence > 0.8:
            color = (0, 0, 255)  # 红色：高置信度
        elif confidence > 0.6:
            color = (0, 165, 255)  # 橙色：中等置信度
        else:
            color = (0, 255, 255)  # 黄色：低置信度
        
        cv2.rectangle(detection_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(detection_img, f"ID:{bbox_info['id']} ({confidence:.2f})", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    axes[1, 1].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'目标检测结果\n检测数: {len(detected_bboxes)}', fontsize=12)
    axes[1, 1].axis('off')
    
    # 置信度分布
    if detected_bboxes:
        confidences = [b['confidence'] for b in detected_bboxes]
        axes[1, 2].hist(confidences, bins=10, alpha=0.7, color='blue')
        axes[1, 2].set_title('置信度分布', fontsize=12)
        axes[1, 2].set_xlabel('置信度')
        axes[1, 2].set_ylabel('数量')
    else:
        axes[1, 2].text(0.5, 0.5, '无检测结果', ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=14)
        axes[1, 2].set_title('置信度分布', fontsize=12)
    
    # 面积分布
    if detected_bboxes:
        areas = [b['area'] for b in detected_bboxes]
        axes[1, 3].hist(areas, bins=10, alpha=0.7, color='green')
        axes[1, 3].set_title('区域面积分布', fontsize=12)
        axes[1, 3].set_xlabel('面积 (像素)')
        axes[1, 3].set_ylabel('数量')
    else:
        axes[1, 3].text(0.5, 0.5, '无检测结果', ha='center', va='center', 
                       transform=axes[1, 3].transAxes, fontsize=14)
        axes[1, 3].set_title('区域面积分布', fontsize=12)
    
    # 第三行：详细信息
    # 检测统计
    if detected_bboxes:
        high_conf = len([b for b in detected_bboxes if b['confidence'] > 0.8])
        med_conf = len([b for b in detected_bboxes if 0.6 < b['confidence'] <= 0.8])
        low_conf = len([b for b in detected_bboxes if b['confidence'] <= 0.6])
        
        stats_text = f"""检测统计信息:

总检测数: {len(detected_bboxes)}
高置信度 (>0.8): {high_conf}
中等置信度 (0.6-0.8): {med_conf}
低置信度 (≤0.6): {low_conf}

平均置信度: {np.mean([b['confidence'] for b in detected_bboxes]):.3f}
平均面积: {np.mean([b['area'] for b in detected_bboxes]):.0f} px²

最大差异区域:
ID: {detected_bboxes[0]['id']}
置信度: {detected_bboxes[0]['confidence']:.3f}
面积: {detected_bboxes[0]['area']:.0f} px²
"""
    else:
        stats_text = """检测统计信息:

未检测到显著差异区域

可能原因:
• 图像变化很小
• 差异阈值过高
• 最小区域面积过大
• 配准质量很好
"""
    
    color = 'lightcoral' if len(detected_bboxes) > 3 else 'lightyellow' if len(detected_bboxes) > 0 else 'lightgreen'
    axes[2, 0].text(0.05, 0.95, stats_text, ha='left', va='top', transform=axes[2, 0].transAxes,
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    axes[2, 0].set_xlim(0, 1)
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].axis('off')
    axes[2, 0].set_title('检测统计', fontsize=12)
    
    # 技术改进说明
    tech_text = """技术改进要点:

1. 黑色掩码处理:
   • 自动检测有效区域
   • 排除透视校正黑边
   • 只在有效区域计算差异

2. 弹性配准优化:
   • 掩码内相似性匹配
   • 有效区域光流计算
   • 避免黑边干扰

3. 目标检测功能:
   • 差异区域分割
   • bbox坐标提取
   • 置信度计算

4. 结果可靠性:
   • 形态学后处理
   • 面积阈值过滤
   • 多指标评估
"""
    
    axes[2, 1].text(0.05, 0.95, tech_text, ha='left', va='top', transform=axes[2, 1].transAxes,
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    axes[2, 1].set_title('技术改进', fontsize=12)
    
    # bbox详细信息
    if detected_bboxes:
        bbox_text = "检测到的区域详情:\n\n"
        for i, bbox in enumerate(detected_bboxes[:5]):  # 显示前5个
            bbox_text += f"区域 {bbox['id']}:\n"
            bbox_text += f"  位置: ({bbox['bbox'][0]}, {bbox['bbox'][1]})\n"
            bbox_text += f"  尺寸: {bbox['bbox'][2]}×{bbox['bbox'][3]}\n"
            bbox_text += f"  置信度: {bbox['confidence']:.3f}\n"
            bbox_text += f"  面积: {bbox['area']:.0f} px²\n\n"
        
        if len(detected_bboxes) > 5:
            bbox_text += f"... 还有 {len(detected_bboxes)-5} 个区域"
    else:
        bbox_text = "未检测到差异区域\n\n这可能意味着:\n• 图像配准质量很好\n• 两期图像变化很小\n• 需要调整检测参数"
    
    axes[2, 2].text(0.05, 0.95, bbox_text, ha='left', va='top', transform=axes[2, 2].transAxes,
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('区域详情', fontsize=12)
    
    # 参数设置
    params_text = f"""检测参数设置:

差异阈值: 50
最小区域面积: 500 px²
形态学核: 7×7 椭圆
有效区域阈值: 10

配准参数:
窗口大小: 64×64
步长: 32
相似度阈值: 0.7
搜索范围: ±50 px

质量指标:
有效区域比例: {np.sum(common_mask)/common_mask.size*100:.1f}%
SSIM改进: {ssim_after-ssim_before:+.4f}
匹配点数: {len(local_matches)}
"""
    
    axes[2, 3].text(0.05, 0.95, params_text, ha='left', va='top', transform=axes[2, 3].transAxes,
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    axes[2, 3].set_title('参数设置', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_img1': base_img,
        'original_img2': aligned_img,
        'elastically_aligned': elastically_aligned,
        'valid_mask': common_mask,
        'masked_difference': masked_diff,
        'detected_bboxes': detected_bboxes,
        'ssim_before': ssim_before,
        'ssim_after': ssim_after,
        'improvement': ssim_after - ssim_before,
        'valid_area_ratio': np.sum(common_mask) / common_mask.size,
        'local_matches': len(local_matches),
        'detection_count': len(detected_bboxes)
    }