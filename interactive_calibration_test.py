#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式图像配准和形态学操作测试演示系统
基于Detection_of_unauthorized_building_works.py
支持实时参数调整和对应关系记录
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time


class InteractiveCalibrationTester:
    """交互式配准测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.img1 = None
        self.img2 = None
        self.img1_path = ""
        self.img2_path = ""
        
        # 配准参数
        self.calibration_params = {
            'detector_type': 'SIFT',
            'nfeatures': 1000,
            'contrast_threshold': 0.04,
            'edge_threshold': 10,
            'ratio_threshold': 0.7,
            'ransac_threshold': 5.0,
            'min_matches': 8
        }
        
        # 形态学操作参数
        self.morphology_params = {
            'enable_morphology': True,
            'kernel_type': 'MORPH_RECT',  # MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS
            'kernel_size': 5,
            'erosion_iterations': 1,
            'dilation_iterations': 1,
            'opening_kernel_size': 3,
            'closing_kernel_size': 3,
            'enable_opening': True,
            'enable_closing': True
        }
        
        # 结果存储
        self.results_history = []
        self.current_result = None
        
        # 输出目录
        self.output_dir = "interactive_calibration_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("交互式配准测试器初始化完成")
        print(f"输出目录: {self.output_dir}")
    
    def load_image_with_chinese_path(self, path):
        """加载包含中文路径的图像"""
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            return img
        except Exception as e:
            print(f"读取图像失败 {path}: {str(e)}")
            return None
    
    def create_detector(self, detector_type, params):
        """根据参数创建特征检测器"""
        if detector_type == 'SIFT':
            return cv2.SIFT_create(
                nfeatures=params['nfeatures'],
                contrastThreshold=params['contrast_threshold'],
                edgeThreshold=params['edge_threshold']
            )
        elif detector_type == 'ORB':
            return cv2.ORB_create(
                nfeatures=params['nfeatures'],
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=params['edge_threshold']
            )
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create(
                threshold=params['contrast_threshold'] / 100,  # 转换为AKAZE的阈值范围
                nOctaves=4,
                nOctaveLayers=4
            )
        else:
            return cv2.SIFT_create()
    
    def apply_morphology_operations(self, image, params):
        """应用形态学操作"""
        if not params['enable_morphology']:
            return image
        
        result = image.copy()
        
        # 创建核
        kernel_type_map = {
            'MORPH_RECT': cv2.MORPH_RECT,
            'MORPH_ELLIPSE': cv2.MORPH_ELLIPSE,
            'MORPH_CROSS': cv2.MORPH_CROSS
        }
        
        kernel_type = kernel_type_map.get(params['kernel_type'], cv2.MORPH_RECT)
        
        # 腐蚀操作
        if params['erosion_iterations'] > 0:
            kernel = cv2.getStructuringElement(kernel_type, (params['kernel_size'], params['kernel_size']))
            result = cv2.erode(result, kernel, iterations=params['erosion_iterations'])
        
        # 膨胀操作
        if params['dilation_iterations'] > 0:
            kernel = cv2.getStructuringElement(kernel_type, (params['kernel_size'], params['kernel_size']))
            result = cv2.dilate(result, kernel, iterations=params['dilation_iterations'])
        
        # 开运算
        if params['enable_opening'] and params['opening_kernel_size'] > 0:
            kernel = cv2.getStructuringElement(kernel_type, 
                                             (params['opening_kernel_size'], params['opening_kernel_size']))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        # 闭运算
        if params['enable_closing'] and params['closing_kernel_size'] > 0:
            kernel = cv2.getStructuringElement(kernel_type, 
                                             (params['closing_kernel_size'], params['closing_kernel_size']))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def perform_calibration_with_morphology(self):
        """执行角点匹配对齐然后形态学操作的配准"""
        if self.img1 is None or self.img2 is None:
            print("请先加载两张图像")
            return None
        
        print("执行角点匹配对齐和形态学操作...")
        
        # 1. 转换为灰度图
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        
        # 2. 首先进行角点匹配和图像对齐
        print("步骤1: 角点匹配和图像对齐...")
        detector = self.create_detector(self.calibration_params['detector_type'], self.calibration_params)
        
        # 检测原始图像的特征点
        kp1_orig, des1_orig = detector.detectAndCompute(gray1, None)
        kp2_orig, des2_orig = detector.detectAndCompute(gray2, None)
        
        print(f"原始特征点: {len(kp1_orig)}/{len(kp2_orig)}")
        
        if des1_orig is None or des2_orig is None or len(kp1_orig) < 4 or len(kp2_orig) < 4:
            print("特征点不足，无法进行配准")
            return None
        
        # 3. 特征匹配进行图像对齐
        print("步骤2: 特征匹配...")
        matcher = cv2.BFMatcher(cv2.NORM_L2 if self.calibration_params['detector_type'] == 'SIFT' else cv2.NORM_HAMMING, 
                               crossCheck=False)
        
        matches_orig = matcher.knnMatch(des1_orig, des2_orig, k=2)
        
        # 4. 应用比值测试筛选匹配
        def filter_matches(matches, ratio_threshold):
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            return good_matches
        
        good_matches_orig = filter_matches(matches_orig, self.calibration_params['ratio_threshold'])
        print(f"原始良好匹配: {len(good_matches_orig)}")
        
        # 5. 计算单应性矩阵进行图像对齐
        print("步骤3: 计算透视变换矩阵...")
        if len(good_matches_orig) < self.calibration_params['min_matches']:
            print("匹配点不足，无法进行配准")
            return None
        
        src_pts = np.float32([kp1_orig[m.queryIdx].pt for m in good_matches_orig]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2_orig[m.trainIdx].pt for m in good_matches_orig]).reshape(-1, 1, 2)
        
        M_orig, mask_orig = cv2.findHomography(src_pts, dst_pts, 
                                             cv2.RANSAC, 
                                             ransacReprojThreshold=self.calibration_params['ransac_threshold'])
        
        inlier_ratio_orig = np.sum(mask_orig) / len(mask_orig) if mask_orig is not None else 0
        print(f"原始内点比例: {inlier_ratio_orig:.3f}")
        
        # 6. 应用透视变换对齐图像
        print("步骤4: 应用透视变换对齐图像...")
        h, w = self.img1.shape[:2]
        aligned_img2_orig = cv2.warpPerspective(self.img2, M_orig, (w, h)) if M_orig is not None else self.img2
        aligned_gray2_orig = cv2.warpPerspective(gray2, M_orig, (w, h)) if M_orig is not None else gray2
        
        # 7. 对对齐后的图像应用形态学操作
        print("步骤5: 对对齐后的图像应用形态学操作...")
        processed_gray1 = self.apply_morphology_operations(gray1, self.morphology_params)
        processed_aligned_gray2 = self.apply_morphology_operations(aligned_gray2_orig, self.morphology_params)
        
        # 8. 在形态学处理后的图像上重新检测特征点
        print("步骤6: 在形态学处理后重新检测特征点...")
        kp1_proc, des1_proc = detector.detectAndCompute(processed_gray1, None)
        kp2_proc, des2_proc = detector.detectAndCompute(processed_aligned_gray2, None)
        
        print(f"形态学处理后特征点: {len(kp1_proc)}/{len(kp2_proc)}")
        
        # 9. 重新匹配形态学处理后的特征点
        matches_proc = []
        good_matches_proc = []
        inlier_ratio_proc = 0
        M_proc = M_orig  # 使用原始的变换矩阵
        
        if des1_proc is not None and des2_proc is not None and len(kp1_proc) >= 4 and len(kp2_proc) >= 4:
            matches_proc = matcher.knnMatch(des1_proc, des2_proc, k=2)
            good_matches_proc = filter_matches(matches_proc, self.calibration_params['ratio_threshold'])
            print(f"形态学处理后良好匹配: {len(good_matches_proc)}")
            
            if len(good_matches_proc) >= self.calibration_params['min_matches']:
                src_pts_proc = np.float32([kp1_proc[m.queryIdx].pt for m in good_matches_proc]).reshape(-1, 1, 2)
                dst_pts_proc = np.float32([kp2_proc[m.trainIdx].pt for m in good_matches_proc]).reshape(-1, 1, 2)
                
                M_proc, mask_proc = cv2.findHomography(src_pts_proc, dst_pts_proc, 
                                                     cv2.RANSAC, 
                                                     ransacReprojThreshold=self.calibration_params['ransac_threshold'])
                
                inlier_ratio_proc = np.sum(mask_proc) / len(mask_proc) if mask_proc is not None else 0
                print(f"形态学处理后内点比例: {inlier_ratio_proc:.3f}")
        
        # 10. 生成最终对齐结果
        aligned_img2_proc = cv2.warpPerspective(self.img2, M_proc, (w, h)) if M_proc is not None else aligned_img2_orig
        
        # 11. 计算配准质量指标
        def calculate_alignment_quality(img1, img2):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # 结构相似性
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_score = ssim(gray1, gray2)
            except ImportError:
                # 如果没有skimage，使用简单的相关系数
                ssim_score = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
            
            # 均方误差
            mse = np.mean((gray1.astype(np.float32) - gray2.astype(np.float32)) ** 2)
            
            return ssim_score, mse
        
        ssim_orig, mse_orig = calculate_alignment_quality(self.img1, aligned_img2_orig)
        ssim_proc, mse_proc = calculate_alignment_quality(self.img1, aligned_img2_proc)
        
        # 12. 整理结果
        print(f"配准处理完成")
        print(f"角点移除效果: {len(kp1_orig)} -> {len(kp1_proc)} (移除 {len(kp1_orig) - len(kp1_proc)} 个)")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'calibration_params': self.calibration_params.copy(),
            'morphology_params': self.morphology_params.copy(),
            'workflow': 'corner_matching_then_morphology',  # 标记工作流程
            'original': {
                'keypoints': len(kp1_orig),
                'matches': len(good_matches_orig),
                'inlier_ratio': float(inlier_ratio_orig),
                'ssim': float(ssim_orig),
                'mse': float(mse_orig),
                'homography_matrix': M_orig.tolist() if M_orig is not None else None
            },
            'processed': {
                'keypoints': len(kp1_proc),
                'matches': len(good_matches_proc),
                'inlier_ratio': float(inlier_ratio_proc),
                'ssim': float(ssim_proc),
                'mse': float(mse_proc),
                'homography_matrix': M_proc.tolist() if M_proc is not None else None
            },
            'improvement': {
                'keypoint_change': len(kp1_proc) - len(kp1_orig),
                'match_change': len(good_matches_proc) - len(good_matches_orig),
                'inlier_ratio_improvement': float(inlier_ratio_proc - inlier_ratio_orig),
                'ssim_improvement': float(ssim_proc - ssim_orig),
                'mse_improvement': float(mse_orig - mse_proc)
            },
            'corner_removal_effect': {
                'corners_before_morphology': len(kp1_orig),
                'corners_after_morphology': len(kp1_proc),
                'corners_removed': len(kp1_orig) - len(kp1_proc),
                'removal_percentage': (len(kp1_orig) - len(kp1_proc)) / len(kp1_orig) * 100 if len(kp1_orig) > 0 else 0
            }
        }
        
        # 存储可视化数据
        result['visualization_data'] = {
            'original_img1': self.img1,
            'original_img2': self.img2,
            'aligned_img2_orig': aligned_img2_orig,  # 仅角点对齐的结果
            'processed_gray1': processed_gray1,
            'processed_aligned_gray2': processed_aligned_gray2,
            'aligned_img2_proc': aligned_img2_proc,  # 角点对齐+形态学处理的结果
            'keypoints_orig': (kp1_orig, kp2_orig),
            'keypoints_proc': (kp1_proc, kp2_proc),
            'matches_orig': good_matches_orig,
            'matches_proc': good_matches_proc,
            'corner_removal_visualization': {
                'before_morphology': len(kp1_orig),
                'after_morphology': len(kp1_proc),
                'removed_corners': len(kp1_orig) - len(kp1_proc)
            }
        }
        
        self.current_result = result
        return result
    
    def create_visualization(self, result):
        """创建可视化图像"""
        if result is None:
            return None
        
        viz_data = result['visualization_data']
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('角点匹配对齐 + 形态学操作效果对比 (包含重叠验证)', fontsize=16, fontweight='bold')
        
        # 创建重叠图像用于验证对齐效果
        def create_overlap_images():
            """创建各种重叠图像来验证对齐效果"""
            # 1. 原始图像重叠 (显示对齐前的差异)
            original_overlap = cv2.addWeighted(viz_data['original_img1'], 0.5, viz_data['original_img2'], 0.5, 0)
            
            # 2. 角点对齐后重叠 (显示角点匹配的对齐效果)
            aligned_overlap = cv2.addWeighted(viz_data['original_img1'], 0.5, viz_data['aligned_img2_orig'], 0.5, 0)
            
            # 3. 形态学处理后重叠 (显示最终效果)
            final_overlap = cv2.addWeighted(viz_data['original_img1'], 0.5, viz_data['aligned_img2_proc'], 0.5, 0)
            
            # 4. 红绿重叠 - 角点对齐后 (红色=图像1，绿色=图像2，黄色=重合区域)
            red_green_aligned = np.zeros_like(viz_data['original_img1'])
            gray1 = cv2.cvtColor(viz_data['original_img1'], cv2.COLOR_BGR2GRAY)
            gray2_aligned = cv2.cvtColor(viz_data['aligned_img2_orig'], cv2.COLOR_BGR2GRAY)
            red_green_aligned[:, :, 0] = gray2_aligned  # 蓝色通道
            red_green_aligned[:, :, 1] = gray1  # 绿色通道 
            red_green_aligned[:, :, 2] = gray2_aligned  # 红色通道
            
            # 5. 红绿重叠 - 形态学处理后
            red_green_final = np.zeros_like(viz_data['original_img1'])
            gray1_proc = viz_data['processed_gray1']
            gray2_proc = viz_data['processed_aligned_gray2']
            red_green_final[:, :, 0] = gray2_proc  # 蓝色通道
            red_green_final[:, :, 1] = gray1_proc  # 绿色通道
            red_green_final[:, :, 2] = gray2_proc  # 红色通道
            
            # 6. 棋盘格重叠 - 用于精确验证对齐
            def create_checkerboard_overlay(img1, img2, block_size=50):
                overlay = img1.copy()
                h, w = overlay.shape[:2]
                for i in range(0, h, block_size):
                    for j in range(0, w, block_size):
                        if (i // block_size + j // block_size) % 2 == 1:
                            i_end = min(i + block_size, h)
                            j_end = min(j + block_size, w)
                            overlay[i:i_end, j:j_end] = img2[i:i_end, j:j_end]
                return overlay
            
            checkerboard_aligned = create_checkerboard_overlay(viz_data['original_img1'], viz_data['aligned_img2_orig'])
            checkerboard_final = create_checkerboard_overlay(viz_data['original_img1'], viz_data['aligned_img2_proc'])
            
            return {
                'original_overlap': original_overlap,
                'aligned_overlap': aligned_overlap,
                'final_overlap': final_overlap,
                'red_green_aligned': red_green_aligned,
                'red_green_final': red_green_final,
                'checkerboard_aligned': checkerboard_aligned,
                'checkerboard_final': checkerboard_final
            }
        
        overlap_images = create_overlap_images()
        
        # 第一行：原始图像和重叠对比
        axes[0, 0].imshow(cv2.cvtColor(viz_data['original_img1'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('基准图像 (图像1)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(viz_data['original_img2'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('待对齐图像 (图像2)', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(overlap_images['original_overlap'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('原始重叠 (对齐前)', fontsize=12)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(cv2.cvtColor(overlap_images['aligned_overlap'], cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title('角点匹配对齐后重叠', fontsize=12)
        axes[0, 3].axis('off')
        
        # 第二行：红绿重叠和棋盘格验证
        axes[1, 0].imshow(cv2.cvtColor(overlap_images['red_green_aligned'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('红绿重叠 - 角点对齐后\n(红=图2, 绿=图1, 黄=重合)', fontsize=11)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(overlap_images['red_green_final'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('红绿重叠 - 形态学处理后\n(红=图2, 绿=图1, 黄=重合)', fontsize=11)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(overlap_images['checkerboard_aligned'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('棋盘格重叠 - 角点对齐后\n(用于精确验证对齐)', fontsize=11)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(cv2.cvtColor(overlap_images['checkerboard_final'], cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title('棋盘格重叠 - 形态学处理后\n(最终对齐效果)', fontsize=11)
        axes[1, 3].axis('off')
        
        # 第三行：特征点检测对比
        kp1_orig, kp2_orig = viz_data['keypoints_orig']
        kp1_proc, kp2_proc = viz_data['keypoints_proc']
        
        img1_kp_orig = cv2.drawKeypoints(viz_data['original_img1'], kp1_orig, None, color=(0, 255, 0), flags=0)
        img1_kp_proc = cv2.drawKeypoints(viz_data['original_img1'], kp1_proc, None, color=(255, 0, 0), flags=0)
        
        axes[2, 0].imshow(cv2.cvtColor(img1_kp_orig, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title(f'原始特征点: {len(kp1_orig)}', fontsize=12)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(cv2.cvtColor(img1_kp_proc, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title(f'处理后特征点: {len(kp1_proc)}', fontsize=12)
        axes[2, 1].axis('off')
        
        # 形态学处理后的图像
        axes[2, 2].imshow(viz_data['processed_gray1'], cmap='gray')
        axes[2, 2].set_title('形态学处理后 (图像1)', fontsize=12)
        axes[2, 2].axis('off')
        
        axes[2, 3].imshow(viz_data['processed_aligned_gray2'], cmap='gray')
        axes[2, 3].set_title('形态学处理后 (对齐的图像2)', fontsize=12)
        axes[2, 3].axis('off')
        
        # 第四行：配准结果和统计信息
        axes[3, 0].imshow(cv2.cvtColor(viz_data['aligned_img2_orig'], cv2.COLOR_BGR2RGB))
        axes[3, 0].set_title(f'角点匹配对齐结果\nSSIM: {result["original"]["ssim"]:.3f}', fontsize=12)
        axes[3, 0].axis('off')
        
        axes[3, 1].imshow(cv2.cvtColor(viz_data['aligned_img2_proc'], cv2.COLOR_BGR2RGB))
        axes[3, 1].set_title(f'形态学处理后结果\nSSIM: {result["processed"]["ssim"]:.3f}', fontsize=12)
        axes[3, 1].axis('off')
        
        # 工作流程和参数显示
        param_text = f"""工作流程: 角点匹配对齐 → 形态学操作

配准参数:
检测器: {result['calibration_params']['detector_type']}
特征点数: {result['calibration_params']['nfeatures']}
对比度阈值: {result['calibration_params']['contrast_threshold']}
边缘阈值: {result['calibration_params']['edge_threshold']}
比值阈值: {result['calibration_params']['ratio_threshold']}
RANSAC阈值: {result['calibration_params']['ransac_threshold']}

形态学参数:
启用: {result['morphology_params']['enable_morphology']}
核类型: {result['morphology_params']['kernel_type']}
核大小: {result['morphology_params']['kernel_size']}
腐蚀迭代: {result['morphology_params']['erosion_iterations']}
膨胀迭代: {result['morphology_params']['dilation_iterations']}
开运算: {result['morphology_params']['enable_opening']}
闭运算: {result['morphology_params']['enable_closing']}"""
        
        axes[3, 2].text(0.05, 0.95, param_text, ha='left', va='top', 
                        transform=axes[3, 2].transAxes, fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        axes[3, 2].set_xlim(0, 1)
        axes[3, 2].set_ylim(0, 1)
        axes[3, 2].axis('off')
        axes[3, 2].set_title('当前参数', fontsize=12)
        
        # 角点移除效果和性能对比
        improvement = result['improvement']
        corner_effect = result['corner_removal_effect']
        stats_text = f"""角点移除效果:

对齐前角点: {corner_effect['corners_before_morphology']}
形态学后角点: {corner_effect['corners_after_morphology']}
移除角点数: {corner_effect['corners_removed']}
移除比例: {corner_effect['removal_percentage']:.1f}%

性能对比:
匹配数变化: {improvement['match_change']:+d}
内点率改进: {improvement['inlier_ratio_improvement']:+.3f}
SSIM改进: {improvement['ssim_improvement']:+.3f}
MSE改进: {improvement['mse_improvement']:+.1f}

配准质量:
原始SSIM: {result['original']['ssim']:.3f}
处理后SSIM: {result['processed']['ssim']:.3f}

结论: {'形态学操作移除了无效角点' if corner_effect['corners_removed'] > 0 else '未移除角点'}
     {'并提升了配准质量' if improvement['ssim_improvement'] > 0 else '但配准质量未显著提升'}"""
        
        color = 'lightgreen' if improvement['ssim_improvement'] > 0 else 'lightcoral'
        axes[3, 3].text(0.05, 0.95, stats_text, ha='left', va='top', 
                        transform=axes[3, 3].transAxes, fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        axes[3, 3].set_xlim(0, 1)
        axes[3, 3].set_ylim(0, 1)
        axes[3, 3].axis('off')
        axes[3, 3].set_title('效果统计', fontsize=12)
        
        # 添加重叠验证说明
        overlap_explanation = """重叠验证说明:

1. 原始重叠: 显示对齐前的错位
2. 半透明重叠: 50%透明度混合
3. 红绿重叠: 错位=彩色，重合=灰色
4. 棋盘格: 精确验证对齐质量

观察要点:
• 重叠图中的彩色边缘越少越好
• 棋盘格中的连续性越好越好
• 红绿图中的黄色区域越多越好"""
        
        # 在空白区域添加说明（如果有的话，可以覆盖一些不重要的区域）
        fig.text(0.02, 0.02, overlap_explanation, fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.output_dir, f"calibration_morphology_test_{timestamp}.jpg")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return image_path
    
    def save_result(self, result):
        """保存测试结果"""
        if result is None:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 移除不可序列化的数据
        save_result = result.copy()
        if 'visualization_data' in save_result:
            del save_result['visualization_data']
        
        # 保存JSON结果
        json_path = os.path.join(self.output_dir, f"calibration_result_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        
        # 添加到历史记录
        self.results_history.append(save_result)
        
        print(f"结果已保存: {json_path}")
        return json_path
    
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("交互式图像配准和形态学操作测试")
        self.root.geometry("1200x800")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 图像加载区域
        image_frame = ttk.LabelFrame(main_frame, text="图像加载", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_frame, text="加载图像1", command=self.load_image1).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(image_frame, text="加载图像2", command=self.load_image2).pack(side=tk.LEFT, padx=(0, 5))
        self.image_status = ttk.Label(image_frame, text="未加载图像")
        self.image_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # 参数控制区域
        params_frame = ttk.LabelFrame(main_frame, text="参数控制", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建左右两列
        left_column = ttk.Frame(params_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_column = ttk.Frame(params_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # 配准参数
        calib_frame = ttk.LabelFrame(left_column, text="配准参数", padding=10)
        calib_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_calibration_controls(calib_frame)
        
        # 形态学参数
        morph_frame = ttk.LabelFrame(right_column, text="形态学参数", padding=10)
        morph_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_morphology_controls(morph_frame)
        
        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="执行测试", command=self.run_test).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="保存结果", command=self.save_current_result).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="查看历史", command=self.show_history).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="重置参数", command=self.reset_parameters).pack(side=tk.LEFT, padx=(0, 5))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="测试结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return self.root
    
    def create_calibration_controls(self, parent):
        """创建配准参数控制"""
        # 检测器类型
        ttk.Label(parent, text="检测器类型:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.detector_var = tk.StringVar(value=self.calibration_params['detector_type'])
        detector_combo = ttk.Combobox(parent, textvariable=self.detector_var, 
                                     values=['SIFT', 'ORB', 'AKAZE'], state='readonly')
        detector_combo.grid(row=0, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        detector_combo.bind('<<ComboboxSelected>>', self.update_calibration_params)
        
        # 特征点数量
        ttk.Label(parent, text="特征点数量:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.nfeatures_var = tk.IntVar(value=self.calibration_params['nfeatures'])
        nfeatures_scale = ttk.Scale(parent, from_=100, to=3000, variable=self.nfeatures_var,
                                   orient=tk.HORIZONTAL, command=self.update_calibration_params)
        nfeatures_scale.grid(row=1, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 对比度阈值
        ttk.Label(parent, text="对比度阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.contrast_var = tk.DoubleVar(value=self.calibration_params['contrast_threshold'])
        contrast_scale = ttk.Scale(parent, from_=0.01, to=0.1, variable=self.contrast_var,
                                  orient=tk.HORIZONTAL, command=self.update_calibration_params)
        contrast_scale.grid(row=2, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 边缘阈值
        ttk.Label(parent, text="边缘阈值:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.edge_var = tk.IntVar(value=self.calibration_params['edge_threshold'])
        edge_scale = ttk.Scale(parent, from_=5, to=50, variable=self.edge_var,
                              orient=tk.HORIZONTAL, command=self.update_calibration_params)
        edge_scale.grid(row=3, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 比值阈值
        ttk.Label(parent, text="比值阈值:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.ratio_var = tk.DoubleVar(value=self.calibration_params['ratio_threshold'])
        ratio_scale = ttk.Scale(parent, from_=0.3, to=0.9, variable=self.ratio_var,
                               orient=tk.HORIZONTAL, command=self.update_calibration_params)
        ratio_scale.grid(row=4, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # RANSAC阈值
        ttk.Label(parent, text="RANSAC阈值:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.ransac_var = tk.DoubleVar(value=self.calibration_params['ransac_threshold'])
        ransac_scale = ttk.Scale(parent, from_=1.0, to=10.0, variable=self.ransac_var,
                                orient=tk.HORIZONTAL, command=self.update_calibration_params)
        ransac_scale.grid(row=5, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        parent.columnconfigure(1, weight=1)
    
    def create_morphology_controls(self, parent):
        """创建形态学参数控制"""
        # 启用形态学操作
        self.morph_enable_var = tk.BooleanVar(value=self.morphology_params['enable_morphology'])
        ttk.Checkbutton(parent, text="启用形态学操作", variable=self.morph_enable_var,
                       command=self.update_morphology_params).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # 核类型
        ttk.Label(parent, text="核类型:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.kernel_type_var = tk.StringVar(value=self.morphology_params['kernel_type'])
        kernel_combo = ttk.Combobox(parent, textvariable=self.kernel_type_var,
                                   values=['MORPH_RECT', 'MORPH_ELLIPSE', 'MORPH_CROSS'], state='readonly')
        kernel_combo.grid(row=1, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        kernel_combo.bind('<<ComboboxSelected>>', self.update_morphology_params)
        
        # 核大小
        ttk.Label(parent, text="核大小:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.kernel_size_var = tk.IntVar(value=self.morphology_params['kernel_size'])
        kernel_scale = ttk.Scale(parent, from_=3, to=15, variable=self.kernel_size_var,
                                orient=tk.HORIZONTAL, command=self.update_morphology_params)
        kernel_scale.grid(row=2, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 腐蚀迭代次数
        ttk.Label(parent, text="腐蚀迭代:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.erosion_var = tk.IntVar(value=self.morphology_params['erosion_iterations'])
        erosion_scale = ttk.Scale(parent, from_=0, to=5, variable=self.erosion_var,
                                 orient=tk.HORIZONTAL, command=self.update_morphology_params)
        erosion_scale.grid(row=3, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 膨胀迭代次数
        ttk.Label(parent, text="膨胀迭代:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.dilation_var = tk.IntVar(value=self.morphology_params['dilation_iterations'])
        dilation_scale = ttk.Scale(parent, from_=0, to=5, variable=self.dilation_var,
                                  orient=tk.HORIZONTAL, command=self.update_morphology_params)
        dilation_scale.grid(row=4, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 开运算
        self.opening_enable_var = tk.BooleanVar(value=self.morphology_params['enable_opening'])
        ttk.Checkbutton(parent, text="开运算", variable=self.opening_enable_var,
                       command=self.update_morphology_params).grid(row=5, column=0, sticky=tk.W, pady=2)
        
        self.opening_size_var = tk.IntVar(value=self.morphology_params['opening_kernel_size'])
        opening_scale = ttk.Scale(parent, from_=3, to=11, variable=self.opening_size_var,
                                 orient=tk.HORIZONTAL, command=self.update_morphology_params)
        opening_scale.grid(row=5, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # 闭运算
        self.closing_enable_var = tk.BooleanVar(value=self.morphology_params['enable_closing'])
        ttk.Checkbutton(parent, text="闭运算", variable=self.closing_enable_var,
                       command=self.update_morphology_params).grid(row=6, column=0, sticky=tk.W, pady=2)
        
        self.closing_size_var = tk.IntVar(value=self.morphology_params['closing_kernel_size'])
        closing_scale = ttk.Scale(parent, from_=3, to=11, variable=self.closing_size_var,
                                 orient=tk.HORIZONTAL, command=self.update_morphology_params)
        closing_scale.grid(row=6, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        parent.columnconfigure(1, weight=1)
    
    def update_calibration_params(self, event=None):
        """更新配准参数"""
        self.calibration_params['detector_type'] = self.detector_var.get()
        self.calibration_params['nfeatures'] = int(self.nfeatures_var.get())
        self.calibration_params['contrast_threshold'] = self.contrast_var.get()
        self.calibration_params['edge_threshold'] = int(self.edge_var.get())
        self.calibration_params['ratio_threshold'] = self.ratio_var.get()
        self.calibration_params['ransac_threshold'] = self.ransac_var.get()
    
    def update_morphology_params(self, event=None):
        """更新形态学参数"""
        self.morphology_params['enable_morphology'] = self.morph_enable_var.get()
        self.morphology_params['kernel_type'] = self.kernel_type_var.get()
        self.morphology_params['kernel_size'] = int(self.kernel_size_var.get())
        self.morphology_params['erosion_iterations'] = int(self.erosion_var.get())
        self.morphology_params['dilation_iterations'] = int(self.dilation_var.get())
        self.morphology_params['enable_opening'] = self.opening_enable_var.get()
        self.morphology_params['opening_kernel_size'] = int(self.opening_size_var.get())
        self.morphology_params['enable_closing'] = self.closing_enable_var.get()
        self.morphology_params['closing_kernel_size'] = int(self.closing_size_var.get())
    
    def load_image1(self):
        """加载第一张图像"""
        file_path = filedialog.askopenfilename(
            title="选择第一张图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.img1 = self.load_image_with_chinese_path(file_path)
            self.img1_path = file_path
            self.update_image_status()
    
    def load_image2(self):
        """加载第二张图像"""
        file_path = filedialog.askopenfilename(
            title="选择第二张图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.img2 = self.load_image_with_chinese_path(file_path)
            self.img2_path = file_path
            self.update_image_status()
    
    def update_image_status(self):
        """更新图像状态显示"""
        if self.img1 is not None and self.img2 is not None:
            status = f"已加载两张图像 ({self.img1.shape[:2]} & {self.img2.shape[:2]})"
        elif self.img1 is not None:
            status = f"已加载图像1 ({self.img1.shape[:2]})"
        elif self.img2 is not None:
            status = f"已加载图像2 ({self.img2.shape[:2]})"
        else:
            status = "未加载图像"
        
        self.image_status.config(text=status)
    
    def run_test(self):
        """运行测试"""
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("错误", "请先加载两张图像")
            return
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在执行测试...\n")
        self.root.update()
        
        # 在新线程中执行测试
        threading.Thread(target=self._run_test_thread, daemon=True).start()
    
    def _run_test_thread(self):
        """在线程中运行测试"""
        try:
            result = self.perform_calibration_with_morphology()
            if result:
                # 在主线程中更新GUI
                self.root.after(0, lambda: self._update_result_display(result))
                # 创建可视化
                self.root.after(100, lambda: self.create_visualization(result))
            else:
                self.root.after(0, lambda: self.result_text.insert(tk.END, "测试失败\n"))
        except Exception as e:
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"测试出错: {str(e)}\n"))
    
    def _update_result_display(self, result):
        """更新结果显示"""
        self.result_text.delete(1.0, tk.END)
        
        corner_effect = result.get('corner_removal_effect', {})
        
        text = f"""测试完成时间: {result['timestamp']}
工作流程: {result.get('workflow', '角点匹配对齐 → 形态学操作')}

角点匹配对齐结果:
  检测特征点: {result['original']['keypoints']}
  良好匹配: {result['original']['matches']}
  内点比例: {result['original']['inlier_ratio']:.3f}
  SSIM: {result['original']['ssim']:.3f}
  MSE: {result['original']['mse']:.1f}

形态学处理后结果:
  处理后特征点: {result['processed']['keypoints']}
  处理后匹配: {result['processed']['matches']}
  处理后内点比例: {result['processed']['inlier_ratio']:.3f}
  处理后SSIM: {result['processed']['ssim']:.3f}
  处理后MSE: {result['processed']['mse']:.1f}

角点移除效果:
  移除角点数: {corner_effect.get('corners_removed', 0)}
  移除比例: {corner_effect.get('removal_percentage', 0):.1f}%
  
改进效果:
  匹配数变化: {result['improvement']['match_change']:+d}
  内点率改进: {result['improvement']['inlier_ratio_improvement']:+.3f}
  SSIM改进: {result['improvement']['ssim_improvement']:+.3f}
  MSE改进: {result['improvement']['mse_improvement']:+.1f}

总结: {'成功移除了 ' + str(corner_effect.get('corners_removed', 0)) + ' 个无效角点' if corner_effect.get('corners_removed', 0) > 0 else '未移除角点'}
      {'，配准质量得到提升' if result['improvement']['ssim_improvement'] > 0 else '，但配准质量未显著提升'}
"""
        self.result_text.insert(tk.END, text)
    
    def save_current_result(self):
        """保存当前结果"""
        if self.current_result:
            json_path = self.save_result(self.current_result)
            if json_path:
                messagebox.showinfo("保存成功", f"结果已保存到: {json_path}")
        else:
            messagebox.showwarning("警告", "没有可保存的结果")
    
    def show_history(self):
        """显示历史记录"""
        if not self.results_history:
            messagebox.showinfo("信息", "没有历史记录")
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("测试历史记录")
        history_window.geometry("800x600")
        
        # 创建表格显示历史记录
        columns = ('时间', '检测器', '原始SSIM', '处理后SSIM', '改进')
        tree = ttk.Treeview(history_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        for i, result in enumerate(self.results_history):
            tree.insert('', 'end', values=(
                result['timestamp'][:19],
                result['calibration_params']['detector_type'],
                f"{result['original']['ssim']:.3f}",
                f"{result['processed']['ssim']:.3f}",
                f"{result['improvement']['ssim_improvement']:+.3f}"
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(history_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def reset_parameters(self):
        """重置参数到默认值"""
        # 重置配准参数
        self.detector_var.set('SIFT')
        self.nfeatures_var.set(1000)
        self.contrast_var.set(0.04)
        self.edge_var.set(10)
        self.ratio_var.set(0.7)
        self.ransac_var.set(5.0)
        
        # 重置形态学参数
        self.morph_enable_var.set(True)
        self.kernel_type_var.set('MORPH_RECT')
        self.kernel_size_var.set(5)
        self.erosion_var.set(1)
        self.dilation_var.set(1)
        self.opening_enable_var.set(True)
        self.opening_size_var.set(3)
        self.closing_enable_var.set(True)
        self.closing_size_var.set(3)
        
        self.update_calibration_params()
        self.update_morphology_params()
        
        messagebox.showinfo("信息", "参数已重置到默认值")
    
    def run_gui(self):
        """运行GUI"""
        root = self.create_gui()
        root.mainloop()


def main():
    """主函数"""
    # 创建测试器
    tester = InteractiveCalibrationTester()
    
    # 运行GUI
    tester.run_gui()


if __name__ == "__main__":
    main() 