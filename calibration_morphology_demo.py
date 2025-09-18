#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像配准和形态学操作演示
简化版本，用于快速测试不同参数的效果
基于Detection_of_unauthorized_building_works.py
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


class CalibrationMorphologyDemo:
    """配准和形态学操作演示"""
    
    def __init__(self, output_dir="calibration_demo_results"):
        """初始化演示器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 默认参数
        self.params = {
            'detector_type': 'SIFT',
            'nfeatures': 1000,
            'contrast_threshold': 0.04,
            'edge_threshold': 10,
            'ratio_threshold': 0.7,
            'ransac_threshold': 5.0,
            'min_matches': 8,
            # 形态学参数
            'enable_morphology': True,
            'kernel_size': 5,
            'erosion_iterations': 1,
            'dilation_iterations': 1,
            'enable_opening': True,
            'opening_kernel_size': 3,
            'enable_closing': True,
            'closing_kernel_size': 3
        }
        
        print("配准和形态学操作演示初始化完成")
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
    
    def create_detector(self):
        """创建特征检测器"""
        if self.params['detector_type'] == 'SIFT':
            return cv2.SIFT_create(
                nfeatures=self.params['nfeatures'],
                contrastThreshold=self.params['contrast_threshold'],
                edgeThreshold=self.params['edge_threshold']
            )
        elif self.params['detector_type'] == 'ORB':
            return cv2.ORB_create(
                nfeatures=self.params['nfeatures'],
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=self.params['edge_threshold']
            )
        elif self.params['detector_type'] == 'AKAZE':
            return cv2.AKAZE_create(
                threshold=self.params['contrast_threshold'] / 100,
                nOctaves=4,
                nOctaveLayers=4
            )
        else:
            return cv2.SIFT_create()
    
    def apply_morphology_operations(self, image):
        """应用形态学操作"""
        if not self.params['enable_morphology']:
            return image
        
        result = image.copy()
        
        # 创建矩形核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                         (self.params['kernel_size'], self.params['kernel_size']))
        
        # 腐蚀操作
        if self.params['erosion_iterations'] > 0:
            result = cv2.erode(result, kernel, iterations=self.params['erosion_iterations'])
        
        # 膨胀操作
        if self.params['dilation_iterations'] > 0:
            result = cv2.dilate(result, kernel, iterations=self.params['dilation_iterations'])
        
        # 开运算 (先腐蚀后膨胀，去除小的噪声点)
        if self.params['enable_opening'] and self.params['opening_kernel_size'] > 0:
            opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                     (self.params['opening_kernel_size'], 
                                                      self.params['opening_kernel_size']))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, opening_kernel)
        
        # 闭运算 (先膨胀后腐蚀，填充小的空洞)
        if self.params['enable_closing'] and self.params['closing_kernel_size'] > 0:
            closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                     (self.params['closing_kernel_size'], 
                                                      self.params['closing_kernel_size']))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, closing_kernel)
        
        return result
    
    def perform_calibration_test(self, img1_path, img2_path):
        """执行角点匹配对齐然后膨胀腐蚀的配准测试"""
        print(f"\n=== 角点匹配对齐 + 膨胀腐蚀测试: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} ===")
        
        # 加载图像
        img1 = self.load_image_with_chinese_path(img1_path)
        img2 = self.load_image_with_chinese_path(img2_path)
        
        if img1 is None or img2 is None:
            print("图像加载失败")
            return None
        
        print(f"图像尺寸: {img1.shape} vs {img2.shape}")
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 步骤1: 角点匹配和图像对齐
        print("步骤1: 角点匹配和图像对齐...")
        detector = self.create_detector()
        
        # 检测原始图像的特征点
        kp1_orig, des1_orig = detector.detectAndCompute(gray1, None)
        kp2_orig, des2_orig = detector.detectAndCompute(gray2, None)
        print(f"原始特征点: {len(kp1_orig)}/{len(kp2_orig)}")
        
        if des1_orig is None or des2_orig is None or len(kp1_orig) < 4 or len(kp2_orig) < 4:
            print("特征点不足，无法进行配准")
            return None
        
        # 步骤2: 特征匹配进行图像对齐
        print("步骤2: 特征匹配...")
        if self.params['detector_type'] == 'SIFT':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # k-近邻匹配
        matches_orig = matcher.knnMatch(des1_orig, des2_orig, k=2)
        
        # 应用比值测试
        good_matches_orig = []
        for match_pair in matches_orig:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.params['ratio_threshold'] * n.distance:
                    good_matches_orig.append(m)
        
        print(f"原始良好匹配: {len(good_matches_orig)}")
        
        if len(good_matches_orig) < self.params['min_matches']:
            print("匹配点不足，无法进行配准")
            return None
        
        # 步骤3: 计算透视变换矩阵进行图像对齐
        print("步骤3: 计算透视变换矩阵...")
        src_pts = np.float32([kp1_orig[m.queryIdx].pt for m in good_matches_orig]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2_orig[m.trainIdx].pt for m in good_matches_orig]).reshape(-1, 1, 2)
        
        M_orig, mask_orig = cv2.findHomography(src_pts, dst_pts, 
                                             cv2.RANSAC, 
                                             ransacReprojThreshold=self.params['ransac_threshold'])
        
        inlier_ratio_orig = np.sum(mask_orig) / len(mask_orig) if mask_orig is not None else 0
        print(f"原始内点比例: {inlier_ratio_orig:.3f}")
        
        # 步骤4: 应用透视变换对齐图像
        print("步骤4: 应用透视变换对齐图像...")
        h, w = img1.shape[:2]
        aligned_img2_orig = cv2.warpPerspective(img2, M_orig, (w, h)) if M_orig is not None else img2
        aligned_gray2_orig = cv2.warpPerspective(gray2, M_orig, (w, h)) if M_orig is not None else gray2
        
        # 步骤5: 对对齐后的图像应用膨胀腐蚀操作
        print("步骤5: 对对齐后的图像应用膨胀腐蚀操作...")
        processed_gray1 = self.apply_morphology_operations(gray1)
        processed_aligned_gray2 = self.apply_morphology_operations(aligned_gray2_orig)
        
        # 步骤6: 在膨胀腐蚀处理后重新检测特征点
        print("步骤6: 在膨胀腐蚀处理后重新检测特征点...")
        kp1_proc, des1_proc = detector.detectAndCompute(processed_gray1, None)
        kp2_proc, des2_proc = detector.detectAndCompute(processed_aligned_gray2, None)
        
        print(f"膨胀腐蚀处理后特征点: {len(kp1_proc)}/{len(kp2_proc)}")
        
        # 重新匹配膨胀腐蚀处理后的特征点
        good_matches_proc = []
        inlier_ratio_proc = inlier_ratio_orig  # 默认使用原始内点比例
        M_proc = M_orig  # 默认使用原始变换矩阵
        
        if des1_proc is not None and des2_proc is not None and len(kp1_proc) >= 4 and len(kp2_proc) >= 4:
            matches_proc = matcher.knnMatch(des1_proc, des2_proc, k=2)
            
            for match_pair in matches_proc:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.params['ratio_threshold'] * n.distance:
                        good_matches_proc.append(m)
            
            print(f"膨胀腐蚀处理后良好匹配: {len(good_matches_proc)}")
            
            if len(good_matches_proc) >= self.params['min_matches']:
                src_pts_proc = np.float32([kp1_proc[m.queryIdx].pt for m in good_matches_proc]).reshape(-1, 1, 2)
                dst_pts_proc = np.float32([kp2_proc[m.trainIdx].pt for m in good_matches_proc]).reshape(-1, 1, 2)
                
                M_proc, mask_proc = cv2.findHomography(src_pts_proc, dst_pts_proc, 
                                                     cv2.RANSAC, 
                                                     ransacReprojThreshold=self.params['ransac_threshold'])
                
                inlier_ratio_proc = np.sum(mask_proc) / len(mask_proc) if mask_proc is not None else 0
                print(f"膨胀腐蚀处理后内点比例: {inlier_ratio_proc:.3f}")
        
        # 生成最终对齐结果
        aligned_img2_proc = cv2.warpPerspective(img2, M_proc, (w, h)) if M_proc is not None else aligned_img2_orig
        
        # 步骤7: 计算配准质量指标
        
        def calculate_quality(img1, img2):
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
        
        ssim_orig, mse_orig = calculate_quality(img1, aligned_img2_orig)
        ssim_proc, mse_proc = calculate_quality(img1, aligned_img2_proc)
        
        print(f"\n配准质量对比:")
        print(f"角点匹配对齐  - SSIM: {ssim_orig:.3f}, MSE: {mse_orig:.1f}")
        print(f"膨胀腐蚀处理后 - SSIM: {ssim_proc:.3f}, MSE: {mse_proc:.1f}")
        print(f"SSIM改进: {ssim_proc - ssim_orig:+.3f}")
        print(f"MSE改进: {mse_orig - mse_proc:+.1f}")
        
        # 角点移除效果统计
        corners_removed = len(kp1_orig) - len(kp1_proc)
        print(f"\n角点移除效果:")
        print(f"对齐前角点: {len(kp1_orig)}")
        print(f"膨胀腐蚀后角点: {len(kp1_proc)}")
        print(f"移除角点数: {corners_removed}")
        print(f"移除比例: {corners_removed / len(kp1_orig) * 100:.1f}%" if len(kp1_orig) > 0 else "移除比例: 0.0%")
        
        # 整理结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'params': self.params.copy(),
            'workflow': 'corner_matching_then_morphology',
            'images': {
                'img1_path': img1_path,
                'img2_path': img2_path,
                'img1_shape': img1.shape,
                'img2_shape': img2.shape
            },
            'original': {
                'keypoints': len(kp1_orig),
                'matches': len(good_matches_orig),
                'inlier_ratio': float(inlier_ratio_orig),
                'ssim': float(ssim_orig),
                'mse': float(mse_orig)
            },
            'processed': {
                'keypoints': len(kp1_proc),
                'matches': len(good_matches_proc),
                'inlier_ratio': float(inlier_ratio_proc),
                'ssim': float(ssim_proc),
                'mse': float(mse_proc)
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
                'corners_removed': corners_removed,
                'removal_percentage': float(corners_removed / len(kp1_orig) * 100) if len(kp1_orig) > 0 else 0.0
            }
        }
        
        # 存储可视化数据
        result['visualization_data'] = {
            'img1': img1,
            'img2': img2,
            'aligned_img2_orig': aligned_img2_orig,  # 角点匹配对齐后的结果
            'processed_gray1': processed_gray1,
            'processed_aligned_gray2': processed_aligned_gray2,
            'aligned_img2_proc': aligned_img2_proc,  # 膨胀腐蚀处理后的结果
            'keypoints_orig': (kp1_orig, kp2_orig),
            'keypoints_proc': (kp1_proc, kp2_proc),
            'matches_orig': good_matches_orig,
            'matches_proc': good_matches_proc,
            'corner_removal_visualization': {
                'before_morphology': len(kp1_orig),
                'after_morphology': len(kp1_proc),
                'removed_corners': corners_removed
            }
        }
        
        return result
    
    def create_visualization(self, result):
        """创建可视化"""
        if result is None:
            return None
        
        viz_data = result['visualization_data']
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('角点匹配对齐 + 膨胀腐蚀操作效果对比', fontsize=16, fontweight='bold')
        
        # 第一行：原始图像和对齐过程
        axes[0, 0].imshow(cv2.cvtColor(viz_data['img1'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('基准图像 (图像1)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(viz_data['img2'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('待对齐图像 (图像2)', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(viz_data['aligned_img2_orig'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('角点匹配对齐后', fontsize=12)
        axes[0, 2].axis('off')
        
        # 创建重叠验证图像
        aligned_overlap = cv2.addWeighted(viz_data['img1'], 0.5, viz_data['aligned_img2_orig'], 0.5, 0)
        axes[0, 3].imshow(cv2.cvtColor(aligned_overlap, cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title('对齐效果验证 (重叠)', fontsize=12)
        axes[0, 3].axis('off')
        
        # 第二行：膨胀腐蚀处理和特征点对比
        axes[1, 0].imshow(viz_data['processed_gray1'], cmap='gray')
        axes[1, 0].set_title('膨胀腐蚀处理 (图像1)', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(viz_data['processed_aligned_gray2'], cmap='gray')
        axes[1, 1].set_title('膨胀腐蚀处理 (对齐后图像2)', fontsize=12)
        axes[1, 1].axis('off')
        
        # 特征点对比
        kp1_orig, kp2_orig = viz_data['keypoints_orig']
        kp1_proc, kp2_proc = viz_data['keypoints_proc']
        
        img1_kp_orig = cv2.drawKeypoints(viz_data['img1'], kp1_orig, None, color=(0, 255, 0), flags=0)
        img1_kp_proc = cv2.drawKeypoints(viz_data['img1'], kp1_proc, None, color=(255, 0, 0), flags=0)
        
        axes[1, 2].imshow(cv2.cvtColor(img1_kp_orig, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'对齐前特征点: {len(kp1_orig)}', fontsize=12)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(cv2.cvtColor(img1_kp_proc, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title(f'膨胀腐蚀后特征点: {len(kp1_proc)}', fontsize=12)
        axes[1, 3].axis('off')
        
        # 第三行：配准结果和统计信息
        axes[2, 0].imshow(cv2.cvtColor(viz_data['aligned_img2_orig'], cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title(f'角点匹配对齐结果\nSSIM: {result["original"]["ssim"]:.3f}', fontsize=12)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(cv2.cvtColor(viz_data['aligned_img2_proc'], cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title(f'膨胀腐蚀处理后结果\nSSIM: {result["processed"]["ssim"]:.3f}', fontsize=12)
        axes[2, 1].axis('off')
        
        # 工作流程和参数统计
        corner_effect = result['corner_removal_effect']
        workflow_text = f"""工作流程: 角点匹配对齐 → 膨胀腐蚀操作

角点移除效果:
对齐前角点: {corner_effect['corners_before_morphology']}
膨胀腐蚀后角点: {corner_effect['corners_after_morphology']}
移除角点数: {corner_effect['corners_removed']}
移除比例: {corner_effect['removal_percentage']:.1f}%

膨胀腐蚀参数:
核大小: {result['params']['kernel_size']}
腐蚀迭代: {result['params']['erosion_iterations']}
膨胀迭代: {result['params']['dilation_iterations']}
开运算核: {result['params']['opening_kernel_size']}
闭运算核: {result['params']['closing_kernel_size']}"""
        
        axes[2, 2].text(0.05, 0.95, workflow_text, ha='left', va='top', 
                        transform=axes[2, 2].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('工作流程参数', fontsize=12)
        
        # 效果评估
        improvement = result['improvement']
        quality_text = f"""效果评估:

角点匹配对齐:
SSIM: {result['original']['ssim']:.3f}
MSE: {result['original']['mse']:.1f}
内点率: {result['original']['inlier_ratio']:.3f}
匹配数: {result['original']['matches']}

膨胀腐蚀处理后:
SSIM: {result['processed']['ssim']:.3f}
MSE: {result['processed']['mse']:.1f}
内点率: {result['processed']['inlier_ratio']:.3f}
匹配数: {result['processed']['matches']}

改进效果:
SSIM改进: {improvement['ssim_improvement']:+.3f}
MSE改进: {improvement['mse_improvement']:+.1f}
匹配数变化: {improvement['match_change']:+d}

结论: {'膨胀腐蚀操作提升了配准质量' if improvement['ssim_improvement'] > 0 else '膨胀腐蚀操作未显著提升配准质量'}
     {'并有效移除了无效角点' if corner_effect['corners_removed'] > 0 else '但未移除角点'}"""
        
        color = 'lightgreen' if improvement['ssim_improvement'] > 0 else 'lightcoral'
        axes[2, 3].text(0.05, 0.95, quality_text, ha='left', va='top', 
                        transform=axes[2, 3].transAxes, fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        axes[2, 3].set_xlim(0, 1)
        axes[2, 3].set_ylim(0, 1)
        axes[2, 3].axis('off')
        axes[2, 3].set_title('效果评估', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.output_dir, f"calibration_morphology_demo_{timestamp}.jpg")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return image_path
    
    def save_result(self, result):
        """保存结果"""
        if result is None:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 移除不可序列化的数据
        save_result = result.copy()
        if 'visualization_data' in save_result:
            del save_result['visualization_data']
        
        json_path = os.path.join(self.output_dir, f"calibration_demo_result_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存: {json_path}")
        return json_path
    
    def test_parameter_combinations(self, img1_path, img2_path):
        """测试不同参数组合"""
        print("\n=== 参数组合测试 ===")
        
        # 定义测试参数组合
        test_combinations = [
            # 基础配置
            {
                'name': '无形态学操作',
                'params': {'enable_morphology': False}
            },
            # 轻度形态学处理
            {
                'name': '轻度处理',
                'params': {
                    'enable_morphology': True,
                    'kernel_size': 3,
                    'erosion_iterations': 1,
                    'dilation_iterations': 1,
                    'opening_kernel_size': 3,
                    'closing_kernel_size': 3
                }
            },
            # 中度形态学处理
            {
                'name': '中度处理',
                'params': {
                    'enable_morphology': True,
                    'kernel_size': 5,
                    'erosion_iterations': 2,
                    'dilation_iterations': 2,
                    'opening_kernel_size': 5,
                    'closing_kernel_size': 5
                }
            },
            # 强度形态学处理
            {
                'name': '强度处理',
                'params': {
                    'enable_morphology': True,
                    'kernel_size': 7,
                    'erosion_iterations': 3,
                    'dilation_iterations': 3,
                    'opening_kernel_size': 7,
                    'closing_kernel_size': 7
                }
            }
        ]
        
        results = []
        
        for combination in test_combinations:
            print(f"\n--- 测试: {combination['name']} ---")
            
            # 更新参数
            original_params = self.params.copy()
            self.params.update(combination['params'])
            
            # 执行测试
            result = self.perform_calibration_test(img1_path, img2_path)
            if result:
                result['test_name'] = combination['name']
                results.append(result)
            
            # 恢复原始参数
            self.params = original_params
        
        # 生成对比报告
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results):
        """生成对比报告"""
        if not results:
            return
        
        print("\n" + "="*60)
        print("参数组合对比报告")
        print("="*60)
        
        # 表头
        print(f"{'配置':<15} {'SSIM改进':<10} {'MSE改进':<10} {'内点率改进':<12} {'特征点变化':<12}")
        print("-" * 60)
        
        # 排序结果（按SSIM改进排序）
        sorted_results = sorted(results, key=lambda x: x['improvement']['ssim_improvement'], reverse=True)
        
        for result in sorted_results:
            print(f"{result['test_name']:<15} "
                  f"{result['improvement']['ssim_improvement']:+.3f}     "
                  f"{result['improvement']['mse_improvement']:+.1f}     "
                  f"{result['improvement']['inlier_ratio_improvement']:+.3f}        "
                  f"{result['improvement']['keypoint_change']:+d}")
        
        # 最佳配置推荐
        best_result = sorted_results[0]
        print(f"\n推荐配置: {best_result['test_name']}")
        print(f"SSIM改进: {best_result['improvement']['ssim_improvement']:+.3f}")
        print(f"参数设置: {best_result['params']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像配准和形态学操作演示")
    parser.add_argument('--img1', required=True, help='第一张图像路径')
    parser.add_argument('--img2', required=True, help='第二张图像路径')
    parser.add_argument('--detector', default='SIFT', choices=['SIFT', 'ORB', 'AKAZE'], help='特征检测器类型')
    parser.add_argument('--test-combinations', action='store_true', help='测试不同参数组合')
    parser.add_argument('--kernel-size', type=int, default=5, help='形态学核大小')
    parser.add_argument('--erosion', type=int, default=1, help='腐蚀迭代次数')
    parser.add_argument('--dilation', type=int, default=1, help='膨胀迭代次数')
    
    args = parser.parse_args()
    
    # 创建演示器
    demo = CalibrationMorphologyDemo()
    
    # 更新参数
    demo.params.update({
        'detector_type': args.detector,
        'kernel_size': args.kernel_size,
        'erosion_iterations': args.erosion,
        'dilation_iterations': args.dilation
    })
    
    try:
        if args.test_combinations:
            # 测试参数组合
            results = demo.test_parameter_combinations(args.img1, args.img2)
        else:
            # 单次测试
            result = demo.perform_calibration_test(args.img1, args.img2)
            if result:
                # 创建可视化
                image_path = demo.create_visualization(result)
                # 保存结果
                json_path = demo.save_result(result)
                print(f"\n可视化图像: {image_path}")
                print(f"结果文件: {json_path}")
            else:
                print("测试失败")
    
    except Exception as e:
        print(f"演示过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果直接运行，使用默认图像路径
    if len(os.sys.argv) == 1:
        # 默认图像路径
        img1_path = r'C:\Users\admin\Desktop\两期比对\两期比对\Snipaste_2025-09-08_15-39-45.png'
        img2_path = r'C:\Users\admin\Desktop\两期比对\两期比对\Snipaste_2025-09-08_15-39-16.png'
        
        demo = CalibrationMorphologyDemo()
        print("使用默认图像路径进行演示...")
        print(f"图像1: {os.path.basename(img1_path)}")
        print(f"图像2: {os.path.basename(img2_path)}")
        
        # 执行测试
        result = demo.perform_calibration_test(img1_path, img2_path)
        if result:
            image_path = demo.create_visualization(result)
            json_path = demo.save_result(result)
            print(f"\n演示完成!")
            print(f"可视化图像: {image_path}")
        
        # 测试参数组合
        print("\n是否测试不同参数组合? (y/n): ", end="")
        if input().lower() == 'y':
            demo.test_parameter_combinations(img1_path, img2_path)
    else:
        main() 