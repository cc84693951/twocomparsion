#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角点匹配算法动态参数测试演示系统
基于Detection_of_unauthorized_building_works.py的角点检测功能
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class MatchingResult:
    """匹配结果数据类"""
    algorithm: str
    params: Dict[str, Any]
    keypoints1: int
    keypoints2: int
    matches: int
    good_matches: int
    inlier_ratio: float
    processing_time: float
    homography_quality: float
    match_quality_score: float


class AnglePointMatchingTester:
    """角点匹配算法测试器"""
    
    def __init__(self, output_dir="angle_point_test_results"):
        """
        初始化测试器
        
        Args:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试配置
        self.test_algorithms = ['SIFT', 'ORB', 'AKAZE', 'BRISK']
        self.test_results = []
        
        print("角点匹配算法测试器初始化完成")
        print(f"输出目录: {output_dir}")
        print(f"支持的算法: {', '.join(self.test_algorithms)}")
    
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
    
    def create_sift_detector(self, nfeatures=1000, contrast_threshold=0.04, 
                           edge_threshold=10, sigma=1.6):
        """创建SIFT检测器"""
        return cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    def create_orb_detector(self, nfeatures=1000, scale_factor=1.2, 
                          nlevels=8, edge_threshold=31):
        """创建ORB检测器"""
        return cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            edgeThreshold=edge_threshold,
            scoreType=cv2.ORB_HARRIS_SCORE
        )
    
    def create_akaze_detector(self, threshold=0.001, noctaves=4, 
                            noctave_layers=4):
        """创建AKAZE检测器"""
        return cv2.AKAZE_create(
            threshold=threshold,
            nOctaves=noctaves,
            nOctaveLayers=noctave_layers,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
    
    def create_brisk_detector(self, thresh=30, octaves=3, pattern_scale=1.0):
        """创建BRISK检测器"""
        return cv2.BRISK_create(
            thresh=thresh,
            octaves=octaves,
            patternScale=pattern_scale
        )
    
    def create_matcher(self, algorithm, descriptor_type='float'):
        """
        创建特征匹配器
        
        Args:
            algorithm (str): 匹配算法 ('FLANN' 或 'BF')
            descriptor_type (str): 描述符类型 ('float' 或 'binary')
        """
        if algorithm == 'FLANN' and descriptor_type == 'float':
            # 对于SIFT, SURF等浮点描述符
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # 对于所有其他情况使用暴力匹配器
            if descriptor_type == 'float':
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_and_match(self, img1, img2, detector, algorithm_name, 
                        params, ratio_threshold=0.7):
        """
        检测特征点并进行匹配
        
        Args:
            img1, img2: 输入图像
            detector: 特征检测器
            algorithm_name: 算法名称
            params: 参数字典
            ratio_threshold: 比值测试阈值
            
        Returns:
            MatchingResult: 匹配结果
        """
        start_time = time.time()
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 检测关键点和描述符
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        processing_time = time.time() - start_time
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return MatchingResult(
                algorithm=algorithm_name,
                params=params,
                keypoints1=len(kp1) if kp1 else 0,
                keypoints2=len(kp2) if kp2 else 0,
                matches=0,
                good_matches=0,
                inlier_ratio=0.0,
                processing_time=processing_time,
                homography_quality=0.0,
                match_quality_score=0.0
            )
        
        # 创建匹配器 - 使用BF匹配器确保兼容性
        descriptor_type = 'binary' if algorithm_name in ['ORB', 'BRISK', 'AKAZE'] else 'float'
        matcher = self.create_matcher('BF', descriptor_type)
        
        # 进行匹配
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # 应用比值测试筛选好的匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # 计算单应性矩阵和内点比例
        inlier_ratio = 0.0
        homography_quality = 0.0
        
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, 
                                       cv2.RANSAC, 
                                       ransacReprojThreshold=5.0)
            
            if M is not None and mask is not None:
                inlier_ratio = np.sum(mask) / len(mask)
                
                # 计算单应性矩阵质量（基于条件数）
                try:
                    homography_quality = 1.0 / np.linalg.cond(M)
                except:
                    homography_quality = 0.0
        
        # 计算匹配质量分数
        match_quality_score = (
            0.3 * min(1.0, len(good_matches) / 100) +  # 匹配点数量
            0.4 * inlier_ratio +                        # 内点比例
            0.3 * min(1.0, homography_quality * 1000)   # 单应性质量
        )
        
        return MatchingResult(
            algorithm=algorithm_name,
            params=params,
            keypoints1=len(kp1),
            keypoints2=len(kp2),
            matches=len(matches),
            good_matches=len(good_matches),
            inlier_ratio=inlier_ratio,
            processing_time=processing_time,
            homography_quality=homography_quality,
            match_quality_score=match_quality_score
        )
    
    def get_best_parameters(self):
        """
        获取各算法的最佳参数组合
        
        Returns:
            dict: 各算法的参数配置
        """
        return {
            'SIFT': [
                # 高精度配置
                {'nfeatures': 2000, 'contrast_threshold': 0.03, 'edge_threshold': 10, 'sigma': 1.6},
                # 标准配置
                {'nfeatures': 1000, 'contrast_threshold': 0.04, 'edge_threshold': 10, 'sigma': 1.6},
                # 快速配置
                {'nfeatures': 500, 'contrast_threshold': 0.05, 'edge_threshold': 15, 'sigma': 1.6},
            ],
            'ORB': [
                # 高精度配置
                {'nfeatures': 2000, 'scale_factor': 1.2, 'nlevels': 10, 'edge_threshold': 31},
                # 标准配置
                {'nfeatures': 1000, 'scale_factor': 1.2, 'nlevels': 8, 'edge_threshold': 31},
                # 快速配置
                {'nfeatures': 500, 'scale_factor': 1.3, 'nlevels': 6, 'edge_threshold': 31},
            ],
            'AKAZE': [
                # 高精度配置
                {'threshold': 0.0005, 'noctaves': 5, 'noctave_layers': 4},
                # 标准配置
                {'threshold': 0.001, 'noctaves': 4, 'noctave_layers': 4},
                # 快速配置
                {'threshold': 0.002, 'noctaves': 3, 'noctave_layers': 3},
            ],
            'BRISK': [
                # 高精度配置
                {'thresh': 20, 'octaves': 4, 'pattern_scale': 1.0},
                # 标准配置
                {'thresh': 30, 'octaves': 3, 'pattern_scale': 1.0},
                # 快速配置
                {'thresh': 40, 'octaves': 2, 'pattern_scale': 1.2},
            ]
        }
    
    def run_comprehensive_test(self, img1_path, img2_path):
        """
        运行综合测试
        
        Args:
            img1_path (str): 第一张图像路径
            img2_path (str): 第二张图像路径
            
        Returns:
            list: 测试结果列表
        """
        print("开始运行角点匹配综合测试...")
        
        # 加载图像
        img1 = self.load_image_with_chinese_path(img1_path)
        img2 = self.load_image_with_chinese_path(img2_path)
        
        if img1 is None or img2 is None:
            print("图像加载失败")
            return []
        
        print(f"图像1尺寸: {img1.shape}")
        print(f"图像2尺寸: {img2.shape}")
        
        # 获取参数配置
        param_configs = self.get_best_parameters()
        
        # 测试不同的比值阈值
        ratio_thresholds = [0.6, 0.7, 0.8]
        
        results = []
        
        for algorithm in self.test_algorithms:
            print(f"\n测试算法: {algorithm}")
            
            for i, params in enumerate(param_configs[algorithm]):
                config_name = ['高精度', '标准', '快速'][i]
                print(f"  配置: {config_name} - {params}")
                
                # 创建检测器
                if algorithm == 'SIFT':
                    detector = self.create_sift_detector(**params)
                elif algorithm == 'ORB':
                    detector = self.create_orb_detector(**params)
                elif algorithm == 'AKAZE':
                    detector = self.create_akaze_detector(**params)
                elif algorithm == 'BRISK':
                    detector = self.create_brisk_detector(**params)
                
                # 测试不同比值阈值
                for ratio_threshold in ratio_thresholds:
                    # 添加配置信息到参数
                    test_params = params.copy()
                    test_params['config'] = config_name
                    test_params['ratio_threshold'] = ratio_threshold
                    
                    result = self.detect_and_match(
                        img1, img2, detector, algorithm, 
                        test_params, ratio_threshold
                    )
                    
                    results.append(result)
                    
                    print(f"    比值阈值{ratio_threshold}: "
                          f"特征点{result.keypoints1}/{result.keypoints2}, "
                          f"匹配{result.good_matches}, "
                          f"内点率{result.inlier_ratio:.3f}, "
                          f"质量分数{result.match_quality_score:.3f}")
        
        self.test_results = results
        print(f"\n测试完成，共获得 {len(results)} 个结果")
        
        return results
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.test_results:
            print("没有测试结果可分析")
            return
        
        print("\n=== 测试结果分析 ===")
        
        # 按算法分组分析
        by_algorithm = {}
        for result in self.test_results:
            if result.algorithm not in by_algorithm:
                by_algorithm[result.algorithm] = []
            by_algorithm[result.algorithm].append(result)
        
        # 分析每个算法的最佳结果
        print("\n各算法最佳结果:")
        best_overall = None
        best_score = 0
        
        for algorithm, results in by_algorithm.items():
            best_result = max(results, key=lambda x: x.match_quality_score)
            print(f"\n{algorithm}:")
            print(f"  最佳配置: {best_result.params.get('config', 'Unknown')}")
            print(f"  比值阈值: {best_result.params.get('ratio_threshold', 'Unknown')}")
            print(f"  特征点数: {best_result.keypoints1}/{best_result.keypoints2}")
            print(f"  良好匹配: {best_result.good_matches}")
            print(f"  内点比例: {best_result.inlier_ratio:.3f}")
            print(f"  处理时间: {best_result.processing_time:.3f}s")
            print(f"  质量分数: {best_result.match_quality_score:.3f}")
            
            if best_result.match_quality_score > best_score:
                best_score = best_result.match_quality_score
                best_overall = best_result
        
        print(f"\n总体最佳算法: {best_overall.algorithm}")
        print(f"最佳参数: {best_overall.params}")
        print(f"最佳质量分数: {best_overall.match_quality_score:.3f}")
        
        return best_overall
    
    def create_comparison_visualization(self, img1_path, img2_path):
        """
        创建对比可视化
        
        Args:
            img1_path (str): 第一张图像路径
            img2_path (str): 第二张图像路径
        """
        if not self.test_results:
            print("没有测试结果可可视化")
            return
        
        # 加载图像
        img1 = self.load_image_with_chinese_path(img1_path)
        img2 = self.load_image_with_chinese_path(img2_path)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建大型对比图
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('角点匹配算法动态参数测试对比结果', fontsize=16, fontweight='bold')
        
        # 为每个算法选择最佳结果
        algorithms = ['SIFT', 'ORB', 'AKAZE', 'BRISK']
        
        for i, algorithm in enumerate(algorithms):
            # 找到该算法的最佳结果
            algo_results = [r for r in self.test_results if r.algorithm == algorithm]
            if not algo_results:
                continue
            
            best_result = max(algo_results, key=lambda x: x.match_quality_score)
            
            # 重新运行最佳配置以获取关键点
            if algorithm == 'SIFT':
                detector = self.create_sift_detector(
                    nfeatures=best_result.params.get('nfeatures', 1000),
                    contrast_threshold=best_result.params.get('contrast_threshold', 0.04),
                    edge_threshold=best_result.params.get('edge_threshold', 10),
                    sigma=best_result.params.get('sigma', 1.6)
                )
            elif algorithm == 'ORB':
                detector = self.create_orb_detector(
                    nfeatures=best_result.params.get('nfeatures', 1000),
                    scale_factor=best_result.params.get('scale_factor', 1.2),
                    nlevels=best_result.params.get('nlevels', 8),
                    edge_threshold=best_result.params.get('edge_threshold', 31)
                )
            elif algorithm == 'AKAZE':
                detector = self.create_akaze_detector(
                    threshold=best_result.params.get('threshold', 0.001),
                    noctaves=best_result.params.get('noctaves', 4),
                    noctave_layers=best_result.params.get('noctave_layers', 4)
                )
            elif algorithm == 'BRISK':
                detector = self.create_brisk_detector(
                    thresh=best_result.params.get('thresh', 30),
                    octaves=best_result.params.get('octaves', 3),
                    pattern_scale=best_result.params.get('pattern_scale', 1.0)
                )
            
            # 检测特征点
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            # 绘制特征点
            img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
            img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
            
            # 显示带特征点的图像
            axes[i, 0].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f'{algorithm} - 图像1特征点\n检测数量: {len(kp1)}', fontsize=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f'{algorithm} - 图像2特征点\n检测数量: {len(kp2)}', fontsize=10)
            axes[i, 1].axis('off')
            
            # 进行匹配
            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                descriptor_type = 'binary' if algorithm in ['ORB', 'BRISK', 'AKAZE'] else 'float'
                matcher = self.create_matcher('BF', descriptor_type)
                
                matches = matcher.knnMatch(des1, des2, k=2)
                
                # 筛选好的匹配
                good_matches = []
                ratio_threshold = best_result.params.get('ratio_threshold', 0.7)
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)
                
                # 绘制匹配结果
                match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                axes[i, 2].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
                axes[i, 2].set_title(f'{algorithm} - 特征匹配\n良好匹配: {len(good_matches)}', fontsize=10)
                axes[i, 2].axis('off')
            else:
                axes[i, 2].text(0.5, 0.5, '匹配失败', ha='center', va='center', 
                               transform=axes[i, 2].transAxes, fontsize=12)
                axes[i, 2].set_title(f'{algorithm} - 特征匹配', fontsize=10)
                axes[i, 2].axis('off')
            
            # 显示参数和结果
            result_text = f"""算法: {algorithm}
配置: {best_result.params.get('config', 'Unknown')}

参数设置:
特征点数: {best_result.params.get('nfeatures', 'N/A')}
比值阈值: {best_result.params.get('ratio_threshold', 'N/A')}

检测结果:
特征点: {best_result.keypoints1}/{best_result.keypoints2}
匹配数: {best_result.good_matches}
内点率: {best_result.inlier_ratio:.3f}
处理时间: {best_result.processing_time:.3f}s

质量评估:
质量分数: {best_result.match_quality_score:.3f}
"""
            
            axes[i, 3].text(0.05, 0.95, result_text, ha='left', va='top', 
                           transform=axes[i, 3].transAxes, fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
            axes[i, 3].set_xlim(0, 1)
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].axis('off')
            axes[i, 3].set_title(f'{algorithm} - 详细结果', fontsize=10)
        
        plt.tight_layout()
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.output_dir, f"angle_point_matching_comparison_{timestamp}.jpg")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"对比可视化已保存: {image_path}")
        return image_path
    
    def create_sift_comparison_visualization(self, img1_path, img2_path):
        """
        创建专门的SIFT与其他算法对比可视化
        
        Args:
            img1_path (str): 第一张图像路径
            img2_path (str): 第二张图像路径
        """
        if not self.test_results:
            print("没有测试结果可可视化")
            return
        
        # 加载图像
        img1 = self.load_image_with_chinese_path(img1_path)
        img2 = self.load_image_with_chinese_path(img2_path)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建SIFT对比图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('SIFT算法与其他算法详细对比分析', fontsize=16, fontweight='bold')
        
        # 获取各算法的最佳结果
        algorithms = ['SIFT', 'ORB', 'AKAZE', 'BRISK']
        best_results = {}
        
        for algorithm in algorithms:
            algo_results = [r for r in self.test_results if r.algorithm == algorithm]
            if algo_results:
                best_results[algorithm] = max(algo_results, key=lambda x: x.match_quality_score)
        
        # 第一行：特征点检测对比
        for i, algorithm in enumerate(algorithms):
            if algorithm not in best_results:
                continue
                
            result = best_results[algorithm]
            
            # 重新运行检测以获取特征点
            if algorithm == 'SIFT':
                detector = self.create_sift_detector(
                    nfeatures=result.params.get('nfeatures', 1000),
                    contrast_threshold=result.params.get('contrast_threshold', 0.04),
                    edge_threshold=result.params.get('edge_threshold', 10),
                    sigma=result.params.get('sigma', 1.6)
                )
            elif algorithm == 'ORB':
                detector = self.create_orb_detector(
                    nfeatures=result.params.get('nfeatures', 1000),
                    scale_factor=result.params.get('scale_factor', 1.2),
                    nlevels=result.params.get('nlevels', 8),
                    edge_threshold=result.params.get('edge_threshold', 31)
                )
            elif algorithm == 'AKAZE':
                detector = self.create_akaze_detector(
                    threshold=result.params.get('threshold', 0.001),
                    noctaves=result.params.get('noctaves', 4),
                    noctave_layers=result.params.get('noctave_layers', 4)
                )
            elif algorithm == 'BRISK':
                detector = self.create_brisk_detector(
                    thresh=result.params.get('thresh', 30),
                    octaves=result.params.get('octaves', 3),
                    pattern_scale=result.params.get('pattern_scale', 1.0)
                )
            
            # 检测特征点
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp1, des1 = detector.detectAndCompute(gray1, None)
            
            # 绘制特征点
            img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
            
            # 特殊标记SIFT
            if algorithm == 'SIFT':
                # SIFT使用红色边框突出显示
                axes[0, i].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f'SIFT (基准算法)\n特征点: {len(kp1)}', fontsize=12, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
                # 添加红色边框
                for spine in axes[0, i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            else:
                axes[0, i].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f'{algorithm}\n特征点: {len(kp1)}', fontsize=12)
            
            axes[0, i].axis('off')
        
        # 第二行：性能指标对比雷达图
        ax_radar = axes[1, 0]
        
        # 准备雷达图数据
        categories = ['质量分数', '内点比例', '处理速度', '匹配数量']
        sift_result = best_results.get('SIFT')
        
        if sift_result:
            # 归一化数据到0-1范围
            max_quality = max([r.match_quality_score for r in best_results.values()])
            max_inlier = max([r.inlier_ratio for r in best_results.values()])
            min_time = min([r.processing_time for r in best_results.values()])
            max_matches = max([r.good_matches for r in best_results.values()])
            
            # SIFT数据
            sift_values = [
                sift_result.match_quality_score / max_quality,
                sift_result.inlier_ratio / max_inlier,
                min_time / sift_result.processing_time,  # 速度越快值越大
                sift_result.good_matches / max_matches
            ]
            
            # 创建简化的雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            sift_values += sift_values[:1]  # 闭合图形
            angles = np.concatenate((angles, [angles[0]]))
            
            ax_radar.plot(angles, sift_values, 'ro-', linewidth=2, label='SIFT', color='red')
            ax_radar.fill(angles, sift_values, alpha=0.25, color='red')
            
            # 添加其他算法对比
            colors = ['blue', 'green', 'orange']
            for j, algorithm in enumerate(['ORB', 'AKAZE', 'BRISK']):
                if algorithm in best_results:
                    result = best_results[algorithm]
                    values = [
                        result.match_quality_score / max_quality,
                        result.inlier_ratio / max_inlier,
                        min_time / result.processing_time,
                        result.good_matches / max_matches
                    ]
                    values += values[:1]
                    ax_radar.plot(angles, values, 'o-', linewidth=1, label=algorithm, color=colors[j])
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories)
            ax_radar.set_ylim(0, 1)
            ax_radar.legend()
            ax_radar.set_title('SIFT vs 其他算法性能雷达图', fontsize=12)
            ax_radar.grid(True)
        
        # 第二行：详细数值对比表
        comparison_text = "SIFT算法详细对比分析:\n\n"
        
        if 'SIFT' in best_results:
            sift_result = best_results['SIFT']
            comparison_text += f"SIFT算法 (基准):\n"
            comparison_text += f"  质量分数: {sift_result.match_quality_score:.3f}\n"
            comparison_text += f"  内点比例: {sift_result.inlier_ratio:.3f}\n"
            comparison_text += f"  处理时间: {sift_result.processing_time:.3f}s\n"
            comparison_text += f"  匹配数量: {sift_result.good_matches}\n\n"
            
            comparison_text += "与其他算法对比:\n"
            for algorithm in ['ORB', 'AKAZE', 'BRISK']:
                if algorithm in best_results:
                    result = best_results[algorithm]
                    quality_diff = result.match_quality_score - sift_result.match_quality_score
                    inlier_diff = result.inlier_ratio - sift_result.inlier_ratio
                    time_ratio = sift_result.processing_time / result.processing_time
                    
                    comparison_text += f"\n{algorithm}:\n"
                    comparison_text += f"  质量分数: {result.match_quality_score:.3f} "
                    comparison_text += f"({quality_diff:+.3f})\n"
                    comparison_text += f"  内点比例: {result.inlier_ratio:.3f} "
                    comparison_text += f"({inlier_diff:+.3f})\n"
                    comparison_text += f"  速度优势: {time_ratio:.1f}x\n"
        
        axes[1, 1].text(0.05, 0.95, comparison_text, ha='left', va='top', 
                       transform=axes[1, 1].transAxes, fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('数值对比分析', fontsize=12)
        
        # 第二行：SIFT优势分析
        sift_advantages = """SIFT算法特点分析:

优势:
• 尺度不变性强
• 旋转不变性好
• 光照变化鲁棒
• 特征点稳定
• 描述符区分度高
• 经典可靠算法

劣势:
• 计算复杂度高
• 处理时间较长
• 专利保护问题
• 内存占用较大

适用场景:
• 高精度要求应用
• 建筑物检测
• 图像配准
• 科研项目
• 离线处理应用"""
        
        axes[1, 2].text(0.05, 0.95, sift_advantages, ha='left', va='top', 
                       transform=axes[1, 2].transAxes, fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('SIFT算法特点', fontsize=12)
        
        # 第二行：推荐建议
        recommendations = """基于测试结果的建议:

当选择SIFT时:
• 优先考虑匹配精度
• 可接受较长处理时间
• 需要稳定可靠的结果
• 有充足的计算资源

SIFT替代方案:
• 高精度: AKAZE算法
• 高速度: ORB算法
• 平衡性: BRISK算法

参数优化建议:
• nfeatures: 1000-1500
• contrastThreshold: 0.03-0.04
• ratio_threshold: 0.65-0.7
• ransac_threshold: 3.0-4.0

结论:
SIFT仍是高精度应用的
优选算法，特别适合
建筑物检测等场景"""
        
        axes[1, 3].text(0.05, 0.95, recommendations, ha='left', va='top', 
                       transform=axes[1, 3].transAxes, fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        axes[1, 3].set_xlim(0, 1)
        axes[1, 3].set_ylim(0, 1)
        axes[1, 3].axis('off')
        axes[1, 3].set_title('选择建议', fontsize=12)
        
        # 第三行：匹配结果可视化对比
        for i, algorithm in enumerate(algorithms):
            if algorithm not in best_results:
                continue
                
            result = best_results[algorithm]
            
            # 重新进行匹配以获取可视化结果
            if algorithm == 'SIFT':
                detector = self.create_sift_detector(
                    nfeatures=result.params.get('nfeatures', 1000),
                    contrast_threshold=result.params.get('contrast_threshold', 0.04),
                    edge_threshold=result.params.get('edge_threshold', 10),
                    sigma=result.params.get('sigma', 1.6)
                )
            elif algorithm == 'ORB':
                detector = self.create_orb_detector(
                    nfeatures=result.params.get('nfeatures', 1000),
                    scale_factor=result.params.get('scale_factor', 1.2),
                    nlevels=result.params.get('nlevels', 8),
                    edge_threshold=result.params.get('edge_threshold', 31)
                )
            elif algorithm == 'AKAZE':
                detector = self.create_akaze_detector(
                    threshold=result.params.get('threshold', 0.001),
                    noctaves=result.params.get('noctaves', 4),
                    noctave_layers=result.params.get('noctave_layers', 4)
                )
            elif algorithm == 'BRISK':
                detector = self.create_brisk_detector(
                    thresh=result.params.get('thresh', 30),
                    octaves=result.params.get('octaves', 3),
                    pattern_scale=result.params.get('pattern_scale', 1.0)
                )
            
            # 检测和匹配
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                descriptor_type = 'binary' if algorithm in ['ORB', 'BRISK', 'AKAZE'] else 'float'
                matcher = self.create_matcher('BF', descriptor_type)
                
                matches = matcher.knnMatch(des1, des2, k=2)
                
                # 筛选好的匹配
                good_matches = []
                ratio_threshold = result.params.get('ratio_threshold', 0.7)
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)
                
                # 绘制匹配结果（只显示前20个最佳匹配）
                match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None, 
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # 调整图像大小以适应显示
                h, w = match_img.shape[:2]
                if w > 800:
                    scale = 800 / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    match_img = cv2.resize(match_img, (new_w, new_h))
                
                axes[2, i].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
                
                # 特殊标记SIFT
                if algorithm == 'SIFT':
                    axes[2, i].set_title(f'SIFT匹配结果\n匹配数: {len(good_matches)}, 内点率: {result.inlier_ratio:.3f}', 
                                        fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
                    # 添加红色边框
                    for spine in axes[2, i].spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                else:
                    axes[2, i].set_title(f'{algorithm}匹配结果\n匹配数: {len(good_matches)}, 内点率: {result.inlier_ratio:.3f}', 
                                        fontsize=10)
                
                axes[2, i].axis('off')
            else:
                axes[2, i].text(0.5, 0.5, f'{algorithm}\n匹配失败', ha='center', va='center', 
                               transform=axes[2, i].transAxes, fontsize=12)
                axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # 保存SIFT对比结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sift_comparison_path = os.path.join(self.output_dir, f"sift_detailed_comparison_{timestamp}.jpg")
        plt.savefig(sift_comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"SIFT详细对比可视化已保存: {sift_comparison_path}")
        return sift_comparison_path
    
    def save_results_to_file(self):
        """保存测试结果到文件"""
        if not self.test_results:
            print("没有测试结果可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON详细结果
        json_path = os.path.join(self.output_dir, f"angle_point_test_results_{timestamp}.json")
        
        # 转换结果为可序列化格式
        serializable_results = []
        for result in self.test_results:
            serializable_results.append({
                'algorithm': result.algorithm,
                'params': result.params,
                'keypoints1': result.keypoints1,
                'keypoints2': result.keypoints2,
                'matches': result.matches,
                'good_matches': result.good_matches,
                'inlier_ratio': result.inlier_ratio,
                'processing_time': result.processing_time,
                'homography_quality': result.homography_quality,
                'match_quality_score': result.match_quality_score
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_timestamp': timestamp,
                'total_tests': len(self.test_results),
                'results': serializable_results
            }, f, indent=2, ensure_ascii=False)
        
        # 生成文本报告
        report_path = os.path.join(self.output_dir, f"angle_point_test_report_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("角点匹配算法动态参数测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"测试总数: {len(self.test_results)}\n\n")
            
            # 按算法分组统计
            by_algorithm = {}
            for result in self.test_results:
                if result.algorithm not in by_algorithm:
                    by_algorithm[result.algorithm] = []
                by_algorithm[result.algorithm].append(result)
            
            f.write("各算法最佳结果:\n")
            f.write("-" * 30 + "\n")
            
            for algorithm, results in by_algorithm.items():
                best_result = max(results, key=lambda x: x.match_quality_score)
                f.write(f"\n{algorithm}算法:\n")
                f.write(f"  最佳配置: {best_result.params.get('config', 'Unknown')}\n")
                f.write(f"  参数设置: {best_result.params}\n")
                f.write(f"  特征点数: {best_result.keypoints1}/{best_result.keypoints2}\n")
                f.write(f"  良好匹配: {best_result.good_matches}\n")
                f.write(f"  内点比例: {best_result.inlier_ratio:.3f}\n")
                f.write(f"  处理时间: {best_result.processing_time:.3f}秒\n")
                f.write(f"  质量分数: {best_result.match_quality_score:.3f}\n")
            
            # 总体推荐
            best_overall = max(self.test_results, key=lambda x: x.match_quality_score)
            f.write(f"\n总体推荐:\n")
            f.write("-" * 20 + "\n")
            f.write(f"算法: {best_overall.algorithm}\n")
            f.write(f"配置: {best_overall.params.get('config', 'Unknown')}\n")
            f.write(f"参数: {best_overall.params}\n")
            f.write(f"质量分数: {best_overall.match_quality_score:.3f}\n")
        
        print(f"测试结果已保存:")
        print(f"  JSON文件: {json_path}")
        print(f"  文本报告: {report_path}")
        
        return json_path, report_path


def main():
    """主函数"""
    # 测试图像路径 - 请修改为您的实际图像路径
    img1_path = r'C:\Users\admin\Desktop\两期比对\两期比对\Snipaste_2025-09-08_15-39-45.png'
    img2_path = r'C:\Users\admin\Desktop\两期比对\两期比对\Snipaste_2025-09-08_15-39-16.png'
    
    try:
        # 创建测试器
        tester = AnglePointMatchingTester()
        
        # 运行综合测试
        print("开始角点匹配算法动态参数测试...")
        results = tester.run_comprehensive_test(img1_path, img2_path)
        
        if results:
            # 分析结果
            best_result = tester.analyze_results()
            
            # 创建对比可视化
            visualization_path = tester.create_comparison_visualization(img1_path, img2_path)
            
            # 创建SIFT详细对比可视化
            sift_comparison_path = tester.create_sift_comparison_visualization(img1_path, img2_path)
            
            # 保存结果
            json_path, report_path = tester.save_results_to_file()
            
            print(f"\n测试完成!")
            print(f"最佳算法: {best_result.algorithm}")
            print(f"最佳配置: {best_result.params.get('config', 'Unknown')}")
            print(f"质量分数: {best_result.match_quality_score:.3f}")
            print(f"可视化图像: {visualization_path}")
            print(f"SIFT详细对比可视化: {sift_comparison_path}")
            
        else:
            print("测试失败，请检查图像路径")
            
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 