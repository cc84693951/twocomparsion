#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像配准模块
使用AKAZE算法进行特征检测和匹配，计算单应性矩阵进行透视变换
支持掩码处理和变换记录
"""

import cv2
import numpy as np
import logging
import json
from typing import Dict, Tuple, Optional, Union, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cuda_utils import CUDAUtils, ensure_cuda_available, cp, CUDA_AVAILABLE
from config import ImageRegistrationConfig, CUDAConfig


class ImageRegistration:
    """图像配准器"""
    
    def __init__(self, config: ImageRegistrationConfig = None,
                 cuda_config: CUDAConfig = None):
        self.config = config or ImageRegistrationConfig()
        self.cuda_config = cuda_config or CUDAConfig()
        
        # 初始化CUDA工具
        self.cuda_utils = CUDAUtils(
            use_cuda=self.cuda_config.use_cuda,
            device_id=self.cuda_config.device_id
        )
        
        # 初始化AKAZE检测器
        self.akaze = cv2.AKAZE_create(
            descriptor_type=self.config.akaze_descriptor_type,
            descriptor_size=self.config.akaze_descriptor_size,
            descriptor_channels=self.config.akaze_descriptor_channels,
            threshold=self.config.akaze_threshold,
            nOctaves=self.config.akaze_nOctaves,
            nOctaveLayers=self.config.akaze_nOctaveLayers
        )
        
        # 初始化特征匹配器
        if self.config.matcher_type == "BF":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # FLANN
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                               table_number=6,
                               key_size=12,
                               multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        logging.info(f"图像配准器初始化完成，CUDA: {'启用' if self.cuda_utils.use_cuda else '禁用'}")
    
    def register_cube_faces(self, faces1: Dict[str, np.ndarray], 
                           faces2: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], 
                                                                  Dict[str, Dict[str, Any]]]:
        """
        批量配准立方体面
        
        Args:
            faces1: 第一组立方体面（参考图像）
            faces2: 第二组立方体面（待配准图像）
            
        Returns:
            (配准后的faces2, 配准信息字典)
        """
        logging.info(f"开始批量配准 {len(faces1)} 个立方体面")
        
        aligned_faces = {}
        registration_info = {}
        
        for face_name in faces1.keys():
            if face_name in faces2:
                logging.debug(f"配准立方体面: {face_name}")
                
                # 配准单个面
                aligned_face, transform_matrix, reg_info = self.register_single_pair(
                    faces1[face_name], faces2[face_name], face_name
                )
                
                aligned_faces[face_name] = aligned_face
                registration_info[face_name] = {
                    'transform_matrix': transform_matrix,
                    'registration_info': reg_info,
                    'image1': faces1[face_name],
                    'image2': faces2[face_name],
                    'aligned_image': aligned_face
                }
            else:
                logging.warning(f"立方体面 {face_name} 在第二组中不存在")
        
        # 详细日志记录配准结果
        logging.info("=" * 50)
        logging.info("图像配准详细结果:")
        
        successful_registrations = 0
        failed_registrations = 0
        
        for face_name, reg_result in registration_info.items():
            # 跳过summary等非面数据
            if face_name == 'summary' or not isinstance(reg_result, dict) or 'registration_info' not in reg_result:
                continue
                
            reg_info = reg_result['registration_info']
            success = reg_info.get('registration_success', False)
            keypoints1 = reg_info.get('keypoints1', 0)
            keypoints2 = reg_info.get('keypoints2', 0)
            matches = reg_info.get('matches', 0)
            inlier_ratio = reg_info.get('inlier_ratio', 0.0)
            
            if success:
                successful_registrations += 1
                logging.info(f"  {face_name}: ✓ 配准成功")
            else:
                failed_registrations += 1
                logging.info(f"  {face_name}: ✗ 配准失败")
            
            logging.info(f"    特征点: img1={keypoints1}, img2={keypoints2}")
            logging.info(f"    匹配点: {matches}, 内点比例: {inlier_ratio:.3f}")
        
        logging.info(f"配准摘要: 成功 {successful_registrations} 个, 失败 {failed_registrations} 个")
        logging.info("=" * 50)
        
        return aligned_faces, registration_info
    
    def register_single_pair(self, img1: np.ndarray, img2: np.ndarray, face_name: str = "") -> Tuple[np.ndarray, 
                                                                               Optional[np.ndarray],
                                                                               Dict[str, Any]]:
        """
        配准单对图像
        
        Args:
            img1: 参考图像
            img2: 待配准图像
            
        Returns:
            (配准后的img2, 变换矩阵, 配准信息)
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 1. 特征检测
        kp1, des1 = self.akaze.detectAndCompute(gray1, None)
        kp2, des2 = self.akaze.detectAndCompute(gray2, None)
        
        face_info = f" ({face_name})" if face_name else ""
        logging.debug(f"检测到特征点{face_info}: img1={len(kp1)}, img2={len(kp2)}")
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logging.warning(f"面 {face_name} 特征点不足，返回原图像" if face_name else "特征点不足，返回原图像")
            return img2, None, {
                "keypoints1": len(kp1) if kp1 else 0,
                "keypoints2": len(kp2) if kp2 else 0,
                "matches": 0,
                "inlier_ratio": 0.0,
                "registration_success": False
            }
        
        # 2. 特征匹配
        good_matches = self.match_features(des1, des2)
        
        logging.debug(f"良好匹配点{face_info}: {len(good_matches)}")
        
        if len(good_matches) < self.config.min_match_count:
            logging.warning(f"面 {face_name} 良好匹配点不足 {self.config.min_match_count} 个" if face_name else f"良好匹配点不足 {self.config.min_match_count} 个")
            return img2, None, {
                "keypoints1": len(kp1),
                "keypoints2": len(kp2),
                "matches": len(good_matches),
                "inlier_ratio": 0.0,
                "registration_success": False
            }
        
        # 3. 计算单应性矩阵
        transform_matrix, inlier_mask = self.compute_homography(kp1, kp2, good_matches)
        
        if transform_matrix is None:
            logging.warning(f"面 {face_name} 无法计算单应性矩阵" if face_name else "无法计算单应性矩阵")
            return img2, None, {
                "keypoints1": len(kp1),
                "keypoints2": len(kp2),
                "matches": len(good_matches),
                "inlier_ratio": 0.0,
                "registration_success": False
            }
        
        # 4. 应用透视变换
        h, w = img1.shape[:2]
        aligned_img = cv2.warpPerspective(img2, transform_matrix, (w, h))
        
        # 5. 掩码处理
        aligned_img = self.handle_mask(aligned_img)
        
        # 6. 计算配准质量指标
        inlier_ratio = np.sum(inlier_mask) / len(inlier_mask) if inlier_mask is not None else 0
        
        registration_info = {
            "keypoints1": len(kp1),
            "keypoints2": len(kp2),
            "matches": len(good_matches),
            "inliers": int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
            "inlier_ratio": float(inlier_ratio),
            "registration_success": True,
            "transform_matrix": transform_matrix.tolist() if transform_matrix is not None else None
        }
        
        logging.debug(f"面 {face_name} 配准成功，内点比例: {inlier_ratio:.2%}" if face_name else f"配准成功，内点比例: {inlier_ratio:.2%}")
        
        return aligned_img, transform_matrix, registration_info
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        """
        特征匹配
        
        Args:
            des1: 第一组描述符
            des2: 第二组描述符
            
        Returns:
            良好的匹配点列表
        """
        # 进行knn匹配
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # 应用比率测试筛选良好匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.match_ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                          good_matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], 
                                                                  Optional[np.ndarray]]:
        """
        计算单应性矩阵
        
        Args:
            kp1: 第一组关键点
            kp2: 第二组关键点
            good_matches: 良好匹配列表
            
        Returns:
            (单应性矩阵, 内点掩码)
        """
        if len(good_matches) < 4:
            return None, None
        
        # 提取匹配点坐标
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        try:
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                confidence=self.config.ransac_confidence,
                maxIters=self.config.ransac_max_iterations
            )
            
            return homography, mask.ravel() if mask is not None else None
            
        except Exception as e:
            logging.error(f"计算单应性矩阵失败: {e}")
            return None, None
    
    def handle_mask(self, image: np.ndarray) -> np.ndarray:
        """
        处理透视变换后的掩码（黑边）
        
        Args:
            image: 透视变换后的图像
            
        Returns:
            处理后的图像
        """
        # 创建有效区域掩码
        mask = self.create_valid_mask(image)
        
        # 转换为uint8格式供OpenCV使用
        mask = mask.astype(np.uint8) * 255
        
        # 形态学操作清理掩码
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.config.mask_morphology_kernel_size, self.config.mask_morphology_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 转换回boolean掩码
        mask = mask > 0
        
        # 应用掩码（将无效区域设为0）
        if len(image.shape) == 3:
            image[~mask] = [0, 0, 0]
        else:
            image[~mask] = 0
        
        return image
    
    def create_valid_mask(self, image: np.ndarray) -> np.ndarray:
        """
        创建有效区域掩码
        
        Args:
            image: 输入图像
            
        Returns:
            有效区域掩码
        """
        if len(image.shape) == 3:
            # 彩色图像：检查所有通道
            mask = np.any(image > self.config.mask_threshold, axis=2)
        else:
            # 灰度图像
            mask = image > self.config.mask_threshold
        
        return mask.astype(bool)
    
    def evaluate_registration_quality(self, img1: np.ndarray, img2_aligned: np.ndarray, 
                                    mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估配准质量
        
        Args:
            img1: 参考图像
            img2_aligned: 配准后的图像
            mask: 有效区域掩码
            
        Returns:
            质量评估指标
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY) if len(img2_aligned.shape) == 3 else img2_aligned
        
        # 如果没有提供掩码，创建一个
        if mask is None:
            mask = self.create_valid_mask(img2_aligned)
        
        # 只在有效区域计算指标
        gray1_masked = gray1[mask]
        gray2_masked = gray2[mask]
        
        if len(gray1_masked) == 0:
            return {
                'ssim': 0.0,
                'ncc': 0.0,
                'mse': float('inf'),
                'psnr': 0.0
            }
        
        # 结构相似性指数 (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(gray1_masked.reshape(int(np.sqrt(len(gray1_masked))), -1), 
                            gray2_masked.reshape(int(np.sqrt(len(gray2_masked))), -1))
        except:
            ssim_score = self._compute_simple_ssim(gray1_masked, gray2_masked)
        
        # 归一化互相关 (NCC)
        ncc_score = self._compute_ncc(gray1_masked, gray2_masked)
        
        # 均方误差 (MSE)
        mse = np.mean((gray1_masked.astype(np.float32) - gray2_masked.astype(np.float32)) ** 2)
        
        # 峰值信噪比 (PSNR)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        return {
            'ssim': float(ssim_score),
            'ncc': float(ncc_score),
            'mse': float(mse),
            'psnr': float(psnr)
        }
    
    def _compute_simple_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """简单的SSIM计算"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    def _compute_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算归一化互相关"""
        img1_norm = (img1 - np.mean(img1)) / np.std(img1) if np.std(img1) > 0 else img1
        img2_norm = (img2 - np.mean(img2)) / np.std(img2) if np.std(img2) > 0 else img2
        return np.mean(img1_norm * img2_norm)
    
    def save_registration_transforms(self, registration_info: Dict[str, Dict[str, Any]], 
                                   output_path: str) -> str:
        """
        保存配准变换信息到JSON文件
        
        Args:
            registration_info: 配准信息字典
            output_path: 输出文件路径
            
        Returns:
            保存的文件路径
        """
        # 清理registration_info，使其可JSON序列化
        serializable_info = {}
        
        def make_serializable(obj):
            """递归转换对象为可JSON序列化格式"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif obj is None:
                return None
            else:
                return obj
        
        for face_name, face_info in registration_info.items():
            serializable_info[face_name] = {
                'registration_info': make_serializable(face_info['registration_info']),
                'transform_matrix': make_serializable(face_info.get('transform_matrix', None)),
                'image1_shape': face_info['image1'].shape if 'image1' in face_info else None,
                'image2_shape': face_info['image2'].shape if 'image2' in face_info else None,
                'aligned_image_shape': face_info['aligned_image'].shape if 'aligned_image' in face_info else None
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, indent=2, ensure_ascii=False)
        
        logging.info(f"配准变换信息已保存: {output_path}")
        return output_path
    
    def load_registration_transforms(self, json_path: str) -> Dict[str, Dict[str, Any]]:
        """
        从JSON文件加载配准变换信息
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            配准变换信息字典
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            transforms_data = json.load(f)
        
        # 转换变换矩阵回numpy数组
        for face_name, face_data in transforms_data.items():
            if face_data.get('transform_matrix') is not None:
                face_data['transform_matrix'] = np.array(face_data['transform_matrix'])
        
        logging.info(f"配准变换信息已加载: {json_path}")
        return transforms_data
    
    def apply_saved_transform(self, image: np.ndarray, 
                            transform_matrix: np.ndarray, 
                            target_shape: Tuple[int, int]) -> np.ndarray:
        """
        应用保存的变换矩阵
        
        Args:
            image: 待变换图像
            transform_matrix: 变换矩阵
            target_shape: 目标图像尺寸 (width, height)
            
        Returns:
            变换后的图像
        """
        if transform_matrix is None:
            return image
        
        transformed = cv2.warpPerspective(image, transform_matrix, target_shape)
        transformed = self.handle_mask(transformed)
        
        return transformed
    
    def batch_evaluate_registration(self, faces1: Dict[str, np.ndarray], 
                                  aligned_faces: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        批量评估配准质量
        
        Args:
            faces1: 参考立方体面
            aligned_faces: 配准后的立方体面
            
        Returns:
            每个面的质量评估结果
        """
        quality_results = {}
        
        for face_name in faces1.keys():
            if face_name in aligned_faces:
                mask = self.create_valid_mask(aligned_faces[face_name])
                quality = self.evaluate_registration_quality(
                    faces1[face_name], 
                    aligned_faces[face_name], 
                    mask
                )
                quality_results[face_name] = quality
                
                logging.debug(f"{face_name} 配准质量: SSIM={quality['ssim']:.3f}, "
                             f"NCC={quality['ncc']:.3f}, PSNR={quality['psnr']:.1f}")
        
        return quality_results
    
    def get_registration_summary(self, registration_info: Dict[str, Dict[str, Any]], 
                               quality_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        生成配准总结
        
        Args:
            registration_info: 配准信息
            quality_results: 质量评估结果
            
        Returns:
            配准总结
        """
        successful_registrations = []
        failed_registrations = []
        total_keypoints1 = 0
        total_keypoints2 = 0
        total_matches = 0
        total_inliers = 0
        
        ssim_scores = []
        ncc_scores = []
        psnr_scores = []
        
        for face_name, reg_info in registration_info.items():
            info = reg_info['registration_info']
            
            if info['registration_success']:
                successful_registrations.append(face_name)
                total_keypoints1 += info['keypoints1']
                total_keypoints2 += info['keypoints2']
                total_matches += info['matches']
                total_inliers += info.get('inliers', 0)
                
                if face_name in quality_results:
                    quality = quality_results[face_name]
                    ssim_scores.append(quality['ssim'])
                    ncc_scores.append(quality['ncc'])
                    psnr_scores.append(quality['psnr'])
            else:
                failed_registrations.append(face_name)
        
        return {
            'total_faces': len(registration_info),
            'successful_registrations': len(successful_registrations),
            'failed_registrations': len(failed_registrations),
            'success_rate': len(successful_registrations) / len(registration_info) if registration_info else 0,
            'successful_faces': successful_registrations,
            'failed_faces': failed_registrations,
            'average_keypoints1': total_keypoints1 / len(successful_registrations) if successful_registrations else 0,
            'average_keypoints2': total_keypoints2 / len(successful_registrations) if successful_registrations else 0,
            'average_matches': total_matches / len(successful_registrations) if successful_registrations else 0,
            'average_inliers': total_inliers / len(successful_registrations) if successful_registrations else 0,
            'average_ssim': np.mean(ssim_scores) if ssim_scores else 0,
            'average_ncc': np.mean(ncc_scores) if ncc_scores else 0,
            'average_psnr': np.mean(psnr_scores) if psnr_scores else 0,
            'quality_scores': {
                'ssim': ssim_scores,
                'ncc': ncc_scores,
                'psnr': psnr_scores
            }
        }
    
    def cleanup_memory(self):
        """清理内存"""
        if self.cuda_utils.use_cuda:
            self.cuda_utils.cleanup_memory()
