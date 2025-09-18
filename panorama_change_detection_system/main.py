#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图变化检测系统主控制器
整合所有模块，提供完整的处理流程
"""

import os
import cv2
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List

from config import SystemConfig, get_default_config, save_config_to_file
from modules import (
    PanoramaSplitter, 
    ImagePreprocessor, 
    ImageRegistration, 
    ChangeDetector, 
    ResultMapper
)
from utils import VisualizationUtils


class PanoramaChangeDetectionSystem:
    """全景图变化检测系统主控制器"""
    
    def __init__(self, config: SystemConfig = None):
        """
        初始化系统
        
        Args:
            config: 系统配置，如果为None则使用默认配置
        """
        self.config = config or get_default_config()
        
        # 设置日志
        self._setup_logging()
        
        # 创建输出目录
        self.output_dir = self._create_output_directory()
        
        # 初始化各个模块
        self._initialize_modules()
        
        # 初始化可视化工具
        self.visualizer = VisualizationUtils(
            output_dir=self.output_dir,
            dpi=self.config.visualization.output_dpi,
            figure_size=self.config.visualization.figure_size
        )
        
        logging.info("全景图变化检测系统初始化完成")
        logging.info(f"输出目录: {self.output_dir}")
        
    def _setup_logging(self):
        """设置日志系统"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建文件处理器
        if hasattr(self, 'output_dir'):
            log_file = os.path.join(self.output_dir, 'system.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
    
    def _create_output_directory(self) -> str:
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.config.output_root, f"change_detection_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
        
        return output_dir
    
    def _initialize_modules(self):
        """初始化各个处理模块"""
        logging.info("初始化处理模块...")
        
        # 全景图分割模块
        self.splitter = PanoramaSplitter(
            config=self.config.panorama_splitter,
            cuda_config=self.config.cuda
        )
        
        # 图像预处理模块
        self.preprocessor = ImagePreprocessor(
            config=self.config.image_preprocessor,
            cuda_config=self.config.cuda
        )
        
        # 图像配准模块
        self.registration = ImageRegistration(
            config=self.config.image_registration,
            cuda_config=self.config.cuda
        )
        
        # 变化检测模块
        self.change_detector = ChangeDetector(
            config=self.config.change_detector,
            cuda_config=self.config.cuda
        )
        
        # 结果映射模块
        self.result_mapper = ResultMapper(
            config=self.config.result_mapper,
            cuda_config=self.config.cuda
        )
        
        logging.info("所有处理模块初始化完成")
    
    def process_panorama_pair(self, panorama1_path: str, panorama2_path: str,
                             save_intermediate: bool = True) -> Dict[str, Any]:
        """
        处理一对全景图进行变化检测
        
        Args:
            panorama1_path: 第一期全景图路径
            panorama2_path: 第二期全景图路径
            save_intermediate: 是否保存中间结果
            
        Returns:
            完整的处理结果字典
        """
        start_time = datetime.now()
        
        logging.info(f"开始处理全景图对: {os.path.basename(panorama1_path)} vs {os.path.basename(panorama2_path)}")
        
        # 保存配置
        config_path = os.path.join(self.output_dir, 'system_config.json')
        save_config_to_file(self.config, config_path)
        
        try:
            # 1. 加载全景图
            panorama1, panorama2 = self._load_panoramas(panorama1_path, panorama2_path)
            
            # 2. 全景图分割为立方体面
            cube_faces1, cube_faces2 = self._split_panoramas(panorama1, panorama2, save_intermediate)
            
            # 3. 图像预处理
            preprocessed_faces1, preprocessed_faces2 = self._preprocess_faces(
                cube_faces1, cube_faces2, save_intermediate
            )
            
            # 4. 图像配准
            aligned_faces, registration_info = self._register_faces(
                preprocessed_faces1, preprocessed_faces2, save_intermediate
            )
            
            # 5. 变化检测
            change_results = self._detect_changes(
                preprocessed_faces1, aligned_faces, save_intermediate
            )
            
            # 6. 结果映射与全景图重建（使用原图尺寸确保像素一致）
            final_panorama, mapped_results = self._map_results_and_reconstruct(
                change_results, registration_info, cube_faces1, 
                panorama1.shape[1], panorama1.shape[0], save_intermediate
            )
            
            # 7. 生成综合报告
            processing_time = datetime.now() - start_time
            results = self._compile_results(
                panorama1_path, panorama2_path, panorama1, panorama2,
                cube_faces1, cube_faces2, preprocessed_faces1, preprocessed_faces2,
                registration_info, change_results, mapped_results, final_panorama,
                processing_time
            )
            
            # 8. 生成可视化和报告
            if save_intermediate:
                self._generate_visualizations_and_reports(results)
            
            logging.info(f"全景图变化检测完成，耗时: {processing_time}")
            
            return results
            
        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 清理内存
            self._cleanup_memory()
    
    def _load_panoramas(self, path1: str, path2: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载全景图"""
        logging.info("加载全景图...")
        
        # 支持中文路径的图像加载
        def load_image_chinese_path(path):
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            return img
        
        panorama1 = load_image_chinese_path(path1)
        panorama2 = load_image_chinese_path(path2)
        
        # 验证全景图
        if not (self.splitter.validate_panorama(panorama1) and 
                self.splitter.validate_panorama(panorama2)):
            logging.warning("全景图验证警告，继续处理")
        
        logging.info(f"全景图加载完成: {panorama1.shape} vs {panorama2.shape}")
        
        return panorama1, panorama2
    
    def _split_panoramas(self, panorama1: np.ndarray, panorama2: np.ndarray,
                        save_intermediate: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """分割全景图为立方体面"""
        logging.info("分割全景图为立方体面...")
        
        cube_faces1, cube_faces2 = self.splitter.split_two_panoramas(
            panorama1, panorama2, self.config.panorama_splitter.cube_size
        )
        
        if save_intermediate:
            # 保存立方体面
            self.splitter.save_cube_faces(cube_faces1, self.output_dir, "period1")
            self.splitter.save_cube_faces(cube_faces2, self.output_dir, "period2")
        
        return cube_faces1, cube_faces2
    
    def _preprocess_faces(self, faces1: Dict[str, np.ndarray], faces2: Dict[str, np.ndarray],
                         save_intermediate: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """预处理立方体面"""
        logging.info("预处理立方体面...")
        
        preprocessed_faces1, preprocessed_faces2 = self.preprocessor.preprocess_two_face_sets(
            faces1, faces2
        )
        
        if save_intermediate:
            # 保存预处理后的立方体面
            self.splitter.save_cube_faces(preprocessed_faces1, self.output_dir, "preprocessed_period1")
            self.splitter.save_cube_faces(preprocessed_faces2, self.output_dir, "preprocessed_period2")
        
        return preprocessed_faces1, preprocessed_faces2
    
    def _register_faces(self, faces1: Dict[str, np.ndarray], faces2: Dict[str, np.ndarray],
                       save_intermediate: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """配准立方体面"""
        logging.info("配准立方体面...")
        
        aligned_faces, registration_info = self.registration.register_cube_faces(faces1, faces2)
        
        # 评估配准质量
        quality_results = self.registration.batch_evaluate_registration(faces1, aligned_faces)
        
        # 生成配准摘要
        registration_summary = self.registration.get_registration_summary(registration_info, quality_results)
        
        if save_intermediate:
            # 保存配准后的立方体面
            self.splitter.save_cube_faces(aligned_faces, self.output_dir, "aligned_period2")
            
            # 保存配准信息
            registration_json_path = os.path.join(self.output_dir, "registration_info.json")
            self.registration.save_registration_transforms(registration_info, registration_json_path)
            
            # 保存配准摘要
            summary_path = os.path.join(self.output_dir, "registration_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(registration_summary, f, indent=2, ensure_ascii=False)
        
        # 将质量结果添加到registration_info中
        for face_name, quality in quality_results.items():
            if face_name in registration_info:
                registration_info[face_name]['quality_metrics'] = quality
        
        registration_info['summary'] = registration_summary
        
        return aligned_faces, registration_info
    
    def _detect_changes(self, faces1: Dict[str, np.ndarray], faces2_aligned: Dict[str, np.ndarray],
                       save_intermediate: bool) -> Dict[str, Dict[str, Any]]:
        """检测变化区域"""
        logging.info("检测变化区域...")
        
        change_results = self.change_detector.detect_changes_in_faces(faces1, faces2_aligned)
        
        # 生成检测摘要
        detection_summary = self.change_detector.get_detection_summary(change_results)
        
        if save_intermediate:
            # 保存中间处理图像
            intermediate_dir = os.path.join(self.output_dir, "intermediate", "change_detection")
            os.makedirs(intermediate_dir, exist_ok=True)
            
            for face_name, result in change_results.items():
                # 保存差分图像
                diff_path = os.path.join(intermediate_dir, f"{face_name}_difference.jpg")
                cv2.imwrite(diff_path, result['difference_image'])
                
                # 保存二值化图像
                binary_path = os.path.join(intermediate_dir, f"{face_name}_binary.jpg")
                cv2.imwrite(binary_path, result['binary_image'])
                
                # 保存检测结果可视化
                detection_path = os.path.join(intermediate_dir, f"{face_name}_detection.jpg")
                cv2.imwrite(detection_path, result['detection_result'])
                
                # 为每个面生成详细分析图
                if self.config.visualization.save_intermediate:
                    self.visualizer.save_cube_face_analysis(face_name, {
                        'cube_faces_1': {face_name: faces1[face_name]},
                        'cube_faces_2': {face_name: faces2_aligned[face_name]},
                        'change_detection': {face_name: result}
                    })
            
            # 保存检测摘要
            summary_path = os.path.join(self.output_dir, "detection_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(detection_summary, f, indent=2, ensure_ascii=False)
        
        change_results['summary'] = detection_summary
        
        return change_results
    
    def _map_results_and_reconstruct(self, change_results: Dict[str, Dict[str, Any]], 
                                   registration_info: Dict[str, Dict[str, Any]],
                                   original_faces: Dict[str, np.ndarray],
                                   panorama_width: int, panorama_height: int,
                                   save_intermediate: bool) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
        """映射结果并重建全景图"""
        logging.info("映射检测结果并重建全景图...")
        
        # 映射检测结果到原始坐标
        mapped_results = self.result_mapper.map_detections_to_original_faces(
            change_results, registration_info, original_faces
        )
        
        # 重建包含检测结果的全景图
        final_panorama = self.result_mapper.reconstruct_panorama_with_detections(
            original_faces, mapped_results, panorama_width, panorama_height
        )
        
        # 在全景图上添加检测摘要
        final_panorama_with_summary = self.result_mapper.create_detection_summary_on_panorama(
            final_panorama, mapped_results, self.config.panorama_splitter.cube_size
        )
        
        if save_intermediate:
            # 保存映射后的立方体面
            self.result_mapper.save_cube_faces_with_detections(mapped_results, self.output_dir)
            
            # 保存最终全景图
            final_panorama_path = os.path.join(self.output_dir, "final_panorama_with_detections.jpg")
            cv2.imwrite(final_panorama_path, final_panorama_with_summary)
            
            # 保存映射统计
            mapping_stats = self.result_mapper.get_mapping_statistics(mapped_results)
            stats_path = os.path.join(self.output_dir, "mapping_statistics.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(mapping_stats, f, indent=2, ensure_ascii=False)
        
        return final_panorama_with_summary, mapped_results
    
    def _compile_results(self, panorama1_path: str, panorama2_path: str,
                        panorama1: np.ndarray, panorama2: np.ndarray,
                        cube_faces1: Dict[str, np.ndarray], cube_faces2: Dict[str, np.ndarray],
                        preprocessed_faces1: Dict[str, np.ndarray], preprocessed_faces2: Dict[str, np.ndarray],
                        registration_info: Dict[str, Dict[str, Any]], 
                        change_results: Dict[str, Dict[str, Any]],
                        mapped_results: Dict[str, Dict[str, Any]],
                        final_panorama: np.ndarray,
                        processing_time) -> Dict[str, Any]:
        """编译完整的处理结果"""
        
        return {
            'input_info': {
                'image1_path': panorama1_path,
                'image2_path': panorama2_path,
                'image1_name': os.path.basename(panorama1_path),
                'image2_name': os.path.basename(panorama2_path),
                'image_size': panorama1.shape[:2],
                'cube_size': self.config.panorama_splitter.cube_size,
                'processing_time': str(processing_time)
            },
            'panorama_1': panorama1,
            'panorama_2': panorama2,
            'cube_faces_1': cube_faces1,
            'cube_faces_2': cube_faces2,
            'preprocessing': {
                'before': {'period1': cube_faces1, 'period2': cube_faces2},
                'after': {'period1': preprocessed_faces1, 'period2': preprocessed_faces2},
                'parameters': self.preprocessor.get_preprocessing_parameters()
            },
            'registration': registration_info,
            'change_detection': change_results,
            'mapped_results': mapped_results,
            'final_result': final_panorama,
            'statistics': self._compute_overall_statistics(change_results, mapped_results),
            'parameters': self._get_system_parameters(),
            'output_directory': self.output_dir
        }
    
    def _compute_overall_statistics(self, change_results: Dict[str, Dict[str, Any]], 
                                  mapped_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算整体统计信息"""
        # 从change_results中获取summary（如果存在）
        if 'summary' in change_results:
            detection_summary = change_results['summary']
        else:
            detection_summary = self.change_detector.get_detection_summary(
                {k: v for k, v in change_results.items() if k != 'summary'}
            )
        
        # 获取映射统计
        mapping_stats = self.result_mapper.get_mapping_statistics(mapped_results)
        
        # 计算置信度分布
        all_detections = []
        for face_results in mapped_results.values():
            all_detections.extend(face_results['detections'])
        
        high_conf = len([d for d in all_detections if d['confidence'] > 0.8])
        med_conf = len([d for d in all_detections if 0.6 <= d['confidence'] <= 0.8])
        low_conf = len([d for d in all_detections if d['confidence'] < 0.6])
        
        return {
            'total_detections': detection_summary['total_detections'],
            'faces_with_changes': detection_summary['faces_with_changes'],
            'faces_without_changes': detection_summary['faces_without_changes'],
            'overall_change_ratio': detection_summary['overall_change_ratio'],
            'avg_area': mapping_stats['overall_avg_area'],
            'avg_confidence': mapping_stats['overall_avg_confidence'],
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'detection_distribution': mapping_stats['detection_distribution']
        }
    
    def _get_system_parameters(self) -> Dict[str, Any]:
        """获取系统参数"""
        return {
            'cube_size': self.config.panorama_splitter.cube_size,
            'use_cuda': self.config.cuda.use_cuda,
            'threshold': self.config.change_detector.fixed_threshold,
            'min_area': self.config.change_detector.min_contour_area,
            'kernel_size': self.config.change_detector.close_kernel_size,
            'preprocessing_enabled': self.config.image_preprocessor.enable_clahe,
            'registration_method': 'AKAZE',
            'detection_method': self.config.change_detector.diff_method
        }
    
    def _generate_visualizations_and_reports(self, results: Dict[str, Any]):
        """生成可视化和报告"""
        logging.info("生成可视化和报告...")
        
        # 生成处理总览
        overview_path = self.visualizer.save_processing_overview(results)
        
        # 生成详细JSON报告
        json_report_path = self.visualizer.save_json_report(results)
        
        # 为每个有检测结果的面生成详细可视化
        if 'mapped_results' in results:
            for face_name, face_result in results['mapped_results'].items():
                if face_result['detections']:
                    self.visualizer.save_detection_bboxes_visualization(
                        face_result['visualization'],
                        face_result['detections'],
                        f"Face_{face_name}"
                    )
        
        logging.info(f"可视化和报告生成完成")
        logging.info(f"  - 处理总览: {os.path.basename(overview_path)}")
        logging.info(f"  - JSON报告: {os.path.basename(json_report_path)}")
    
    def _cleanup_memory(self):
        """清理各模块内存"""
        self.splitter.cuda_utils.cleanup_memory()
        self.preprocessor.cleanup_memory()
        self.registration.cleanup_memory()
        self.change_detector.cleanup_memory()
        self.result_mapper.cleanup_memory()
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'version': '1.0.0',
            'cuda_available': self.config.cuda.use_cuda,
            'modules_initialized': True,
            'output_directory': self.output_dir,
            'config': self.config.__dict__,
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp'],
            'memory_info': {
                'splitter': self.splitter.get_memory_usage(),
                'cuda': self.splitter.cuda_utils.get_memory_info() if self.config.cuda.use_cuda else None
            }
        }


def main():
    """主函数示例"""
    # 示例用法
    system = PanoramaChangeDetectionSystem()
    
    # 示例全景图路径（请根据实际情况修改）
    panorama1_path = "/Users/chenyu/Desktop/twocomparsion/test/20250910164040_0002_V.jpeg"
    panorama2_path = "/Users/chenyu/Desktop/twocomparsion/test/20250910164151_0003_V.jpeg"
    
    if os.path.exists(panorama1_path) and os.path.exists(panorama2_path):
        try:
            results = system.process_panorama_pair(panorama1_path, panorama2_path)
            
            print("\n" + "="*60)
            print("全景图变化检测完成!")
            print("="*60)
            print(f"输出目录: {results['output_directory']}")
            print(f"检测到的变化区域总数: {results['statistics']['total_detections']}")
            print(f"有变化的面数: {results['statistics']['faces_with_changes']}")
            print(f"整体变化比例: {results['statistics']['overall_change_ratio']:.2%}")
            print("="*60)
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("请确保示例全景图文件存在：")
        print(f"  - {panorama1_path}")
        print(f"  - {panorama2_path}")
        print("\n您可以修改main()函数中的路径来使用您自己的全景图。")


if __name__ == "__main__":
    main()
