#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
提供统一的可视化功能，包括检测结果、处理过程、统计信息等
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple, Optional
import logging

# 设置中文字体 - 支持多平台
import platform
import matplotlib.font_manager as fm

def setup_chinese_fonts():
    """配置中文字体支持"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        fonts = ['Heiti TC', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    elif system == 'Windows':  # Windows
        fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'DejaVu Sans']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans', 'Liberation Sans']
    
    # 尝试设置可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            print(f"使用中文字体: {font}")
            break
    else:
        print("警告: 未找到合适的中文字体，中文可能显示为方块")
        # 尝试使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
setup_chinese_fonts()

class VisualizationUtils:
    """可视化工具类"""
    
    def __init__(self, output_dir: str = "results", dpi: int = 150, 
                 figure_size: Tuple[int, int] = (20, 15)):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figure_size = figure_size
        
        # 默认颜色配置
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def save_processing_overview(self, results: Dict[str, Any], 
                               output_path: str = None) -> str:
        """保存处理过程总览"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"processing_overview_{timestamp}.jpg")
        
        # 创建大画布显示所有处理步骤
        fig, axes = plt.subplots(4, 6, figsize=(30, 20))
        fig.suptitle('全景图变化检测系统处理流程总览', fontsize=20, fontweight='bold')
        
        # 第一行：原始输入
        self._plot_panorama_inputs(axes[0], results)
        
        # 第二行：立方体分割结果
        self._plot_cube_faces(axes[1], results)
        
        # 第三行：预处理和配准结果
        self._plot_preprocessing_results(axes[2], results)
        
        # 第四行：变化检测和最终结果
        self._plot_detection_results(axes[3], results)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logging.info(f"处理总览已保存: {output_path}")
        return output_path
    
    def _plot_panorama_inputs(self, axes, results):
        """绘制全景图输入"""
        if 'panorama_1' in results:
            axes[0].imshow(cv2.cvtColor(results['panorama_1'], cv2.COLOR_BGR2RGB))
            axes[0].set_title('第一期全景图', fontsize=12)
            axes[0].axis('off')
        
        if 'panorama_2' in results:
            axes[1].imshow(cv2.cvtColor(results['panorama_2'], cv2.COLOR_BGR2RGB))
            axes[1].set_title('第二期全景图', fontsize=12)
            axes[1].axis('off')
        
        # 显示输入信息
        if 'input_info' in results:
            info_text = self._format_input_info(results['input_info'])
            axes[2].text(0.1, 0.5, info_text, ha='left', va='center', 
                        transform=axes[2].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        axes[2].set_title('输入信息', fontsize=12)
        axes[2].axis('off')
        
        # 填充剩余子图
        for i in range(3, 6):
            axes[i].axis('off')
    
    def _plot_cube_faces(self, axes, results):
        """绘制立方体面"""
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        
        for i, face_name in enumerate(face_names):
            if 'cube_faces_1' in results and face_name in results['cube_faces_1']:
                face_img = results['cube_faces_1'][face_name]
                axes[i].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f'立方体面: {face_name}', fontsize=10)
            axes[i].axis('off')
    
    def _plot_preprocessing_results(self, axes, results):
        """绘制预处理结果"""
        # 预处理前后对比
        if 'preprocessing' in results:
            preprocessing = results['preprocessing']
            
            # 显示一个面的预处理前后对比
            if 'before' in preprocessing and 'after' in preprocessing:
                face_name = list(preprocessing['before'].keys())[0]
                
                axes[0].imshow(cv2.cvtColor(preprocessing['before'][face_name], cv2.COLOR_BGR2RGB))
                axes[0].set_title(f'预处理前 ({face_name})', fontsize=10)
                axes[0].axis('off')
                
                axes[1].imshow(cv2.cvtColor(preprocessing['after'][face_name], cv2.COLOR_BGR2RGB))
                axes[1].set_title(f'预处理后 ({face_name})', fontsize=10)
                axes[1].axis('off')
        
        # 配准结果
        if 'registration' in results:
            registration = results['registration']
            
            for i, face_name in enumerate(list(registration.keys())[:4]):
                if i + 2 < len(axes):
                    face_result = registration[face_name]
                    if 'aligned_image' in face_result:
                        axes[i + 2].imshow(cv2.cvtColor(face_result['aligned_image'], cv2.COLOR_BGR2RGB))
                        axes[i + 2].set_title(f'配准后 ({face_name})', fontsize=10)
                    axes[i + 2].axis('off')
    
    def _plot_detection_results(self, axes, results):
        """绘制检测结果"""
        # 变化检测结果
        if 'change_detection' in results:
            change_detection = results['change_detection']
            
            for i, face_name in enumerate(list(change_detection.keys())[:3]):
                if i < len(axes):
                    face_result = change_detection[face_name]
                    if 'difference_image' in face_result:
                        axes[i].imshow(face_result['difference_image'], cmap='hot')
                        axes[i].set_title(f'变化检测 ({face_name})', fontsize=10)
                    axes[i].axis('off')
        
        # 最终全景图结果
        if 'final_result' in results:
            axes[3].imshow(cv2.cvtColor(results['final_result'], cv2.COLOR_BGR2RGB))
            axes[3].set_title('最终结果', fontsize=10)
            axes[3].axis('off')
        
        # 统计信息
        if 'statistics' in results:
            stats_text = self._format_statistics(results['statistics'])
            axes[4].text(0.1, 0.5, stats_text, ha='left', va='center', 
                        transform=axes[4].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            axes[4].set_title('检测统计', fontsize=12)
        axes[4].axis('off')
        
        # 处理参数
        if 'parameters' in results:
            params_text = self._format_parameters(results['parameters'])
            axes[5].text(0.1, 0.5, params_text, ha='left', va='center', 
                        transform=axes[5].transAxes, fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            axes[5].set_title('处理参数', fontsize=12)
        axes[5].axis('off')
    
    def save_cube_face_analysis(self, face_name: str, results: Dict[str, Any], 
                              output_path: str = None) -> str:
        """保存单个立方体面的详细分析"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"face_analysis_{face_name}_{timestamp}.jpg")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'立方体面 "{face_name}" 详细分析', fontsize=16, fontweight='bold')
        
        # 第一行：原始图像和预处理
        self._plot_face_preprocessing(axes[0], face_name, results)
        
        # 第二行：配准过程
        self._plot_face_registration(axes[1], face_name, results)
        
        # 第三行：变化检测
        self._plot_face_change_detection(axes[2], face_name, results)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logging.info(f"立方体面分析已保存: {output_path}")
        return output_path
    
    def _plot_face_preprocessing(self, axes, face_name, results):
        """绘制面的预处理过程"""
        if 'cube_faces_1' in results and face_name in results['cube_faces_1']:
            axes[0].imshow(cv2.cvtColor(results['cube_faces_1'][face_name], cv2.COLOR_BGR2RGB))
            axes[0].set_title('原始图像1', fontsize=12)
            axes[0].axis('off')
        
        if 'cube_faces_2' in results and face_name in results['cube_faces_2']:
            axes[1].imshow(cv2.cvtColor(results['cube_faces_2'][face_name], cv2.COLOR_BGR2RGB))
            axes[1].set_title('原始图像2', fontsize=12)
            axes[1].axis('off')
        
        if 'preprocessing' in results and 'after' in results['preprocessing']:
            if face_name in results['preprocessing']['after']:
                axes[2].imshow(cv2.cvtColor(results['preprocessing']['after'][face_name], cv2.COLOR_BGR2RGB))
                axes[2].set_title('预处理后', fontsize=12)
                axes[2].axis('off')
        
        # 预处理参数
        if 'preprocessing' in results and 'parameters' in results['preprocessing']:
            params = results['preprocessing']['parameters']
            params_text = f"""预处理参数:
去噪方法: {params.get('denoise_method', 'N/A')}
CLAHE限制: {params.get('clahe_clip_limit', 'N/A')}
网格大小: {params.get('clahe_grid_size', 'N/A')}
"""
            axes[3].text(0.1, 0.5, params_text, ha='left', va='center', 
                        transform=axes[3].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
            axes[3].set_title('预处理参数', fontsize=12)
        axes[3].axis('off')
    
    def _plot_face_registration(self, axes, face_name, results):
        """绘制面的配准过程"""
        if 'registration' in results and face_name in results['registration']:
            reg_result = results['registration'][face_name]
            
            # 配准前
            if 'image1' in reg_result:
                axes[0].imshow(cv2.cvtColor(reg_result['image1'], cv2.COLOR_BGR2RGB))
                axes[0].set_title('配准前图像1', fontsize=12)
                axes[0].axis('off')
            
            if 'image2' in reg_result:
                axes[1].imshow(cv2.cvtColor(reg_result['image2'], cv2.COLOR_BGR2RGB))
                axes[1].set_title('配准前图像2', fontsize=12)
                axes[1].axis('off')
            
            # 配准后
            if 'aligned_image' in reg_result:
                axes[2].imshow(cv2.cvtColor(reg_result['aligned_image'], cv2.COLOR_BGR2RGB))
                axes[2].set_title('配准后图像2', fontsize=12)
                axes[2].axis('off')
            
            # 配准信息
            if 'registration_info' in reg_result:
                info = reg_result['registration_info']
                info_text = f"""配准信息:
特征点数1: {info.get('keypoints1', 0)}
特征点数2: {info.get('keypoints2', 0)}
匹配点数: {info.get('matches', 0)}
内点比例: {info.get('inlier_ratio', 0):.2%}
配准质量: {'良好' if info.get('inlier_ratio', 0) > 0.6 else '一般'}
"""
                axes[3].text(0.1, 0.5, info_text, ha='left', va='center', 
                            transform=axes[3].transAxes, fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
                axes[3].set_title('配准信息', fontsize=12)
        axes[3].axis('off')
    
    def _plot_face_change_detection(self, axes, face_name, results):
        """绘制面的变化检测"""
        if 'change_detection' in results and face_name in results['change_detection']:
            det_result = results['change_detection'][face_name]
            
            # 差异图像
            if 'difference_image' in det_result:
                axes[0].imshow(det_result['difference_image'], cmap='hot')
                axes[0].set_title('差异图像', fontsize=12)
                axes[0].axis('off')
            
            # 二值化图像
            if 'binary_image' in det_result:
                axes[1].imshow(det_result['binary_image'], cmap='gray')
                axes[1].set_title('二值化图像', fontsize=12)
                axes[1].axis('off')
            
            # 检测结果
            if 'detection_result' in det_result:
                detection_img = det_result['detection_result']
                axes[2].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
                axes[2].set_title('检测结果', fontsize=12)
                axes[2].axis('off')
            
            # 检测统计
            if 'detection_stats' in det_result:
                stats = det_result['detection_stats']
                stats_text = f"""检测统计:
检测区域数: {stats.get('num_detections', 0)}
总变化像素: {stats.get('total_changed_pixels', 0)}
变化比例: {stats.get('change_ratio', 0):.2%}
平均区域面积: {stats.get('avg_area', 0):.0f}
最大区域面积: {stats.get('max_area', 0)}
"""
                axes[3].text(0.1, 0.5, stats_text, ha='left', va='center', 
                            transform=axes[3].transAxes, fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
                axes[3].set_title('检测统计', fontsize=12)
        axes[3].axis('off')
    
    def save_detection_bboxes_visualization(self, image: np.ndarray, 
                                          bboxes: List[Dict[str, Any]], 
                                          title: str, output_path: str = None) -> str:
        """保存检测框可视化"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = os.path.join(self.output_dir, f"detection_bboxes_{safe_title}_{timestamp}.jpg")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{title} - 检测结果可视化', fontsize=14, fontweight='bold')
        
        # 原始图像
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')
        
        # 带检测框的图像
        image_with_boxes = self.draw_bboxes_on_image(image.copy(), bboxes)
        axes[1].imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'检测结果 ({len(bboxes)} 个区域)', fontsize=12)
        axes[1].axis('off')
        
        # 统计信息
        if bboxes:
            areas = [bbox.get('area', 0) for bbox in bboxes]
            confidences = [bbox.get('confidence', 0) for bbox in bboxes]
            
            stats_text = f"""检测统计:
检测区域数: {len(bboxes)}
平均面积: {np.mean(areas):.0f} px²
最大面积: {np.max(areas):.0f} px²
最小面积: {np.min(areas):.0f} px²
平均置信度: {np.mean(confidences):.3f}
高置信度区域: {len([c for c in confidences if c > 0.8])}
"""
        else:
            stats_text = "未检测到变化区域"
        
        axes[2].text(0.1, 0.5, stats_text, ha='left', va='center', 
                    transform=axes[2].transAxes, fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        axes[2].set_title('统计信息', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logging.info(f"检测框可视化已保存: {output_path}")
        return output_path
    
    def draw_bboxes_on_image(self, image: np.ndarray, 
                           bboxes: List[Dict[str, Any]]) -> np.ndarray:
        """在图像上绘制检测框"""
        overlay = image.copy()
        
        for i, bbox in enumerate(bboxes):
            if 'bbox' in bbox:
                x, y, w, h = bbox['bbox']
                color = self.colors[i % len(self.colors)]
                confidence = bbox.get('confidence', 1.0)
                
                # 绘制半透明填充
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                
                # 绘制边框
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                
                # 绘制角点标记
                corner_size = 10
                cv2.line(image, (x, y), (x + corner_size, y), color, 4)
                cv2.line(image, (x, y), (x, y + corner_size), color, 4)
                cv2.line(image, (x + w, y), (x + w - corner_size, y), color, 4)
                cv2.line(image, (x + w, y), (x + w, y + corner_size), color, 4)
                cv2.line(image, (x, y + h), (x + corner_size, y + h), color, 4)
                cv2.line(image, (x, y + h), (x, y + h - corner_size), color, 4)
                cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), color, 4)
                cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), color, 4)
                
                # 添加标签
                label = f'ID:{i+1} ({confidence:.2f})'
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 标签背景
                label_bg_x1 = max(0, x - 5)
                label_bg_y1 = max(0, y - text_size[1] - 15)
                label_bg_x2 = min(image.shape[1], x + text_size[0] + 5)
                label_bg_y2 = y - 5
                
                cv2.rectangle(image, 
                            (label_bg_x1, label_bg_y1), 
                            (label_bg_x2, label_bg_y2), 
                            (0, 0, 0), -1)
                cv2.rectangle(image, 
                            (label_bg_x1, label_bg_y1), 
                            (label_bg_x2, label_bg_y2), 
                            color, 2)
                
                # 绘制标签文字
                cv2.putText(image, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 融合半透明效果
        cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)
        
        return image
    
    def _format_input_info(self, info: Dict[str, Any]) -> str:
        """格式化输入信息"""
        text = "输入信息:\n\n"
        text += f"第一期图像: {info.get('image1_name', 'N/A')}\n"
        text += f"第二期图像: {info.get('image2_name', 'N/A')}\n"
        text += f"图像尺寸: {info.get('image_size', 'N/A')}\n"
        text += f"立方体尺寸: {info.get('cube_size', 'N/A')}\n"
        text += f"处理时间: {info.get('processing_time', 'N/A')}\n"
        return text
    
    def _format_statistics(self, stats: Dict[str, Any]) -> str:
        """格式化统计信息"""
        text = "检测统计:\n\n"
        text += f"总检测区域: {stats.get('total_detections', 0)}\n"
        text += f"高置信度区域: {stats.get('high_confidence', 0)}\n"
        text += f"平均区域面积: {stats.get('avg_area', 0):.0f} px²\n"
        text += f"总变化像素: {stats.get('total_changed_pixels', 0)}\n"
        text += f"变化比例: {stats.get('change_ratio', 0):.2%}\n"
        return text
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """格式化处理参数"""
        text = "处理参数:\n\n"
        text += f"立方体尺寸: {params.get('cube_size', 1024)}\n"
        text += f"CUDA加速: {'是' if params.get('use_cuda', False) else '否'}\n"
        text += f"差异阈值: {params.get('threshold', 50)}\n"
        text += f"最小区域面积: {params.get('min_area', 500)}\n"
        text += f"形态学核大小: {params.get('kernel_size', 5)}\n"
        return text
    
    def save_json_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """保存JSON格式的详细报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"detailed_report_{timestamp}.json")
        
        # 清理results中的numpy数组，使其可JSON序列化
        serializable_results = self._make_json_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"JSON报告已保存: {output_path}")
        return output_path
    
    def _make_json_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # 对于图像数据，只保存形状信息而不是实际数据
            return {
                "type": "numpy_array",
                "shape": obj.shape,
                "dtype": str(obj.dtype)
            }
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
