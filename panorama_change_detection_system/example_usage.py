#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图变化检测系统使用示例
展示如何使用系统进行变化检测
"""

import os
from main import PanoramaChangeDetectionSystem
from config import get_default_config


def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建系统实例
    system = PanoramaChangeDetectionSystem()
    
    # 设置全景图路径
    panorama1_path = "path/to/your/first_panorama.jpg"
    panorama2_path = "path/to/your/second_panorama.jpg"
    
    # 检查文件是否存在
    if not (os.path.exists(panorama1_path) and os.path.exists(panorama2_path)):
        print("请修改panorama1_path和panorama2_path为实际的全景图路径")
        return
    
    # 处理全景图对
    results = system.process_panorama_pair(panorama1_path, panorama2_path)
    
    # 显示结果
    print(f"检测完成!")
    print(f"输出目录: {results['output_directory']}")
    print(f"检测到的变化区域: {results['statistics']['total_detections']}")
    print(f"有变化的面数: {results['statistics']['faces_with_changes']}")


def custom_config_example():
    """自定义配置示例"""
    print("=== 自定义配置示例 ===")
    
    # 获取默认配置
    config = get_default_config()
    
    # 自定义CUDA设置
    config.cuda.use_cuda = True  # 启用CUDA加速
    config.cuda.device_id = 0    # 使用第一个GPU
    
    # 自定义全景图分割参数
    config.panorama_splitter.cube_size = 2048  # 更高分辨率
    
    # 自定义预处理参数
    config.image_preprocessor.denoise_method = "bilateral"  # 双边滤波去噪
    config.image_preprocessor.enable_clahe = True  # 启用CLAHE
    config.image_preprocessor.clahe_clip_limit = 3.0
    
    # 自定义配准参数
    config.image_registration.akaze_threshold = 0.001  # 更敏感的特征检测
    config.image_registration.min_match_count = 15     # 更高的匹配要求
    
    # 自定义变化检测参数
    config.change_detector.threshold_method = "otsu"     # 使用Otsu阈值
    config.change_detector.min_contour_area = 1000      # 过滤小区域
    config.change_detector.morphology_operations = ["close", "open"]  # 形态学操作
    
    # 自定义可视化参数
    config.visualization.save_intermediate = True  # 保存中间结果
    config.visualization.output_dpi = 200         # 更高分辨率输出
    
    # 使用自定义配置创建系统
    system = PanoramaChangeDetectionSystem(config=config)
    
    # 后续处理同基本示例...
    print("自定义配置系统已创建，可以调用process_panorama_pair()处理图像")


def batch_processing_example():
    """批量处理示例"""
    print("=== 批量处理示例 ===")
    
    # 创建系统实例
    system = PanoramaChangeDetectionSystem()
    
    # 定义要处理的全景图对列表
    image_pairs = [
        ("period1/location_A.jpg", "period2/location_A.jpg"),
        ("period1/location_B.jpg", "period2/location_B.jpg"),
        ("period1/location_C.jpg", "period2/location_C.jpg"),
    ]
    
    # 批量处理
    all_results = []
    
    for i, (path1, path2) in enumerate(image_pairs):
        print(f"处理第 {i+1}/{len(image_pairs)} 对图像...")
        
        # 检查文件是否存在
        if not (os.path.exists(path1) and os.path.exists(path2)):
            print(f"跳过不存在的文件: {path1} or {path2}")
            continue
        
        try:
            # 处理当前图像对
            results = system.process_panorama_pair(path1, path2)
            all_results.append(results)
            
            print(f"  检测到 {results['statistics']['total_detections']} 个变化区域")
            
        except Exception as e:
            print(f"  处理失败: {e}")
            continue
    
    # 汇总所有结果
    total_detections = sum(r['statistics']['total_detections'] for r in all_results)
    print(f"\n批量处理完成:")
    print(f"  成功处理: {len(all_results)} 对图像")
    print(f"  总变化区域: {total_detections} 个")


def system_info_example():
    """系统信息示例"""
    print("=== 系统信息示例 ===")
    
    # 创建系统实例
    system = PanoramaChangeDetectionSystem()
    
    # 获取系统信息
    info = system.get_system_info()
    
    print("系统信息:")
    print(f"  版本: {info['version']}")
    print(f"  CUDA可用: {info['cuda_available']}")
    print(f"  模块已初始化: {info['modules_initialized']}")
    print(f"  输出目录: {info['output_directory']}")
    print(f"  支持格式: {info['supported_formats']}")
    
    # 显示内存信息
    if info['memory_info']['cuda']:
        cuda_info = info['memory_info']['cuda']
        print(f"  GPU内存使用: {cuda_info.get('used_bytes', 0)} bytes")


def advanced_usage_example():
    """高级使用示例"""
    print("=== 高级使用示例 ===")
    
    # 创建系统实例
    system = PanoramaChangeDetectionSystem()
    
    # 设置输入路径
    panorama1_path = "advanced_test/panorama1.jpg"  
    panorama2_path = "advanced_test/panorama2.jpg"
    
    if not (os.path.exists(panorama1_path) and os.path.exists(panorama2_path)):
        print("请准备测试图像文件")
        return
    
    # 执行处理
    results = system.process_panorama_pair(panorama1_path, panorama2_path)
    
    # 高级结果分析
    print("=== 详细结果分析 ===")
    
    # 分析配准质量
    registration_summary = results['registration']['summary']
    print(f"配准成功率: {registration_summary['success_rate']:.1%}")
    print(f"平均SSIM: {registration_summary['average_ssim']:.3f}")
    print(f"平均特征匹配数: {registration_summary['average_matches']:.0f}")
    
    # 分析检测结果
    detection_summary = results['change_detection']['summary']
    print(f"检测覆盖面数: {detection_summary['faces_with_changes']}/{detection_summary['total_faces_processed']}")
    print(f"高置信度检测: {detection_summary['high_confidence_detections']}")
    print(f"平均置信度: {detection_summary['avg_confidence']:.3f}")
    
    # 按面分析检测结果
    print("\n各面检测详情:")
    for face_name, result in results['mapped_results'].items():
        detections = result['detections']
        if detections:
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            avg_area = sum(d['area'] for d in detections) / len(detections)
            print(f"  {face_name}: {len(detections)} 个区域, "
                  f"平均置信度: {avg_conf:.3f}, 平均面积: {avg_area:.0f}")
    
    # 生成自定义报告
    print(f"\n处理结果保存在: {results['output_directory']}")
    print("包含以下文件:")
    print("  - final_panorama_with_detections.jpg  # 最终结果全景图")
    print("  - processing_overview_*.jpg           # 处理流程总览")
    print("  - detailed_report_*.json              # 详细JSON报告")
    print("  - cube_faces_with_mapped_detections/  # 检测结果立方体面")


def main():
    """主示例函数"""
    print("全景图变化检测系统 - 使用示例")
    print("="*50)
    
    # 运行各种示例
    examples = [
        ("基本使用", basic_usage_example),
        ("自定义配置", custom_config_example),
        ("批量处理", batch_processing_example),
        ("系统信息", system_info_example),
        ("高级使用", advanced_usage_example)
    ]
    
    for name, func in examples:
        print(f"\n{'-'*20} {name} {'-'*20}")
        try:
            func()
        except Exception as e:
            print(f"示例执行出错: {e}")
    
    print(f"\n{'-'*50}")
    print("所有示例展示完成!")
    print("请根据需要修改路径和参数来处理您自己的全景图数据。")


if __name__ == "__main__":
    main()
