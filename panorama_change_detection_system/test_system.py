#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图变化检测系统测试脚本
用于验证系统各模块是否正常工作
"""

import os
import sys
import numpy as np
import cv2
import logging
from datetime import datetime

# 添加当前目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config
from main import PanoramaChangeDetectionSystem


def create_test_panoramas(output_dir: str = "test_data"):
    """
    创建测试用的模拟全景图
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建两个测试全景图 (2:1比例)
    width, height = 2048, 1024
    
    print("创建测试全景图...")
    
    # 第一张全景图 - 基础场景
    panorama1 = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加渐变背景
    for y in range(height):
        for x in range(width):
            panorama1[y, x] = [
                int(128 + 127 * np.sin(x * np.pi / width)),
                int(128 + 127 * np.cos(y * np.pi / height)),
                int(128 + 127 * np.sin((x + y) * np.pi / (width + height)))
            ]
    
    # 添加一些几何形状作为"建筑物"
    cv2.rectangle(panorama1, (200, 200), (400, 400), (100, 100, 200), -1)
    cv2.circle(panorama1, (600, 300), 80, (200, 100, 100), -1)
    cv2.rectangle(panorama1, (800, 150), (1000, 350), (100, 200, 100), -1)
    
    # 第二张全景图 - 添加变化
    panorama2 = panorama1.copy()
    
    # 添加新的"建筑物"（变化区域）
    cv2.rectangle(panorama2, (1200, 250), (1400, 450), (200, 200, 100), -1)
    cv2.circle(panorama2, (1600, 400), 60, (100, 200, 200), -1)
    
    # 修改现有"建筑物"
    cv2.rectangle(panorama2, (200, 200), (450, 450), (150, 150, 250), -1)
    
    # 保存测试图像
    panorama1_path = os.path.join(output_dir, "test_panorama1.jpg")
    panorama2_path = os.path.join(output_dir, "test_panorama2.jpg")
    
    cv2.imwrite(panorama1_path, panorama1)
    cv2.imwrite(panorama2_path, panorama2)
    
    print(f"测试全景图已创建:")
    print(f"  - {panorama1_path}")
    print(f"  - {panorama2_path}")
    
    return panorama1_path, panorama2_path


def test_individual_modules():
    """测试各个模块的基本功能"""
    print("\n" + "="*60)
    print("测试各个模块...")
    print("="*60)
    
    try:
        # 创建测试全景图
        panorama1_path, panorama2_path = create_test_panoramas()
        
        # 读取测试图像
        panorama1 = cv2.imread(panorama1_path)
        panorama2 = cv2.imread(panorama2_path)
        
        config = get_default_config()
        config.panorama_splitter.cube_size = 512  # 使用较小尺寸加快测试
        
        # 测试1: 全景图分割模块
        print("1. 测试全景图分割模块...")
        from modules.panorama_splitter import PanoramaSplitter
        splitter = PanoramaSplitter(config.panorama_splitter, config.cuda)
        
        faces1, faces2 = splitter.split_two_panoramas(panorama1, panorama2, config.panorama_splitter.cube_size)
        print(f"   ✓ 分割成功，生成 {len(faces1)} 个立方体面")
        
        # 测试2: 图像预处理模块
        print("2. 测试图像预处理模块...")
        from modules.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor(config.image_preprocessor, config.cuda)
        
        preprocessed1, preprocessed2 = preprocessor.preprocess_two_face_sets(faces1, faces2)
        print(f"   ✓ 预处理成功，处理 {len(preprocessed1)} 个立方体面")
        
        # 测试3: 图像配准模块
        print("3. 测试图像配准模块...")
        from modules.image_registration import ImageRegistration
        registration = ImageRegistration(config.image_registration, config.cuda)
        
        aligned_faces, reg_info = registration.register_cube_faces(preprocessed1, preprocessed2)
        successful_registrations = len([f for f, info in reg_info.items() 
                                      if info['registration_info']['registration_success']])
        print(f"   ✓ 配准成功，{successful_registrations}/{len(faces1)} 个面配准成功")
        
        # 测试4: 变化检测模块
        print("4. 测试变化检测模块...")
        from modules.change_detector import ChangeDetector
        detector = ChangeDetector(config.change_detector, config.cuda)
        
        change_results = detector.detect_changes_in_faces(preprocessed1, aligned_faces)
        total_detections = sum(len(result['detections']) for result in change_results.values())
        print(f"   ✓ 变化检测成功，检测到 {total_detections} 个变化区域")
        
        # 测试5: 结果映射模块
        print("5. 测试结果映射模块...")
        from modules.result_mapper import ResultMapper
        mapper = ResultMapper(config.result_mapper, config.cuda)
        
        mapped_results = mapper.map_detections_to_original_faces(change_results, reg_info, faces2)
        panorama_result = mapper.reconstruct_panorama_with_detections(
            faces2, mapped_results, panorama2.shape[1], panorama2.shape[0]
        )
        print(f"   ✓ 结果映射成功，生成最终全景图 {panorama_result.shape}")
        
        print("\n所有模块测试通过! ✓")
        return True
        
    except Exception as e:
        print(f"\n模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_system():
    """测试完整系统流程"""
    print("\n" + "="*60)
    print("测试完整系统流程...")
    print("="*60)
    
    try:
        # 创建测试全景图
        panorama1_path, panorama2_path = create_test_panoramas()
        
        # 创建系统实例（使用较小配置加快测试）
        config = get_default_config()
        config.panorama_splitter.cube_size = 512
        config.change_detector.min_contour_area = 100
        
        system = PanoramaChangeDetectionSystem(config=config)
        
        print("开始完整流程处理...")
        start_time = datetime.now()
        
        # 执行完整处理流程
        results = system.process_panorama_pair(
            panorama1_path, 
            panorama2_path,
            save_intermediate=True
        )
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        # 显示结果
        print(f"\n完整系统测试完成! ✓")
        print(f"处理时间: {processing_time}")
        print(f"输出目录: {results['output_directory']}")
        print(f"检测到的变化区域: {results['statistics']['total_detections']}")
        print(f"有变化的面数: {results['statistics']['faces_with_changes']}")
        print(f"整体变化比例: {results['statistics']['overall_change_ratio']:.2%}")
        
        return True, results
        
    except Exception as e:
        print(f"\n完整系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_cuda_availability():
    """测试CUDA可用性"""
    print("\n" + "="*60)
    print("测试CUDA可用性...")
    print("="*60)
    
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        device.use()
        
        # 简单测试
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        
        print("✓ CUDA可用，CuPy正常工作")
        print(f"  GPU设备: {device}")
        print(f"  测试计算: {cp.asnumpy(c)}")
        
        # 内存信息
        mempool = cp.get_default_memory_pool()
        print(f"  GPU内存使用: {mempool.used_bytes()} bytes")
        
        return True
        
    except ImportError:
        print("⚠ CuPy未安装，将使用CPU模式")
        print("  安装命令: pip install cupy-cuda11x (根据CUDA版本选择)")
        return False
        
    except Exception as e:
        print(f"⚠ CUDA测试失败: {e}")
        print("  将使用CPU模式")
        return False


def test_dependencies():
    """测试依赖包"""
    print("\n" + "="*60)
    print("测试依赖包...")
    print("="*60)
    
    required_packages = [
        ('numpy', 'np'),
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('scikit-image', 'skimage'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n所有必需依赖包已安装 ✓")
        return True


def main():
    """主测试函数"""
    print("全景图变化检测系统 - 测试套件")
    print("="*60)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    all_tests_passed = True
    
    # 1. 测试依赖包
    if not test_dependencies():
        all_tests_passed = False
        print("\n请先安装所有必需的依赖包")
        return
    
    # 2. 测试CUDA
    cuda_available = test_cuda_availability()
    
    # 3. 测试各个模块
    if not test_individual_modules():
        all_tests_passed = False
    
    # 4. 测试完整系统
    system_success, results = test_full_system()
    if not system_success:
        all_tests_passed = False
    
    # 总结
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 所有测试通过!")
        print("系统已准备就绪，可以开始处理真实的全景图数据")
        
        if results:
            print(f"\n测试结果保存在: {results['output_directory']}")
            print("您可以查看生成的可视化结果和报告")
    else:
        print("❌ 部分测试失败")
        print("请检查错误信息并修复相关问题")
    
    print("="*60)


if __name__ == "__main__":
    main()
