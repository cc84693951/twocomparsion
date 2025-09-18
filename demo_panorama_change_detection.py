#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景图像变化检测系统 - 演示脚本
展示如何使用 PanoramaChangeDetectionSystem 进行完整的检测流程
"""

import os
import sys
from panorama_change_detection_system import PanoramaChangeDetectionSystem


def demo_basic_usage():
    """基础用法演示"""
    print("=" * 60)
    print("🚀 全景图像变化检测系统 - 基础用法演示")
    print("=" * 60)
    
    # 图像路径配置 - 请根据实际情况修改
    panorama1_path = "test/20250910164040_0002_V.jpeg"  # 第一期全景图
    panorama2_path = "test/20250910164151_0003_V.jpeg"  # 第二期全景图
    
    # 创建检测系统
    system = PanoramaChangeDetectionSystem(
        output_dir="demo_results",
        use_cuda=True  # 启用GPU加速（如果可用）
    )
    
    # 检查输入文件
    if not os.path.exists(panorama1_path):
        print(f"❌ 第一期图像不存在: {panorama1_path}")
        return False
    
    if not os.path.exists(panorama2_path):
        print(f"❌ 第二期图像不存在: {panorama2_path}")
        return False
    
    try:
        # 执行完整检测流程
        results = system.process_panorama_pair(panorama1_path, panorama2_path)
        
        if results and results['processing_successful']:
            print("\n🎉 检测完成！结果摘要:")
            print(f"   📊 处理的立方体面: {results['total_faces_processed']}")
            print(f"   🔍 发现变化的面: {results['faces_with_detections']}")
            print(f"   📍 总检测区域: {results['total_detection_count']}")
            print(f"   🗺️ 全景图检测框: {results['panorama_bboxes_count']}")
            
            print(f"\n📁 输出文件位置:")
            for key, path in results['output_files'].items():
                print(f"   • {key}: {path}")
            
            return True
        else:
            print("❌ 检测失败")
            return False
            
    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def demo_custom_config():
    """自定义配置演示"""
    print("\n" + "=" * 60)
    print("⚙️ 自定义配置演示")
    print("=" * 60)
    
    # 创建自定义配置的系统
    system = PanoramaChangeDetectionSystem(
        output_dir="demo_custom_results",
        use_cuda=False  # 强制使用CPU
    )
    
    # 修改系统配置
    system.config.update({
        'cube_size': 512,           # 使用较小的立方体尺寸（更快但精度较低）
        'diff_threshold': 40,       # 提高差异阈值（更严格的检测）
        'min_contour_area': 300,    # 降低最小轮廓面积（检测更小的变化）
        'max_contour_area': 30000,  # 降低最大轮廓面积
    })
    
    print(f"🔧 自定义配置: {system.config}")
    
    # 可以在这里使用自定义配置的系统进行检测
    # results = system.process_panorama_pair(panorama1_path, panorama2_path)


def demo_batch_processing():
    """批量处理演示"""
    print("\n" + "=" * 60)
    print("📦 批量处理演示")
    print("=" * 60)
    
    # 假设有多对图像需要处理
    image_pairs = [
        ("test/pair1_img1.jpg", "test/pair1_img2.jpg"),
        ("test/pair2_img1.jpg", "test/pair2_img2.jpg"),
        # 可以添加更多图像对
    ]
    
    # 创建批量处理系统
    system = PanoramaChangeDetectionSystem(
        output_dir="demo_batch_results",
        use_cuda=True
    )
    
    batch_results = []
    
    for i, (img1_path, img2_path) in enumerate(image_pairs, 1):
        print(f"\n🔄 处理第 {i} 对图像...")
        
        # 检查文件是否存在
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"⚠️ 跳过不存在的图像对: {img1_path}, {img2_path}")
            continue
        
        try:
            # 为每对图像创建独立的输出目录
            system.output_dir = f"demo_batch_results/pair_{i}"
            os.makedirs(system.output_dir, exist_ok=True)
            
            results = system.process_panorama_pair(img1_path, img2_path)
            
            if results and results['processing_successful']:
                batch_results.append({
                    'pair_id': i,
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'detection_count': results['total_detection_count'],
                    'output_dir': system.output_dir
                })
                print(f"✅ 第 {i} 对处理完成")
            else:
                print(f"❌ 第 {i} 对处理失败")
                
        except Exception as e:
            print(f"❌ 第 {i} 对处理出错: {str(e)}")
    
    # 输出批量处理摘要
    if batch_results:
        print(f"\n📊 批量处理摘要:")
        print(f"   成功处理: {len(batch_results)} 对图像")
        total_detections = sum(r['detection_count'] for r in batch_results)
        print(f"   总检测数: {total_detections}")
        
        for result in batch_results:
            print(f"   • 第{result['pair_id']}对: {result['detection_count']} 个检测")


def demo_result_analysis():
    """结果分析演示"""
    print("\n" + "=" * 60)
    print("📈 结果分析演示")
    print("=" * 60)
    
    # 这个函数展示如何分析检测结果
    def analyze_detection_results(results):
        """分析检测结果的详细信息"""
        if not results or not results['processing_successful']:
            print("❌ 无有效结果可分析")
            return
        
        print("🔍 详细结果分析:")
        
        # 分析每个面的检测情况
        print("\n📋 各立方体面检测详情:")
        for face_result in results['face_results']:
            face_name = face_result['face_name']
            bbox_count = len(face_result['bboxes'])
            match_info = face_result['match_info']
            
            print(f"   {face_name} 面:")
            print(f"     检测区域: {bbox_count} 个")
            print(f"     特征匹配: {match_info['matches']} 个")
            print(f"     内点率: {match_info['inlier_ratio']:.2%}")
            
            if bbox_count > 0:
                # 分析bbox的置信度分布
                confidences = [bbox['confidence'] for bbox in face_result['bboxes']]
                avg_confidence = sum(confidences) / len(confidences)
                print(f"     平均置信度: {avg_confidence:.3f}")
                
                # 分析面积分布
                areas = [bbox['area'] for bbox in face_result['bboxes']]
                avg_area = sum(areas) / len(areas)
                print(f"     平均区域面积: {avg_area:.0f} px²")
        
        # 分析全景图映射结果
        print(f"\n🗺️ 全景图映射分析:")
        panorama_bboxes = results['panorama_bboxes']
        print(f"   映射成功的检测框: {len(panorama_bboxes)} 个")
        
        if panorama_bboxes:
            # 按来源面分组统计
            face_groups = {}
            for bbox in panorama_bboxes:
                face_name = bbox['face_name']
                if face_name not in face_groups:
                    face_groups[face_name] = 0
                face_groups[face_name] += 1
            
            print("   各面映射分布:")
            for face_name, count in face_groups.items():
                print(f"     {face_name}: {count} 个")
    
    # 这里可以加载之前的结果进行分析
    print("💡 提示: 运行基础演示后，可以使用 analyze_detection_results() 函数分析结果")


def main():
    """主演示函数"""
    print("🎯 全景图像变化检测系统演示")
    print("请确保已安装所需依赖: opencv-python, numpy, matplotlib, tqdm")
    print("如需GPU加速，请安装: cupy")
    
    # 创建测试目录
    os.makedirs("test", exist_ok=True)
    
    # 运行演示
    success = demo_basic_usage()
    
    if success:
        print("\n🎊 基础演示运行成功！")
        
        # 可以继续运行其他演示
        demo_custom_config()
        demo_batch_processing()
        demo_result_analysis()
    else:
        print("\n📝 演示提示:")
        print("1. 请确保在 test/ 目录下放置测试用的全景图像")
        print("2. 支持的图像格式: .jpg, .jpeg, .png")
        print("3. 建议图像尺寸: 2048x1024 或更大")
        print("4. 确保两期图像拍摄角度基本一致")


if __name__ == "__main__":
    main() 