#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最佳角点检测算法配置
基于建筑物检测应用的优化参数
"""

import cv2


class BestAnglePointConfigs:
    """最佳角点检测算法配置管理器"""
    
    @staticmethod
    def get_recommended_configs():
        """
        获取推荐的算法配置
        
        Returns:
            dict: 包含不同场景的最佳配置
        """
        return {
            # 建筑物检测推荐配置
            'building_detection': {
                'primary': {
                    'algorithm': 'SIFT',
                    'params': {
                        'nfeatures': 1500,           # 特征点数量
                        'contrastThreshold': 0.03,   # 对比度阈值（更敏感）
                        'edgeThreshold': 10,         # 边缘阈值
                        'sigma': 1.6                 # 高斯核标准差
                    },
                    'matching': {
                        'ratio_threshold': 0.65,     # Lowe's ratio test
                        'ransac_threshold': 3.0,     # RANSAC重投影误差
                        'min_matches': 8             # 最小匹配点数
                    },
                    'description': '建筑物检测的最佳配置，高精度低噪声'
                },
                'secondary': {
                    'algorithm': 'AKAZE',
                    'params': {
                        'threshold': 0.0008,         # 检测阈值
                        'nOctaves': 4,              # 组数
                        'nOctaveLayers': 4,         # 每组层数
                        'diffusivity': cv2.KAZE_DIFF_PM_G2
                    },
                    'matching': {
                        'ratio_threshold': 0.7,
                        'ransac_threshold': 4.0,
                        'min_matches': 6
                    },
                    'description': '快速稳定的备选方案'
                }
            },
            
            # 不同质量级别的配置
            'quality_levels': {
                'ultra_high': {
                    'algorithm': 'SIFT',
                    'params': {
                        'nfeatures': 3000,
                        'contrastThreshold': 0.02,
                        'edgeThreshold': 8,
                        'sigma': 1.6
                    },
                    'matching': {
                        'ratio_threshold': 0.6,
                        'ransac_threshold': 2.0,
                        'min_matches': 12
                    },
                    'processing_time': 'slow',
                    'accuracy': 'highest'
                },
                'high': {
                    'algorithm': 'SIFT',
                    'params': {
                        'nfeatures': 1500,
                        'contrastThreshold': 0.03,
                        'edgeThreshold': 10,
                        'sigma': 1.6
                    },
                    'matching': {
                        'ratio_threshold': 0.65,
                        'ransac_threshold': 3.0,
                        'min_matches': 8
                    },
                    'processing_time': 'moderate',
                    'accuracy': 'high'
                },
                'balanced': {
                    'algorithm': 'AKAZE',
                    'params': {
                        'threshold': 0.001,
                        'nOctaves': 4,
                        'nOctaveLayers': 4,
                        'diffusivity': cv2.KAZE_DIFF_PM_G2
                    },
                    'matching': {
                        'ratio_threshold': 0.7,
                        'ransac_threshold': 4.0,
                        'min_matches': 6
                    },
                    'processing_time': 'fast',
                    'accuracy': 'good'
                },
                'fast': {
                    'algorithm': 'ORB',
                    'params': {
                        'nfeatures': 1000,
                        'scaleFactor': 1.2,
                        'nlevels': 8,
                        'edgeThreshold': 31,
                        'scoreType': cv2.ORB_HARRIS_SCORE
                    },
                    'matching': {
                        'ratio_threshold': 0.75,
                        'ransac_threshold': 5.0,
                        'min_matches': 6
                    },
                    'processing_time': 'very_fast',
                    'accuracy': 'moderate'
                }
            },
            
            # 不同场景的特化配置
            'scenarios': {
                'outdoor_building': {
                    'algorithm': 'SIFT',
                    'params': {
                        'nfeatures': 2000,
                        'contrastThreshold': 0.025,  # 户外光照变化大
                        'edgeThreshold': 12,
                        'sigma': 1.6
                    },
                    'matching': {
                        'ratio_threshold': 0.65,
                        'ransac_threshold': 4.0,
                        'min_matches': 10
                    }
                },
                'indoor_structure': {
                    'algorithm': 'AKAZE',
                    'params': {
                        'threshold': 0.0005,         # 室内细节更多
                        'nOctaves': 5,
                        'nOctaveLayers': 4,
                        'diffusivity': cv2.KAZE_DIFF_PM_G2
                    },
                    'matching': {
                        'ratio_threshold': 0.7,
                        'ransac_threshold': 3.0,
                        'min_matches': 8
                    }
                },
                'aerial_view': {
                    'algorithm': 'SIFT',
                    'params': {
                        'nfeatures': 2500,
                        'contrastThreshold': 0.035,  # 航拍图对比度通常较低
                        'edgeThreshold': 15,
                        'sigma': 1.8
                    },
                    'matching': {
                        'ratio_threshold': 0.7,
                        'ransac_threshold': 5.0,
                        'min_matches': 12
                    }
                }
            }
        }
    
    @staticmethod
    def get_flann_params(descriptor_type='float'):
        """
        获取FLANN匹配器的最佳参数
        
        Args:
            descriptor_type (str): 描述符类型 ('float' 或 'binary')
            
        Returns:
            tuple: (index_params, search_params)
        """
        if descriptor_type == 'float':
            # 用于SIFT, SURF等浮点描述符
            index_params = dict(
                algorithm=1,    # FLANN_INDEX_KDTREE
                trees=8         # 增加树的数量提高精度
            )
            search_params = dict(
                checks=100      # 增加检查次数提高精度
            )
        else:
            # 用于ORB, BRISK等二进制描述符
            index_params = dict(
                algorithm=6,        # FLANN_INDEX_LSH
                table_number=8,     # 哈希表数量
                key_size=16,        # 键长度
                multi_probe_level=2 # 多探针级别
            )
            search_params = dict(
                checks=80
            )
        
        return index_params, search_params
    
    @staticmethod
    def create_detector(config_name='building_detection', quality='primary'):
        """
        根据配置创建特征检测器
        
        Args:
            config_name (str): 配置类别
            quality (str): 质量级别
            
        Returns:
            tuple: (detector, algorithm_name, params)
        """
        configs = BestAnglePointConfigs.get_recommended_configs()
        
        if config_name == 'building_detection':
            config = configs[config_name][quality]
        elif config_name == 'quality_levels':
            config = configs[config_name][quality]
        elif config_name == 'scenarios':
            config = configs[config_name][quality]
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        algorithm = config['algorithm']
        params = config['params']
        
        if algorithm == 'SIFT':
            detector = cv2.SIFT_create(
                nfeatures=params['nfeatures'],
                contrastThreshold=params['contrastThreshold'],
                edgeThreshold=params['edgeThreshold'],
                sigma=params['sigma']
            )
        elif algorithm == 'ORB':
            detector = cv2.ORB_create(
                nfeatures=params['nfeatures'],
                scaleFactor=params['scaleFactor'],
                nlevels=params['nlevels'],
                edgeThreshold=params['edgeThreshold'],
                scoreType=params.get('scoreType', cv2.ORB_HARRIS_SCORE)
            )
        elif algorithm == 'AKAZE':
            detector = cv2.AKAZE_create(
                threshold=params['threshold'],
                nOctaves=params['nOctaves'],
                nOctaveLayers=params['nOctaveLayers'],
                diffusivity=params['diffusivity']
            )
        elif algorithm == 'BRISK':
            detector = cv2.BRISK_create(
                thresh=params['thresh'],
                octaves=params['octaves'],
                patternScale=params['patternScale']
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return detector, algorithm, config
    
    @staticmethod
    def print_recommendations():
        """打印所有推荐配置"""
        configs = BestAnglePointConfigs.get_recommended_configs()
        
        print("=" * 60)
        print("最佳角点检测算法配置推荐")
        print("=" * 60)
        
        print("\n建筑物检测专用配置:")
        print("-" * 40)
        for quality, config in configs['building_detection'].items():
            print(f"\n{quality.upper()}配置:")
            print(f"  算法: {config['algorithm']}")
            print(f"  参数: {config['params']}")
            print(f"  匹配: {config['matching']}")
            print(f"  说明: {config['description']}")
        
        print("\n\n不同质量级别配置:")
        print("-" * 40)
        for level, config in configs['quality_levels'].items():
            print(f"\n{level.upper()}质量:")
            print(f"  算法: {config['algorithm']}")
            print(f"  参数: {config['params']}")
            print(f"  处理时间: {config['processing_time']}")
            print(f"  精度: {config['accuracy']}")
        
        print("\n\n不同场景特化配置:")
        print("-" * 40)
        for scenario, config in configs['scenarios'].items():
            print(f"\n{scenario.replace('_', ' ').title()}:")
            print(f"  算法: {config['algorithm']}")
            print(f"  参数: {config['params']}")
            print(f"  匹配: {config['matching']}")


# 使用示例
if __name__ == "__main__":
    # 打印所有推荐配置
    BestAnglePointConfigs.print_recommendations()
    
    # 创建建筑物检测的最佳检测器
    detector, algorithm, config = BestAnglePointConfigs.create_detector(
        'building_detection', 'primary'
    )
    
    print(f"\n已创建 {algorithm} 检测器，配置: {config['description']}") 