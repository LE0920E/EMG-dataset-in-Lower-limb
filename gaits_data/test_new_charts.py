#!/usr/bin/env python3
"""
测试新图表功能脚本
验证所有6种新图表功能是否正常工作，图片是否自动保存
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_visualization import EMGDataVisualizer

def test_all_new_charts():
    """测试所有新图表功能"""
    print("开始测试所有新图表功能...")
    
    # 初始化可视化器
    visualizer = EMGDataVisualizer(".")
    
    # 加载数据
    print("1. 加载数据文件...")
    files_df = visualizer.load_all_files()
    print(f"成功加载 {len(files_df)} 个文件")
    
    # 测试图1：单样本多通道时序信号图
    print("\n2. 测试图1：单样本多通道时序信号图...")
    try:
        visualizer.plot_single_sample_timeseries()
        print("✓ 图1测试成功")
    except Exception as e:
        print(f"✗ 图1测试失败: {e}")
    
    # 测试图2：同一受试者多动作对比图
    print("\n3. 测试图2：同一受试者多动作对比图...")
    try:
        visualizer.plot_same_subject_multi_action()
        print("✓ 图2测试成功")
    except Exception as e:
        print(f"✗ 图2测试失败: {e}")
    
    # 测试图3：正常组与异常组步态平均激活模式图
    print("\n4. 测试图3：正常组与异常组步态平均激活模式图...")
    try:
        visualizer.plot_group_average_gait_pattern()
        print("✓ 图3测试成功")
    except Exception as e:
        print(f"✗ 图3测试失败: {e}")
    
    # 测试图4：股四头肌RMS值箱线图对比
    print("\n5. 测试图4：股四头肌RMS值箱线图对比...")
    try:
        visualizer.plot_rectus_femoris_rms_boxplot()
        print("✓ 图4测试成功")
    except Exception as e:
        print(f"✗ 图4测试失败: {e}")
    
    # 测试图5：多受试者股直肌激活热力图
    print("\n6. 测试图5：多受试者股直肌激活热力图...")
    try:
        visualizer.plot_multi_subject_rectus_femoris_heatmap()
        print("✓ 图5测试成功")
    except Exception as e:
        print(f"✗ 图5测试失败: {e}")
    
    # 测试图6：EMG特征空间PCA分布图
    print("\n7. 测试图6：EMG特征空间PCA分布图...")
    try:
        visualizer.plot_emg_feature_pca()
        print("✓ 图6测试成功")
    except Exception as e:
        print(f"✗ 图6测试失败: {e}")
    
    # 检查plots目录中的文件
    print("\n8. 检查plots目录中的图片文件...")
    plots_dir = visualizer.output_dir
    if plots_dir.exists():
        png_files = list(plots_dir.glob("*.png"))
        print(f"plots目录中找到 {len(png_files)} 个PNG文件")
        
        # 按图表类型分类
        chart_types = {
            "单样本时序图": [f for f in png_files if "single_sample_timeseries" in f.name],
            "多动作对比图": [f for f in png_files if "same_subject_multi_action" in f.name],
            "组平均模式图": [f for f in png_files if "group_average_gait_pattern" in f.name],
            "RMS箱线图": [f for f in png_files if "rectus_femoris_rms_boxplot" in f.name],
            "激活热力图": [f for f in png_files if "multi_subject_rectus_femoris_heatmap" in f.name],
            "PCA分布图": [f for f in png_files if "emg_feature_pca_distribution" in f.name]
        }
        
        for chart_name, files in chart_types.items():
            print(f"  {chart_name}: {len(files)} 个文件")
    else:
        print("✗ plots目录不存在")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_all_new_charts()