import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_normal_gait_plots():
    """创建正常组所有人走路时各个肌肉和关节角度的折线图"""
    
    # 数据目录
    data_dir = Path("./normal")
    output_dir = Path("./plot2")
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    # 肌肉和关节名称映射（英文）
    muscle_names = {
        'Recto Femoral': 'Rectus Femoris',
        'Biceps Femoral': 'Biceps Femoris', 
        'Vasto Medial': 'Vastus Medialis',
        'EMG Semitendinoso': 'Semitendinosus',
        'Flexo-Extension': 'Knee Flexion-Extension Angle'
    }
    
    # 肌肉颜色
    muscle_colors = {
        'Rectus Femoris': '#1f77b4',
        'Biceps Femoris': '#ff7f0e',
        'Vastus Medialis': '#2ca02c',
        'Semitendinosus': '#d62728',
        'Knee Flexion-Extension Angle': '#9467bd'
    }
    
    # 获取所有正常组走路数据文件
    gait_files = []
    for file_path in data_dir.glob("*gait*.csv"):
        if file_path.is_file():
            gait_files.append(file_path)
    
    print(f"Found {len(gait_files)} gait files")
    
    # 为每个肌肉/关节创建单独的图表
    for muscle_key, muscle_name in muscle_names.items():
        print(f"Creating plot for {muscle_name}...")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 设置10秒的时间轴（10000个数据点，采样率1000Hz）
        time_axis = np.arange(0, 10000) / 1000.0  # 转换为秒
        
        # 为每个文件绘制数据
        for file_path in gait_files:
            try:
                # 读取数据
                df = pd.read_csv(file_path)
                
                # 获取受试者ID
                subject_id = file_path.stem.split('N')[0] + 'N'
                
                # 检查数据列
                if muscle_key not in df.columns:
                    # 尝试查找相似的列名
                    matching_cols = [col for col in df.columns if muscle_key.lower() in col.lower()]
                    if matching_cols:
                        actual_col = matching_cols[0]
                    else:
                        print(f"Column {muscle_key} not found in {file_path}. Available columns: {df.columns.tolist()}")
                        continue
                else:
                    actual_col = muscle_key
                
                # 获取数据
                data = df[actual_col].values
                
                # 创建完整的时间序列数据（10秒，不足的留白）
                full_data = np.full(10000, np.nan)  # 初始化为NaN（留白）
                
                # 填充实际数据
                if len(data) > 0:
                    actual_length = min(len(data), 10000)
                    full_data[:actual_length] = data[:actual_length]
                
                # 绘制折线图
                ax.plot(time_axis, full_data, 
                        label=f'Subject {subject_id}', 
                        alpha=0.7, 
                        linewidth=1.5)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # 设置图表属性
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        
        if muscle_name == 'Knee Flexion-Extension Angle':
            ax.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('EMG Amplitude', fontsize=12, fontweight='bold')
        
        ax.set_title(f'{muscle_name} - Normal Group Gait Pattern', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        ax.set_xlim(0, 10)
        
        # 添加图例（如果数据点太多，可以限制图例数量）
        if len(gait_files) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # 对于大量数据，只显示部分图例或使用其他方式
            ax.text(0.02, 0.98, f'{len(gait_files)} subjects', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        filename = muscle_name.lower().replace(' ', '_').replace('-', '_') + '_gait.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    # 创建汇总图表（所有肌肉在一个图中）
    print("Creating summary plot...")
    
    # 选择一个代表性的文件来显示所有肌肉
    if gait_files:
        representative_file = gait_files[0]
        
        try:
            df = pd.read_csv(representative_file)
            subject_id = representative_file.stem.split('N')[0] + 'N'
            
            fig, axes = plt.subplots(5, 1, figsize=(15, 20))
            
            for i, (muscle_key, muscle_name) in enumerate(muscle_names.items()):
                ax = axes[i]
                
                # 检查数据列
                if muscle_key in df.columns:
                    actual_col = muscle_key
                else:
                    # 尝试查找相似的列名
                    matching_cols = [col for col in df.columns if muscle_key.lower() in col.lower()]
                    if matching_cols:
                        actual_col = matching_cols[0]
                    else:
                        continue
                
                # 获取数据
                data = df[actual_col].values
                
                # 创建完整的时间序列数据
                full_data = np.full(10000, np.nan)
                if len(data) > 0:
                    actual_length = min(len(data), 10000)
                    full_data[:actual_length] = data[:actual_length]
                
                # 绘制折线图
                ax.plot(time_axis, full_data, 
                       color=muscle_colors[muscle_name],
                       linewidth=2,
                       label=muscle_name)
                
                # 设置子图属性
                ax.set_title(muscle_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (seconds)' if i == 4 else '', fontsize=10)
                
                if muscle_name == 'Knee Flexion-Extension Angle':
                    ax.set_ylabel('Angle (degrees)', fontsize=10)
                else:
                    ax.set_ylabel('EMG Amplitude', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 10)
                ax.legend()
            
            plt.suptitle(f'All Muscle Activities and Knee Angle - Subject {subject_id} (Normal Group)', 
                        fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            
            # 保存汇总图表
            summary_path = output_dir / 'all_muscles_summary_gait.png'
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved summary: {summary_path}")
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
    
    print("\nAll plots have been generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    create_normal_gait_plots()