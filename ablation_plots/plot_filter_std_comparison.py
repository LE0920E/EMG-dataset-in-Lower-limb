import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 定义文件列表和对应的标签
file_patterns = [
    'all.csv',
    'no_biceps_femoral.csv',
    'no_emg_semitendinoso.csv',
    'no_flexo_extension.csv',
    'no_recto_femoral.csv',
    'no_vasto_medial.csv'
]

labels = [
    'All Features',
    'No Biceps Femoral',
    'No EMG Semitendinoso',
    'No Flexo Extension',
    'No Recto Femoral',
    'No Vasto Medial'
]

# 颜色列表，确保有足够的颜色区分9条线
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# 存储所有数据
all_data = []
max_epochs = 0

print("Loading data from CSV files...")
for i, file_pattern in enumerate(file_patterns):
    file_path = os.path.join(os.path.dirname(__file__), file_pattern)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # 获取epochs和loss数据
        epochs = df['epoch'].values
        train_loss = df['train_loss'].values
        test_loss = df['test_loss'].values
        
        # 更新最大epoch数
        max_epochs = max(max_epochs, len(epochs))
        
        all_data.append({
            'label': labels[i],
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss
        })
        
        print(f"Loaded {file_pattern}: {len(epochs)} epochs")
    else:
        print(f"Warning: File {file_pattern} not found")

print(f"\nMaximum epochs among all files: {max_epochs}")

# 创建训练损失对比图
plt.figure(figsize=(12, 8))

for i, data in enumerate(all_data):
    alpha = 1.0  # 不透明
    linewidth = 2

    # if i == 4:
    #     alpha = 1.0  # 不透明
    #     linewidth = 2
    # else:
    #     alpha = 0.0  # 完全透明
    #     linewidth = 0
    
    plt.plot(data['epochs'], data['train_loss'], 
             label=data['label'], color=colors[i], linewidth=linewidth, alpha=alpha)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.title('Training Loss Comparison - 9 Parameter Combinations', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存训练损失对比图
train_loss_output = os.path.join(os.path.dirname(__file__), 'train_loss_comparison.png')
plt.savefig(train_loss_output, dpi=300, bbox_inches='tight')
print(f"Training loss comparison plot saved as: {train_loss_output}")

# 创建测试损失对比图
plt.figure(figsize=(12, 8))

for i, data in enumerate(all_data):
    alpha = 1.0  # 不透明
    linewidth = 2
    # 除了第一条线（A1 B1）和最后一条线（A3 B3），其他线设置为透明
    # if i == 4:
    #     alpha = 1.0  # 不透明
    #     linewidth = 2
    # else:
    #     alpha = 0.0  # 完全透明
    #     linewidth = 0
    
    plt.plot(data['epochs'], data['test_loss'], 
             label=data['label'], color=colors[i], linewidth=linewidth, alpha=alpha)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.title('Validation Loss Comparison - 9 Parameter Combinations', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存测试损失对比图
test_loss_output = os.path.join(os.path.dirname(__file__), 'test_loss_comparison.png')
plt.savefig(test_loss_output, dpi=300, bbox_inches='tight')
print(f"Validation loss comparison plot saved as: {test_loss_output}")

# 显示统计信息
print("\nStatistical Summary:")
print("=" * 80)
print(f"{'Parameter':<10} {'Epochs':<8} {'Final Train Loss':<15} {'Final Val Loss':<15}")
print("-" * 80)

for data in all_data:
    final_train_loss = data['train_loss'][-1] if len(data['train_loss']) > 0 else 0
    final_test_loss = data['test_loss'][-1] if len(data['test_loss']) > 0 else 0
    print(f"{data['label']:<10} {len(data['epochs']):<8} {final_train_loss:<15.4f} {final_test_loss:<15.4f}")

print("=" * 80)
print("\nPlot generation completed successfully!")

# 显示图片（可选）
plt.show()