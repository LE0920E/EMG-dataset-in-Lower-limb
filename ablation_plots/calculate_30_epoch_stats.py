import pandas as pd
import numpy as np
import os

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

# 存储统计结果
results = []

print("Calculating statistics after 30 epochs...")

for i, file_pattern in enumerate(file_patterns):
    file_path = os.path.join(os.path.dirname(__file__), file_pattern)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # 获取30轮之后的数据（从第31个epoch开始）
        if len(df) > 30:
            after_30_epochs = df.iloc[30:]
            
            # 计算训练损失的平均值和方差
            train_loss_mean = after_30_epochs['train_loss'].mean()
            train_loss_var = after_30_epochs['train_loss'].var()
            
            # 计算测试损失的平均值和方差
            test_loss_mean = after_30_epochs['test_loss'].mean()
            test_loss_var = after_30_epochs['test_loss'].var()
            
            results.append({
                'Parameter Combination': labels[i],
                'Train Loss Mean (after 30 epochs)': round(train_loss_mean, 4),
                'Train Loss Variance (after 30 epochs)': round(train_loss_var, 4),
                'Validation Loss Mean (after 30 epochs)': round(test_loss_mean, 4),
                'Validation Loss Variance (after 30 epochs)': round(test_loss_var, 4)
            })
            
            print(f"Processed {file_pattern}: {len(after_30_epochs)} epochs after 30")
        else:
            print(f"Warning: {file_pattern} has only {len(df)} epochs, not enough data after 30 epochs")
    else:
        print(f"Warning: File {file_pattern} not found")

# 创建DataFrame（不进行排序）
if results:
    df_results = pd.DataFrame(results)
    
    # 保存结果到CSV文件（保持原始顺序）
    output_file = os.path.join(os.path.dirname(__file__), 'after_30_epoch_statistics.csv')
    df_results.to_csv(output_file, index=False)
    
    print(f"\nFile saved as: {output_file}")
    
    # 显示统计信息
    print("\nStatistics Summary (After 30 Epochs):")
    print("=" * 110)
    print(f"{'Parameter Combination':<15} {'Train Loss Mean':<18} {'Train Loss Var':<18} {'Val Loss Mean':<18} {'Val Loss Var':<18}")
    print("-" * 110)
    
    for _, row in df_results.iterrows():
        print(f"{row['Parameter Combination']:<15} {row['Train Loss Mean (after 30 epochs)']:<18} {row['Train Loss Variance (after 30 epochs)']:<18} {row['Validation Loss Mean (after 30 epochs)']:<18} {row['Validation Loss Variance (after 30 epochs)']:<18}")
    
    print("=" * 110)
    
    print("\nCalculation completed successfully!")
else:
    print("No data processed.")