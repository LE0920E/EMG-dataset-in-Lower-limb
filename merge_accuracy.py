import pandas as pd
import os

def merge_all_accuracy():
    """合并所有模型的准确率到models_v2文件夹"""
    
    # 读取原始准确率文件
    original_file = "./models/Accuracy.csv"
    if os.path.exists(original_file):
        df_original = pd.read_csv(original_file)
        print(f"已读取原始准确率文件: {original_file}")
        print(f"包含模型版本: {df_original['Version'].tolist()}")
    else:
        print("原始准确率文件不存在")
        df_original = pd.DataFrame()
    
    # 检查models_v2文件夹中的准确率文件
    models_v2_dir = "./models_v2"
    all_accuracy_files = []
    
    if os.path.exists(models_v2_dir):
        for file in os.listdir(models_v2_dir):
            if file.startswith("Accuracy_") and file.endswith(".csv"):
                file_path = os.path.join(models_v2_dir, file)
                try:
                    df_temp = pd.read_csv(file_path)
                    if not df_temp.empty:
                        all_accuracy_files.append(df_temp)
                        print(f"已读取: {file}")
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
    
    # 合并所有数据
    if df_original.empty and not all_accuracy_files:
        print("没有找到任何准确率数据")
        return
    
    # 创建合并的DataFrame
    if not df_original.empty:
        merged_df = df_original.copy()
    else:
        merged_df = pd.DataFrame()
    
    for df in all_accuracy_files:
        if merged_df.empty:
            merged_df = df.copy()
        else:
            # 检查版本是否已存在
            for _, row in df.iterrows():
                version = row['Version']
                if version not in merged_df['Version'].values:
                    merged_df = pd.concat([merged_df, pd.DataFrame([row])], ignore_index=True)
    
    # 保存合并后的文件
    output_file = "./models_v2/All_Models_Accuracy.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\n合并后的准确率文件已保存到: {output_file}")
    print(f"包含 {len(merged_df)} 个模型版本:")
    print(merged_df[['Version', 'Action accuracy', 'Status accuracy', 'average accuracy']].to_string(index=False))
    
    return merged_df

def create_model_comparison_chart():
    """创建模型比较图表"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 读取合并的准确率文件
    accuracy_file = "./models_v2/All_Models_Accuracy.csv"
    if not os.path.exists(accuracy_file):
        print("合并的准确率文件不存在")
        return
    
    df = pd.read_csv(accuracy_file)
    
    # 创建比较图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 动作准确率比较
    axes[0, 0].bar(df['Version'], df['Action accuracy'] * 100, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('各模型动作识别准确率比较')
    axes[0, 0].set_ylabel('准确率 (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 状态准确率比较
    axes[0, 1].bar(df['Version'], df['Status accuracy'] * 100, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('各模型状态识别准确率比较')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 平均准确率比较
    axes[1, 0].bar(df['Version'], df['average accuracy'] * 100, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('各模型平均准确率比较')
    axes[1, 0].set_ylabel('准确率 (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 综合性能雷达图
    # 简化版本：使用条形图代替雷达图
    versions = df['Version'].tolist()
    action_acc = df['Action accuracy'].tolist()
    status_acc = df['Status accuracy'].tolist()
    
    x = range(len(versions))
    width = 0.35
    
    axes[1, 1].bar([i - width/2 for i in x], action_acc, width, label='动作准确率', alpha=0.7)
    axes[1, 1].bar([i + width/2 for i in x], status_acc, width, label='状态准确率', alpha=0.7)
    axes[1, 1].set_title('动作 vs 状态准确率对比')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(versions, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('./models_v2/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("模型比较图表已保存到: ./models_v2/model_comparison.png")

if __name__ == "__main__":
    # 合并准确率
    merged_df = merge_all_accuracy()
    
    # 创建比较图表
    create_model_comparison_chart()
    
    print("\n所有模型准确率数据已成功合并和可视化！")