import os
import pandas as pd

def get_accuracy_by_version(csv_file, version):
    """
    从CSV文件中读取指定版本的三项准确率，并返回三个变量。

    :param csv_file: str, CSV文件路径
    :param version: str, 要查询的版本号，如 'v1.1.1'
    :return: tuple (action_acc, status_acc, average_acc)，如果未找到则返回 (None, None, None)
    """
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return None, None, None

    try:
        # 读取CSV
        df = pd.read_csv(csv_file)
        
        # 去除列名前后空格
        df.columns = df.columns.str.strip()
        
        # 查找对应版本
        result = df[df['Version'] == version]
        
        if result.empty:
            print(f" 未找到版本 '{version}'")
            return None, None, None
        
        # 提取第一行数据（假设版本唯一）
        action_acc = float(result['Action accuracy'].iloc[0])
        status_acc = float(result['Status accuracy'].iloc[0])
        average_acc = float(result['average accuracy'].iloc[0])
        
        return action_acc, status_acc, average_acc

    except Exception as e:
        print(f"❌ 读取或解析文件时出错: {e}")
        return None, None, None
    

def update_accuracy_by_version(csv_file, version, action_acc=None, status_acc=None, average_acc=None):
    """
    更新CSV文件中指定版本的准确率数据。如果版本不存在，可选择添加新行。

    :param csv_file: str, CSV文件路径
    :param version: str, 要更新的版本号，如 'v1.1.1'
    :param action_acc: float, 动作准确率（可选）
    :param status_acc: float, 状态准确率（可选）
    :param average_acc: float, 平均准确率（可选）
    :return: bool, 是否更新成功
    """
    # 检查参数
    if all(v is None for v in [action_acc, status_acc, average_acc]):
        print("❌ 至少提供一个要更新的准确率值")
        return False

    # 检查文件是否存在，若不存在则创建带表头的空DataFrame
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
    else:
        print(f"🟡 文件 {csv_file} 不存在，将创建新文件")
        df = pd.DataFrame(columns=['Version', 'Action accuracy', 'Status accuracy', 'average accuracy'])

    # 检查是否已有该版本
    if version in df['Version'].values:
        print(f"✅ 找到版本 '{version}'，正在更新...")
        row_idx = df[df['Version'] == version].index[0]

        if action_acc is not None:
            df.loc[row_idx, 'Action accuracy'] = action_acc
        if status_acc is not None:
            df.loc[row_idx, 'Status accuracy'] = status_acc
        if average_acc is not None:
            df.loc[row_idx, 'average accuracy'] = average_acc
    else:
        # 版本不存在，新增一行
        print(f"🆕 版本 '{version}' 不存在，正在新增...")
        new_row = {
            'Version': version,
            'Action accuracy': action_acc if action_acc is not None else 0.0,
            'Status accuracy': status_acc if status_acc is not None else 0.0,
            'average accuracy': average_acc if average_acc is not None else 0.0
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 保存回CSV
    try:
        df.to_csv(csv_file, index=False)
        print(f" 数据已成功保存到 '{csv_file}'")
        return True
    except Exception as e:
        print(f" 保存文件失败: {e}")
        return False

MODEL_VERSION="v1.1.1"
best_acc_action,best_acc_status,best_acc=get_accuracy_by_version(r"./models/Accuracy.csv",MODEL_VERSION)
print(best_acc,best_acc_action,best_acc_status)

best_acc_action,best_acc_status,best_acc=0,0,0

update_accuracy_by_version(r"./models/Accuracy.csv","v2.0",best_acc_action,best_acc_status,best_acc)