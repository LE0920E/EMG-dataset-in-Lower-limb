import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from tqdm import tqdm
from factory_model import device

# 标签定义
ACTION_LABELS = {'gait': 0, 'sitting': 1, 'standing': 2}
STATUS_LABELS = {'N': 0, 'A': 1}  # N: Normal, A: Abnormal
BATCH_SIZE = 32  # 减小批次大小以适应更大的窗口
SEQ_LENGTH = 500  # 序列长度

# 滤波选项
FILTER_OPTIONS = {
    'none': '无滤波',
    'bandpass': '滤波',
    'bandpass_notch': '滤波+工频陷波(60Hz)'
}
FILTER='bandpass'

# 标准化选项
STANDARDIZATION_OPTIONS = {
    'none': '无标准化',
    'zscore': '全训练集z-score标准化',
    'minmax': '归一化到[0,1]范围'
}

STANDARDIZATION='zscore'

class EMGDataset(Dataset):
    def __init__(self, root_dir, seq_length=1000, filter_option='bandpass', standardization_option='zscore', scaler=None):
        """
        增强版EMG数据集类，包含膝盖屈伸角度和滤波处理
        
        Args:
            root_dir: 数据根目录
            seq_length: 序列长度（默认1000个时间点）
            filter_option: 滤波选项 - 'none'（无滤波）、'bandpass'（滤波）、'bandpass_notch'（滤波+工频陷波）
            standardization_option: 标准化选项 - 'none'（无标准化）、'zscore'（全训练集z-score标准化）
            scaler: 预训练的标准化器（可选）
        """
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.filter_option = filter_option
        self.standardization_option = standardization_option
        self.file_paths = []
        self.scaler = scaler  # 使用传入的标准化器
        
        # 滤波参数
        self.lowcut = 20  # 低通滤波截止频率 (Hz)
        self.highcut = 450  # 高通滤波截止频率 (Hz)
        self.notch_freq = 60  # 工频陷波频率 (Hz)
        self.fs = 1000  # 采样频率 (Hz)
        
        # 验证选项
        if filter_option not in FILTER_OPTIONS:
            raise ValueError(f"无效的滤波选项: {filter_option}。可用选项: {list(FILTER_OPTIONS.keys())}")
        if standardization_option not in STANDARDIZATION_OPTIONS:
            raise ValueError(f"无效的标准化选项: {standardization_option}。可用选项: {list(STANDARDIZATION_OPTIONS.keys())}")
        
        print(f"滤波选项: {FILTER_OPTIONS[filter_option]}")
        print(f"标准化选项: {STANDARDIZATION_OPTIONS[standardization_option]}")
        
        self._load_file_paths()

    def _load_file_paths(self):
        """加载所有CSV文件路径"""
        for folder in ['Abnormal', 'Normal']:
            class_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(class_path):
                continue
            for file in os.listdir(class_path):
                if file.endswith('.csv'):
                    self.file_paths.append(os.path.join(class_path, file))
        print(f"Loaded {len(self.file_paths)} files from {self.root_dir}")

    def _apply_filter(self, data):
        """应用滤波处理"""
        if self.filter_option == 'none':
            return data
            
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # 设计巴特沃斯带通滤波器
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 对每个通道应用滤波
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        # 如果需要工频陷波
        if self.filter_option == 'bandpass_notch':
            # 设计工频陷波滤波器（60Hz）
            notch_freq = self.notch_freq / nyquist
            quality = 30  # 品质因数
            b_notch, a_notch = signal.iirnotch(notch_freq, quality)
            
            # 对每个通道应用陷波滤波
            for i in range(filtered_data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b_notch, a_notch, filtered_data[:, i])
            
        return filtered_data

    def fit_scaler(self):
        """拟合标准化器（用于训练集）"""
        all_data = []
        for file_path in tqdm(self.file_paths, desc="Fitting scaler"):
            df = pd.read_csv(file_path)
            
            # 统一列名为小写格式
            df.columns = df.columns.str.strip().str.lower()
            
            # 包含膝盖屈伸角度的所有通道
            wanted_columns = [
                'recto femoral', 'biceps femoral', 'vasto medial', 
                'emg semitendinoso', 'flexo-extension'
            ]
            
            # 检查列名兼容性
            available_columns = []
            for col in wanted_columns:
                if col in df.columns:
                    available_columns.append(col)
                elif 'flexo' in col and 'extension' in col:
                    # 尝试查找屈伸角度列
                    angle_cols = [c for c in df.columns if 'flexo' in c.lower() or 'extension' in c.lower()]
                    if angle_cols:
                        available_columns.append(angle_cols[0])
                    else:
                        available_columns.append(col)
                else:
                    available_columns.append(col)
            
            if len(available_columns) >= 4:  # 至少需要4个肌电通道
                data = df[available_columns].values
                
                # 应用滤波
                if self.filter_option != 'none':
                    data = self._apply_filter(data)
                
                all_data.append(data)

        if all_data:
            all_data = np.vstack(all_data)
            if self.standardization_option == 'zscore':
                self.scaler = StandardScaler()
            elif self.standardization_option == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            self.scaler.fit(all_data)
            print(f"{self.standardization_option} scaler fitted with {all_data.shape[0]} samples")
            return self.scaler
        else:
            print("Warning: No valid data found for scaler fitting")
            return None

    def _parse_labels(self, filename):
        """从文件名解析标签"""
        base_name = os.path.basename(filename).split(' - ')[0]
        
        # 动作标签解析
        action = None
        for act in ['gait', 'sitting', 'standing']:
            if act in base_name.lower():
                action = act
                break
        if action is None:
            # 默认动作
            action = 'gait'
        
        # 状态标签解析
        status = 'A' if 'A' in base_name.upper() else 'N'
        
        action_label = ACTION_LABELS[action]
        status_label = STATUS_LABELS[status]
        
        return action_label, status_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """获取单个样本 - 支持变长数据，不进行零填充"""
        file_path = self.file_paths[idx]
        df = pd.read_csv(file_path)
        
        # 统一列名
        df.columns = df.columns.str.strip().str.lower()
        
        # 包含膝盖屈伸角度的所有通道
        wanted_columns = [
            'recto femoral', 'biceps femoral', 'vasto medial', 
            'emg semitendinoso', 'flexo-extension'
        ]
        
        # 检查列名兼容性
        available_columns = []
        column_mapping = {}
        for col in wanted_columns:
            if col in df.columns:
                available_columns.append(col)
                column_mapping[col] = col
            elif 'flexo' in col and 'extension' in col:
                # 查找屈伸角度列
                angle_cols = [c for c in df.columns if 'flexo' in c.lower() or 'extension' in c.lower()]
                if angle_cols:
                    available_columns.append(angle_cols[0])
                    column_mapping[col] = angle_cols[0]
                else:
                    available_columns.append(col)
                    column_mapping[col] = col
            else:
                available_columns.append(col)
                column_mapping[col] = col
        
        if len(available_columns) < 4:
            raise ValueError(f"Missing expected columns in {file_path}. Available: {df.columns.tolist()}")
        
        data = df[available_columns].values
        
        # 应用滤波
        if self.filter_option != 'none':
            data = self._apply_filter(data)
        
        # 应用标准化
        if self.standardization_option in ['zscore', 'minmax'] and self.scaler is not None:
            data = self.scaler.transform(data)
        
        # 解析标签
        action_label, status_label = self._parse_labels(file_path)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # 窗口化处理 - 支持变长数据
        if data.shape[0] >= self.seq_length:
            # 随机选择窗口
            start_idx = np.random.randint(0, data.shape[0] - self.seq_length + 1)
            window = data_tensor[start_idx:start_idx + self.seq_length]
        else:
            # 不进行零填充，直接使用所有可用数据
            window = data_tensor
        
        return window, action_label, status_label


from torch.utils.data import random_split

def create_datasets_with_separate_processing(root_dir='./gaits_data', seq_length=SEQ_LENGTH, 
                                           filter_option=FILTER, standardization_option=STANDARDIZATION,
                                           train_ratio=0.8, random_seed=42):
    """
    创建训练集和测试集，分别进行滤波和标准化处理
    
    Args:
        root_dir: 数据根目录
        seq_length: 序列长度
        filter_option: 滤波选项
        standardization_option: 标准化选项
        train_ratio: 训练集比例
        random_seed: 随机种子
        
    Returns:
        train_dataset: 训练集
        test_dataset: 测试集
    """
    print("=== 数据预处理流程 ===")
    print(f"1. 先分割数据（训练集: {train_ratio*100}%, 测试集: {(1-train_ratio)*100}%）")
    print(f"2. 对训练集和测试集分别进行滤波处理")
    print(f"3. 在训练集上拟合标准化器")
    print(f"4. 将训练集的标准化器应用到测试集")
    print("====================")
    
    # 第一步：先分割文件路径
    print("\n步骤1: 分割文件路径...")
    
    # 创建基础数据集来获取文件路径
    base_dataset = EMGDataset(root_dir, seq_length, filter_option, standardization_option)
    file_paths = base_dataset.file_paths
    
    # 随机分割文件路径
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(file_paths))
    train_size = int(len(file_paths) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_file_paths = [file_paths[i] for i in train_indices]
    test_file_paths = [file_paths[i] for i in test_indices]
    
    print(f"训练集文件数: {len(train_file_paths)}")
    print(f"测试集文件数: {len(test_file_paths)}")
    
    # 第二步：创建训练集并拟合标准化器
    print("\n步骤2: 创建训练集并拟合标准化器...")
    
    # 创建训练集
    train_dataset = EMGDataset(root_dir, seq_length, filter_option, standardization_option)
    train_dataset.file_paths = train_file_paths  # 替换为训练集文件路径
    
    # 在训练集上拟合标准化器
    if standardization_option in ['zscore', 'minmax']:
        train_dataset.fit_scaler()
        print(f"训练集标准化器已拟合")
    else:
        print("无标准化处理")
    
    # 第三步：创建测试集（使用训练集的标准化器）
    print("\n步骤3: 创建测试集...")
    
    test_dataset = EMGDataset(root_dir, seq_length, filter_option, standardization_option, scaler=train_dataset.scaler)
    test_dataset.file_paths = test_file_paths  # 替换为测试集文件路径
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, test_dataset

# 创建数据集
print("开始创建数据集...")
train_dataset, test_dataset = create_datasets_with_separate_processing()
print("数据集创建完成！")

def dynamic_collate_fn(batch):
    """
    动态批处理函数，处理变长序列数据
    返回：
        - sequences: 填充后的序列张量 [batch_size, max_seq_len, features]
        - lengths: 每个序列的实际长度
        - action_labels: 动作标签
        - status_labels: 状态标签
    """
    sequences, action_labels, status_labels = zip(*batch)
    
    # 获取每个序列的实际长度，确保所有长度都大于0
    lengths = [max(seq.shape[0], 1) for seq in sequences]  # 确保最小长度为1
    max_length = max(lengths)
    
    # 获取特征维度
    feature_dim = sequences[0].shape[1]
    
    # 创建填充后的张量
    padded_sequences = torch.zeros(len(sequences), max_length, feature_dim)
    
    for i, seq in enumerate(sequences):
        actual_length = seq.shape[0]
        if actual_length > 0:
            padded_sequences[i, :actual_length] = seq
        else:
            # 如果序列长度为0，使用第一个样本的数据填充
            padded_sequences[i, 0] = sequences[0][0] if sequences[0].shape[0] > 0 else torch.zeros(feature_dim)
    
    # 转换为张量
    action_labels = torch.tensor(action_labels)
    status_labels = torch.tensor(status_labels)
    lengths = torch.tensor(lengths)
    
    return padded_sequences, lengths, action_labels, status_labels

# 创建 DataLoader - 使用动态批处理
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # 训练集启用shuffle
    num_workers=0,  # Windows系统建议设为0
    collate_fn=dynamic_collate_fn
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # 测试集不启用shuffle
    num_workers=0,
    collate_fn=dynamic_collate_fn
)

# 创建完整数据集用于验证（如果需要）
full_dataset = EMGDataset(
    root_dir='./gaits_data', 
    seq_length=SEQ_LENGTH,  
    filter_option=FILTER,  # 启用滤波处理
    standardization_option=STANDARDIZATION,  # 启用标准化
    scaler=train_dataset.scaler  # 使用训练集的标准化器
)

full_loader = DataLoader(
    full_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0,
    collate_fn=dynamic_collate_fn
)

# 示例：查看一个批次的数据形状和标签分布
print("\n=== 数据加载器信息 ===")
for emg_seq, seq_lengths, action_label, status_label in train_loader:
    print(f"训练批次形状: {emg_seq.shape}")  # [batch_size, max_seq_len, 5]
    print(f"序列实际长度: {seq_lengths.tolist()}")
    print(f"最小序列长度: {min(seq_lengths.tolist())}")
    print(f"动作标签范围: {action_label.min().item()} - {action_label.max().item()}")
    print(f"状态标签范围: {status_label.min().item()} - {status_label.max().item()}")
    print(f"训练批次动作标签分布: {torch.bincount(action_label).tolist()}")
    print(f"训练批次状态标签分布: {torch.bincount(status_label).tolist()}")
    print(f"输入特征维度: {emg_seq.shape[-1]} (4个肌电通道 + 1个膝盖屈伸角度通道)")
    print(f"最大序列长度: {emg_seq.shape[1]}, 批次大小: {emg_seq.shape[0]}")
    break

print("\n数据预处理完成！")