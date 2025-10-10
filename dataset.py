import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy import signal
from tqdm import tqdm
from factory_model import device

# 标签定义
ACTION_LABELS = {'gait': 0, 'sitting': 1, 'standing': 2}
STATUS_LABELS = {'N': 0, 'A': 1}  # N: Normal, A: Abnormal
BATCH_SIZE = 32  # 减小批次大小以适应更大的窗口

class EMGDataset(Dataset):
    def __init__(self, root_dir, seq_length=1000, apply_filtering=True, apply_standardization=True):
        """
        增强版EMG数据集类，包含膝盖屈伸角度和滤波处理
        
        Args:
            root_dir: 数据根目录
            seq_length: 序列长度（默认1000个时间点）
            apply_filtering: 是否应用滤波处理
            apply_standardization: 是否应用标准化
        """
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.apply_filtering = apply_filtering
        self.apply_standardization = apply_standardization
        self.file_paths = []
        self.scaler = None
        
        # 滤波参数
        self.lowcut = 20  # 低通滤波截止频率 (Hz)
        self.highcut = 450  # 高通滤波截止频率 (Hz)
        self.fs = 1000  # 采样频率 (Hz)
        
        self._load_file_paths()
        if self.apply_standardization:
            self._fit_scaler()

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

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """巴特沃斯带通滤波器"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data, axis=0)
        return y

    def _fit_scaler(self):
        """收集所有数据拟合标准化器"""
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
                if self.apply_filtering:
                    data = self._butter_bandpass_filter(data, self.lowcut, self.highcut, self.fs)
                
                all_data.append(data)

        if all_data:
            all_data = np.vstack(all_data)
            self.scaler = StandardScaler()
            self.scaler.fit(all_data)
            print(f"Scaler fitted with {all_data.shape[0]} samples")
        else:
            print("Warning: No valid data found for scaler fitting")

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
        
        return ACTION_LABELS[action], STATUS_LABELS[status]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """获取单个样本"""
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
        if self.apply_filtering:
            data = self._butter_bandpass_filter(data, self.lowcut, self.highcut, self.fs)
        
        # 应用标准化
        if self.apply_standardization and self.scaler is not None:
            data = self.scaler.transform(data)
        
        # 解析标签
        action_label, status_label = self._parse_labels(file_path)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # 窗口化处理（1000个时间点）
        if data.shape[0] >= self.seq_length:
            # 随机选择窗口
            start_idx = np.random.randint(0, data.shape[0] - self.seq_length + 1)
            window = data_tensor[start_idx:start_idx + self.seq_length]
        else:
            # 零填充
            pad = torch.zeros((self.seq_length - data.shape[0], data.shape[1]))
            window = torch.cat([data_tensor, pad], dim=0)
        
        return window, action_label, status_label


from torch.utils.data import random_split

# 创建完整数据集（包含膝盖屈伸角度和滤波处理）
full_dataset = EMGDataset(
    root_dir='./gaits_data', 
    seq_length=1000,  # 增大窗口至1000个时间点
    apply_filtering=True,  # 启用滤波处理
    apply_standardization=True  # 启用标准化
)

print(f"完整数据集大小: {len(full_dataset)} 个样本")

# 划分比例：80% 训练，20% 测试
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 随机划分（固定随机种子确保可重复性）
train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # 训练集启用shuffle
    num_workers=0  # Windows系统建议设为0
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # 测试集不启用shuffle
    num_workers=0
)

full_loader = DataLoader(
    full_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=0
)

# 示例：查看一个批次的数据形状和标签分布
print("\n=== 数据加载器信息 ===")
for emg_seq, action_label, status_label in train_loader:
    print(f"训练批次形状: {emg_seq.shape}")  # [batch_size, 1000, 5]
    print(f"训练批次动作标签分布: {torch.bincount(action_label).tolist()}")
    print(f"训练批次状态标签分布: {torch.bincount(status_label).tolist()}")
    print(f"输入特征维度: {emg_seq.shape[-1]} (4个肌电通道 + 1个膝盖屈伸角度通道)")
    break

print("\n数据预处理完成！")