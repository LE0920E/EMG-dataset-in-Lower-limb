import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from factory_model import device

ACTION_LABELS = {'gait': 0, 'sitting': 1, 'standing': 2}
STATUS_LABELS = {'N': 0, 'A': 1}  # N: Normal, A: Abnormal
BATCH_SIZE= 1024

class EMGDataset(Dataset):
    def __init__(self, root_dir, seq_length=30, apply_standardization=True):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.apply_standardization = apply_standardization
        self.file_paths = []
        self.scaler = None

        self._load_file_paths()
        if self.apply_standardization:
            self._fit_scaler()

    def _load_file_paths(self):
        for folder in ['Abnormal', 'Normal']:
            class_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(class_path):
                continue
            for file in os.listdir(class_path):
                if file.endswith('.csv'):
                    self.file_paths.append(os.path.join(class_path, file))

    def _fit_scaler(self):
        """ 收集所有数据拟合标准化器 """
        all_data = []
        for file_path in tqdm(self.file_paths, desc="Fitting scaler"):
            df = pd.read_csv(file_path)

            # 统一列名为小写格式
            df.columns = df.columns.str.strip().str.lower()
            wanted_columns = ['recto femoral', 'biceps femoral', 'vasto medial', 'emg semitendinoso']

            if all(col in df.columns for col in wanted_columns):
                data = df[wanted_columns].values
                all_data.append(data)

        all_data = np.vstack(all_data)
        self.scaler = StandardScaler()
        self.scaler.fit(all_data)

    def _parse_labels(self, filename):
        base_name = os.path.basename(filename).split(' - ')[0]
        action = None
        for act in ['gait', 'sitting', 'standing']:
            if act in base_name:
                action = act
                break
        status = 'A' if 'A' in base_name else 'N'
        return ACTION_LABELS[action], STATUS_LABELS[status]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        df = pd.read_csv(file_path)

        # 标准化列名
        df.columns = df.columns.str.strip().str.lower()
        wanted_columns = ['recto femoral', 'biceps femoral', 'vasto medial', 'emg semitendinoso']

        if not all(col in df.columns for col in wanted_columns):
            raise ValueError(f"Missing expected columns in {file_path}")

        data = df[wanted_columns].values

        # 应用标准化
        if self.apply_standardization and self.scaler is not None:
            data = self.scaler.transform(data)

        action_label, status_label = self._parse_labels(file_path)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 窗口化处理
        if data.shape[0] >= self.seq_length:
            start_idx = np.random.randint(0, data.shape[0] - self.seq_length + 1)
            window = data_tensor[start_idx:start_idx + self.seq_length]
        else:
            pad = torch.zeros((self.seq_length - data.shape[0], data.shape[1]))
            window = torch.cat([data_tensor, pad], dim=0)

        return window, action_label, status_label


from torch.utils.data import random_split

# 创建完整数据集
full_dataset = EMGDataset(root_dir='./gaits_data', seq_length=30, apply_standardization=True)

# 划分比例：80% 训练，20% 测试
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 随机划分
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
full_loader=DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 示例：查看一个批次的数据
for emg_seq, action_label, status_label in train_loader:
    print("Train Shape:", emg_seq.shape)
    print("Train Action Labels:", action_label)
    print("Train Status Labels:", status_label)
    break

for emg_seq, action_label, status_label in test_loader:
    print("Test Shape:", emg_seq.shape)
    print("Test Action Labels:", action_label)
    print("Test Status Labels:", status_label)
    break