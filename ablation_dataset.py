import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from tqdm import tqdm

# 标签定义
ACTION_LABELS = {'gait': 0, 'sitting': 1, 'standing': 2}
STATUS_LABELS = {'N': 0, 'A': 1}  # N: Normal, A: Abnormal
BATCH_SIZE = 32
SEQ_LENGTH = 1000  # 1秒窗口，1000Hz采样率
SEED = 42



# 滤波选项
FILTER_OPTIONS = {
    'none': 'No filtering',
    'bandpass': 'Bandpass filtering',
    'bandpass_notch': 'Bandpass + 60Hz notch filtering'
}
FILTER = 'bandpass'

# 标准化选项
STANDARDIZATION_OPTIONS = {
    'none': 'No standardization',
    'zscore': 'Z-score standardization',
    'minmax': 'Min-max normalization'
}
STANDARDIZATION = 'zscore'

# 可用参数选项（5个参数，可选择任意数量）
AVAILABLE_PARAMETERS = {
    'recto_femoral': 'Recto Femoral',
    'biceps_femoral': 'Biceps Femoral', 
    'vasto_medial': 'Vasto Medial',
    'emg_semitendinoso': 'EMG Semitendinoso',
    'flexo_extension': 'Flexo-Extension'
}

# 默认参数选择（保留前4个，排除关节角度）
DEFAULT_PARAMETERS = ['biceps_femoral', 'vasto_medial',  'flexo_extension', 'emg_semitendinoso']

class EMGAblationDataset(Dataset):
    def __init__(self, root_dir, seq_length=1000, filter_option='bandpass', standardization_option='zscore', 
                 scaler=None, selected_parameters=None):
        """
        消融实验EMG数据集类，可以选择保留哪几个参数（大腿肌肉和关节角度，可选择任意数量）
        
        Args:
            root_dir: 数据根目录
            seq_length: 序列长度（默认1000个时间点，1秒窗口）
            filter_option: 滤波选项
            standardization_option: 标准化选项
            scaler: 预训练的标准化器
            selected_parameters: 选择的参数列表（可选择任意数量），默认为前4个肌肉信号
        """
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.filter_option = filter_option
        self.standardization_option = standardization_option
        self.file_paths = []
        self.scaler = scaler
        
        # 设置选择的参数
        if selected_parameters is None:
            self.selected_parameters = DEFAULT_PARAMETERS
        else:
            self.selected_parameters = selected_parameters
        
        # 验证参数选择
        if len(self.selected_parameters) == 0:
            raise ValueError("必须选择至少1个参数")
        
        for param in self.selected_parameters:
            if param not in AVAILABLE_PARAMETERS:
                raise ValueError(f"无效的参数: {param}. 可用参数: {list(AVAILABLE_PARAMETERS.keys())}")
        
        # 获取排除的参数
        excluded_parameters = [p for p in AVAILABLE_PARAMETERS.keys() if p not in self.selected_parameters]
        if excluded_parameters:
            self.excluded_parameter = excluded_parameters[0]
        else:
            self.excluded_parameter = None
        
        # 滤波参数
        self.lowcut = 20  # 低通滤波截止频率 (Hz)
        self.highcut = 450  # 高通滤波截止频率 (Hz)
        self.notch_freq = 60  # 工频陷波频率 (Hz)
        self.fs = 1000  # 采样频率 (Hz)
        
        # 验证选项
        if filter_option not in FILTER_OPTIONS:
            raise ValueError(f"Invalid filter option: {filter_option}. Available: {list(FILTER_OPTIONS.keys())}")
        if standardization_option not in STANDARDIZATION_OPTIONS:
            raise ValueError(f"Invalid standardization option: {standardization_option}. Available: {list(STANDARDIZATION_OPTIONS.keys())}")
        
        print(f"Filter option: {FILTER_OPTIONS[filter_option]}")
        print(f"Standardization option: {STANDARDIZATION_OPTIONS[standardization_option]}")
        print(f"Selected parameters: {[AVAILABLE_PARAMETERS[p] for p in self.selected_parameters]}")
        print(f"Number of selected parameters: {len(self.selected_parameters)}")
        if self.excluded_parameter:
            print(f"Excluded parameter: {AVAILABLE_PARAMETERS[self.excluded_parameter]}")
        else:
            print("No parameters excluded (all parameters selected)")
        
        self._load_file_paths()

    def _load_file_paths(self):
        """加载所有CSV文件路径"""
        for folder in ['Abnormal', 'normal']:
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
            notch_freq = self.notch_freq / nyquist
            quality = 30  # 品质因数
            b_notch, a_notch = signal.iirnotch(notch_freq, quality)
            
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
            
            # 根据选择的参数获取对应的列名
            selected_columns = []
            for param in self.selected_parameters:
                if param == 'recto_femoral':
                    col_name = 'recto femoral'
                elif param == 'biceps_femoral':
                    col_name = 'biceps femoral'
                elif param == 'vasto_medial':
                    col_name = 'vasto medial'
                elif param == 'emg_semitendinoso':
                    col_name = 'emg semitendinoso'
                elif param == 'flexo_extension':
                    col_name = 'flexo-extension'
                
                # 检查列名兼容性
                if col_name in df.columns:
                    selected_columns.append(col_name)
                else:
                    # 尝试查找相似的列名
                    matching_cols = [c for c in df.columns if col_name.split()[0].lower() in c.lower()]
                    if matching_cols:
                        selected_columns.append(matching_cols[0])
                    else:
                        continue
            
            if len(selected_columns) >= 1:  # 需要至少1个参数
                data = df[selected_columns].values
                
                # 应用滤波（只对肌电信号应用滤波，不对关节角度应用）
                if self.filter_option != 'none':
                    # 创建掩码，只对肌电信号应用滤波
                    emg_mask = [i for i, param in enumerate(self.selected_parameters) 
                              if param != 'flexo_extension']
                    if emg_mask:
                        emg_data = data[:, emg_mask]
                        filtered_emg = self._apply_filter(emg_data)
                        data[:, emg_mask] = filtered_emg
                
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
            action = 'gait'
        
        # 状态标签解析
        status = 'A' if 'A' in base_name.upper() else 'N'
        
        action_label = ACTION_LABELS[action]
        status_label = STATUS_LABELS[status]
        
        return action_label, status_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """获取单个样本 - 根据选择的参数提取数据"""
        file_path = self.file_paths[idx]
        df = pd.read_csv(file_path)
        
        # 统一列名
        df.columns = df.columns.str.strip().str.lower()
        
        # 根据选择的参数获取对应的列名
        selected_columns = []
        for param in self.selected_parameters:
            if param == 'recto_femoral':
                col_name = 'recto femoral'
            elif param == 'biceps_femoral':
                col_name = 'biceps femoral'
            elif param == 'vasto_medial':
                col_name = 'vasto medial'
            elif param == 'emg_semitendinoso':
                col_name = 'emg semitendinoso'
            elif param == 'flexo_extension':
                col_name = 'flexo-extension'
            
            # 检查列名兼容性
            if col_name in df.columns:
                selected_columns.append(col_name)
            else:
                # 尝试查找相似的列名
                matching_cols = [c for c in df.columns if col_name.split()[0].lower() in c.lower()]
                if matching_cols:
                    selected_columns.append(matching_cols[0])
                else:
                    continue
        
        if len(selected_columns) < 1:
            raise ValueError(f"Missing expected columns in {file_path}. Available: {df.columns.tolist()}")
        
        data = df[selected_columns].values
        
        # 应用滤波（只对肌电信号应用滤波，不对关节角度应用）
        if self.filter_option != 'none':
            # 创建掩码，只对肌电信号应用滤波
            emg_mask = [i for i, param in enumerate(self.selected_parameters) 
                      if param != 'flexo_extension']
            if emg_mask:
                emg_data = data[:, emg_mask]
                filtered_emg = self._apply_filter(emg_data)
                data[:, emg_mask] = filtered_emg
        
        # 应用标准化
        if self.standardization_option in ['zscore', 'minmax'] and self.scaler is not None:
            data = self.scaler.transform(data)
        
        # 解析标签
        action_label, status_label = self._parse_labels(file_path)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # 窗口化处理 - 1秒随机窗口
        if data.shape[0] >= self.seq_length:
            # 随机选择1秒窗口
            start_idx = np.random.randint(0, data.shape[0] - self.seq_length + 1)
            window = data_tensor[start_idx:start_idx + self.seq_length]
        else:
            # 如果数据不足1秒，使用所有可用数据
            window = data_tensor
        
        return window, action_label, status_label


def create_ablation_datasets(root_dir='./gaits_data', seq_length=SEQ_LENGTH, 
                            filter_option=FILTER, standardization_option=STANDARDIZATION,
                            train_ratio=0.8, random_seed=SEED, selected_parameters=None):
    """
    创建消融实验的训练集和测试集
    
    Args:
        root_dir: 数据根目录
        seq_length: 序列长度
        filter_option: 滤波选项
        standardization_option: 标准化选项
        train_ratio: 训练集比例
        random_seed: 随机种子
        selected_parameters: 选择的参数列表（可选择任意数量），默认为前4个肌肉信号
        
    Returns:
        train_dataset: 训练集
        test_dataset: 测试集
    """
    print("=== Ablation Dataset Processing ===")
    print(f"1. Split data (Train: {train_ratio*100}%, Test: {(1-train_ratio)*100}%)")
    print(f"2. Apply filtering separately")
    print(f"3. Fit scaler on training set")
    print(f"4. Apply training scaler to test set")
    if selected_parameters:
        print(f"5. Selected parameters: {[AVAILABLE_PARAMETERS[p] for p in selected_parameters]}")
        print(f"6. Number of parameters: {len(selected_parameters)}")
    else:
        print(f"5. Using default parameters: {[AVAILABLE_PARAMETERS[p] for p in DEFAULT_PARAMETERS]}")
        print(f"6. Number of parameters: {len(DEFAULT_PARAMETERS)}")
    print("===================================")
    
    # 第一步：分割文件路径
    print("\nStep 1: Splitting file paths...")
    
    base_dataset = EMGAblationDataset(root_dir, seq_length, filter_option, standardization_option, 
                                     selected_parameters=selected_parameters)
    file_paths = base_dataset.file_paths
    
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(file_paths))
    train_size = int(len(file_paths) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_file_paths = [file_paths[i] for i in train_indices]
    test_file_paths = [file_paths[i] for i in test_indices]
    
    print(f"Training files: {len(train_file_paths)}")
    print(f"Test files: {len(test_file_paths)}")
    
    # 第二步：创建训练集并拟合标准化器
    print("\nStep 2: Creating training set and fitting scaler...")
    
    train_dataset = EMGAblationDataset(root_dir, seq_length, filter_option, standardization_option, 
                                     selected_parameters=selected_parameters)
    train_dataset.file_paths = train_file_paths
    
    if standardization_option in ['zscore', 'minmax']:
        train_dataset.fit_scaler()
        print("Training scaler fitted")
    else:
        print("No standardization")
    
    # 第三步：创建测试集
    print("\nStep 3: Creating test set...")
    
    test_dataset = EMGAblationDataset(root_dir, seq_length, filter_option, standardization_option, 
                                   scaler=train_dataset.scaler, selected_parameters=selected_parameters)
    test_dataset.file_paths = test_file_paths
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def dynamic_collate_fn(batch):
    """
    动态批处理函数，处理变长序列数据
    """
    sequences, action_labels, status_labels = zip(*batch)
    
    # 获取每个序列的实际长度
    lengths = [max(seq.shape[0], 1) for seq in sequences]
    max_length = max(lengths)
    
    # 获取特征维度（根据选择的参数数量）
    feature_dim = sequences[0].shape[1]
    
    # 创建填充后的张量
    padded_sequences = torch.zeros(len(sequences), max_length, feature_dim)
    
    for i, seq in enumerate(sequences):
        actual_length = seq.shape[0]
        if actual_length > 0:
            padded_sequences[i, :actual_length] = seq
        else:
            padded_sequences[i, 0] = sequences[0][0] if sequences[0].shape[0] > 0 else torch.zeros(feature_dim)
    
    # 转换为张量
    action_labels = torch.tensor(action_labels)
    status_labels = torch.tensor(status_labels)
    lengths = torch.tensor(lengths)
    
    return padded_sequences, lengths, action_labels, status_labels


# 创建数据集
if __name__ == "__main__":
    # 示例1：默认参数选择（排除关节角度）
    print("=== 示例1: 默认参数选择（排除关节角度） ===")
    print("Creating ablation datasets with default parameters...")
    train_dataset, test_dataset = create_ablation_datasets()
    print("Ablation datasets created successfully!")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        collate_fn=dynamic_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        collate_fn=dynamic_collate_fn
    )
    
    # 示例：查看一个批次的数据形状
    print("\n=== Data Loader Information ===")
    for emg_seq, seq_lengths, action_label, status_label in train_loader:
        print(f"Batch shape: {emg_seq.shape}")  # [batch_size, max_seq_len, num_parameters]
        print(f"Sequence lengths: {seq_lengths.tolist()}")
        print(f"Min sequence length: {min(seq_lengths.tolist())}")
        print(f"Action label range: {action_label.min().item()} - {action_label.max().item()}")
        print(f"Status label range: {status_label.min().item()} - {status_label.max().item()}")
        print(f"Action label distribution: {torch.bincount(action_label).tolist()}")
        print(f"Status label distribution: {torch.bincount(status_label).tolist()}")
        print(f"Input feature dimension: {emg_seq.shape[-1]} (selected parameters)")
        print(f"Max sequence length: {emg_seq.shape[1]}, Batch size: {emg_seq.shape[0]}")
        break
    
    print("\nAblation data preprocessing completed!")
    
    # 示例2：包含关节角度的参数选择
    print("\n=== 示例2: 包含关节角度的参数选择 ===")
    # 选择包含关节角度的参数组合（排除股直肌）
    selected_params_with_angle = ['biceps_femoral', 'vasto_medial', 'emg_semitendinoso', 'flexo_extension']
    
    print(f"Selected parameters: {[AVAILABLE_PARAMETERS[p] for p in selected_params_with_angle]}")
    print("Creating ablation datasets with joint angle included...")
    
    train_dataset_angle, test_dataset_angle = create_ablation_datasets(
        selected_parameters=selected_params_with_angle
    )
    
    # 创建数据加载器
    train_loader_angle = DataLoader(
        train_dataset_angle, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        collate_fn=dynamic_collate_fn
    )
    
    # 查看包含关节角度的数据形状
    print("\n=== Data Loader Information (with joint angle) ===")
    for emg_seq, seq_lengths, action_label, status_label in train_loader_angle:
        print(f"Batch shape: {emg_seq.shape}")  # [batch_size, max_seq_len, num_parameters]
        print(f"Sequence lengths: {seq_lengths.tolist()}")
        print(f"Input feature dimension: {emg_seq.shape[-1]} (selected parameters)")
        print(f"Max sequence length: {emg_seq.shape[1]}, Batch size: {emg_seq.shape[0]}")
        break
    
    print("\nAll ablation data preprocessing tests completed successfully!")