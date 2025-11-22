# EMG信号分类模型文档

## 模型概览

本目录包含4个不同版本的EMG信号分类模型，用于下肢肌电信号的动作和状态分类任务。

### 模型版本及性能

| 版本 | 模型架构 | 动作准确率 | 状态准确率 | 平均准确率 |
|------|----------|------------|------------|------------|
| v1.1.1 | LSTM模型 | 21.43% | 85.71% | 53.57% |
| v1.1.2 | LSTM+注意力机制 | 64.29% | 85.71% | 75.00% |
| v1.1.3 | CNN-LSTM模型 | 21.43% | 85.71% | 53.57% |
| v2.0 | Transformer模型 | 57.14% | 85.71% | 71.43% |

## 关键配置参数

### 1. LSTM隐藏层大小
所有LSTM相关模型的隐藏层大小统一设置为**200**：

```python
# models.py 中的配置
class LSMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=200, num_actions=3, dropout_rate=0.5):
        # LSTM隐藏层大小设置为200
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=False, batch_first=True)

class LSTM_AttentionModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=200, num_classes_action=3, num_classes_status=1, num_layers=2):
        # LSTM隐藏层大小设置为200
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

class CNN_LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=200, num_classes_action=3, num_classes_status=1, num_layers=2):
        # LSTM隐藏层大小设置为200
        self.lstm = nn.LSTM(num_filters*2, hidden_size, num_layers, batch_first=True)
```

### 2. 学习率调度器选择
在`factory_model.py`中配置了11种学习率调度器，当前默认使用**StepLR调度器**：

```python
# factory_model.py 中的配置
scheduler_name = 'step'  # 当前使用的调度器
scheduler = schedulers[scheduler_name]

# StepLR调度器配置
'step': optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
```



### 3. 数据窗口大小
数据窗口大小设置为**1000个时间点**（1秒数据，1000Hz采样频率）：

```python
# dataset.py 中的配置
SEQ_LENGTH = 1000  # 序列长度（窗口大小）

# 窗口化处理逻辑
if data.shape[0] >= self.seq_length:  # 1000个时间点
    # 随机选择窗口
    start_idx = np.random.randint(0, data.shape[0] - self.seq_length + 1)
    window = data_tensor[start_idx:start_idx + self.seq_length]
else:
    # 不进行零填充，直接使用所有可用数据
    window = data_tensor
```

## 模型架构详解

### v1.1.1 - LSTM模型
- **架构**: 单层LSTM + 共享全连接层 + 双任务分类器
- **隐藏层**: 200个LSTM单元
- **特点**: 支持变长序列，使用pack_padded_sequence处理

### v1.1.2 - LSTM+注意力机制
- **架构**: 双层LSTM + 时间注意力机制 + 双任务分类器
- **隐藏层**: 200个LSTM单元
- **特点**: 注意力机制加权重要时间步，提升特征聚焦能力

### v1.1.3 - CNN-LSTM模型
- **架构**: 1D卷积层 + LSTM + 共享全连接层
- **隐藏层**: 200个LSTM单元
- **特点**: CNN提取局部特征，LSTM建模时序依赖

### v2.0 - Transformer模型
- **架构**: Transformer编码器 + 共享全连接层
- **隐藏层**: 200个Transformer单元
- **特点**: 自注意力机制，并行处理序列

## 训练配置

### 优化器参数
```python
optimizer = optim.Adam(model.parameters(), lr=0.005)
```

### 损失函数
- **动作分类**: CrossEntropyLoss（3类分类）
- **状态分类**: BCELoss（2类分类）

### 训练轮数
- **训练轮数**: 5000 epochs
- **测试轮数**: 100 epochs

## 数据处理流程

### 输入数据特征
- **输入通道**: 5个通道（4个肌电通道 + 1个膝盖屈伸角度通道）
- **采样频率**: 1000Hz
- **窗口大小**: 1000个时间点（1秒数据）

### 预处理步骤
1. **滤波处理**: 20-450Hz巴特沃斯带通滤波
2. **标准化**: Z-score标准化
3. **窗口化**: 支持变长序列，不进行零填充
4. **动态批处理**: 使用dynamic_collate_fn处理不同长度序列

## 性能分析

### 最佳模型
- **v1.1.2**（LSTM+注意力机制）表现最佳，平均准确率达75.00%
- 状态分类任务表现稳定，所有模型均达到85.71%准确率
- 动作分类任务v1.1.2表现最优（64.29%）

### 模型特点对比
- **LSTM模型**: 简单有效，适合时序建模
- **LSTM+注意力**: 增强特征聚焦，提升动作分类性能
- **CNN-LSTM**: 结合局部和全局特征
- **Transformer**: 自注意力机制，适合长序列建模

## 可视化文件

每个模型版本目录包含以下可视化图表：
- `learning_rate_curve.png`: 学习率变化曲线
- `train_loss_curve.png`: 训练损失曲线
- `val_loss_curve.png`: 验证损失曲线
- `confusion_matrix_action.png`: 动作分类混淆矩阵
- `confusion_matrix_status.png`: 状态分类混淆矩阵
- `feature_importance_pca.png`: 特征重要性分析
- `error_analysis.png`: 错误分析图表

## 使用说明

### 加载模型
```python
from factory_model import get_model

# 加载指定版本的模型
model = get_model('v1.1.2', input_size=5)
```

### 切换学习率调度器
```python
from factory_model import set_scheduler

# 切换到余弦退火调度器
set_scheduler('cosine')
```

### 获取当前配置信息
```python
from factory_model import get_scheduler_info

info = get_scheduler_info()
print(f"当前调度器: {info['name']}")
print(f"描述: {info['description']}")
print(f"当前学习率: {info['current_lr']}")
```

## 总结

本模型库提供了多种EMG信号分类架构，重点配置包括：
- **LSTM隐藏层统一为200**，平衡模型复杂度和性能
- **StepLR学习率调度器**，每50个epoch衰减0.1倍
- **1000个时间点的窗口大小**，适合1秒时长的肌电信号分析
- **支持变长序列处理**，适应不同长度的输入数据

最佳实践推荐使用**v1.1.2模型**（LSTM+注意力机制），在动作分类任务上表现最优。