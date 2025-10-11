# 下肢肌电信号分类模型库

本目录包含用于下肢肌电信号分类的深度学习模型，支持动作识别（行走、坐姿、站立）和状态分类（正常、异常）。

## 模型概览

| 模型版本 | 模型架构 | 动作准确率 | 状态准确率 | 平均准确率 |
|---------|---------|-----------|-----------|-----------|
| v1.1.1 | LSMModel | 42.86% | 85.71% | 64.29% |
| v1.1.2 | LSTM_AttentionModel | 78.57% | 85.71% | 82.14% |
| v1.1.3 | CNN_LSTMModel | 28.57% | 78.57% | 53.57% |
| v2.0 | TransformerModel | 78.57% | 85.71% | 82.14% |

## 模型架构说明

### 1. LSMModel v1.1.1
- **架构类型**: 基础LSTM模型
- **特点**: 简单的LSTM架构，适合基础时序模式学习
- **适用场景**: 基础研究和性能基准测试

### 2. LSTM_AttentionModel v1.1.2
- **架构类型**: LSTM + 注意力机制
- **特点**: 引入注意力机制，增强关键时间步的权重
- **适用场景**: 需要关注关键时间点的应用

### 3. CNN_LSTMModel v1.1.3
- **架构类型**: CNN + LSTM混合架构
- **特点**: CNN提取局部特征，LSTM学习时序依赖
- **适用场景**: 需要同时处理局部和全局时序特征

### 4. TransformerModel v2.0
- **架构类型**: Transformer架构
- **特点**: 自注意力机制，并行处理能力强
- **适用场景**: 长序列处理和复杂时序模式学习

## 输入数据属性

### 数据源
- **数据格式**: CSV文件
- **采样频率**: 1000 Hz
- **数据来源**: 下肢肌电信号采集

### 输入特征（5个通道）
1. **Recto Femoral** - 股直肌肌电信号
2. **Biceps Femoral** - 股二头肌肌电信号  
3. **Vasto Medial** - 股内侧肌肌电信号
4. **EMG Semitendinoso** - 半腱肌肌电信号
5. **Flexo-Extension** - 膝盖屈伸角度

### 标签定义
- **动作标签**: gait(0), sitting(1), standing(2)
- **状态标签**: Normal(0), Abnormal(1)

## 数据处理流程

### 1. 数据预处理
```python
# 全局配置参数
SEQ_LENGTH = 2000  # 序列长度（窗口大小）
BATCH_SIZE = 32    # 批次大小
```

### 2. 滤波处理
- **滤波器类型**: 巴特沃斯带通滤波器
- **频率范围**: 20-450 Hz
- **目的**: 去除噪声和干扰信号

### 3. 标准化处理
- **方法**: Z-score标准化
- **目的**: 统一数据尺度，提高模型训练稳定性

### 4. 窗口化处理
```python
# 支持变长序列，不进行零填充
if data.shape[0] >= self.seq_length:  # 2000个时间点
    # 随机选择窗口
    start_idx = np.random.randint(0, data.shape[0] - self.seq_length + 1)
    window = data_tensor[start_idx:start_idx + self.seq_length]
else:
    # 使用实际数据长度
    window = data_tensor
```

### 5. 动态批处理
- **方法**: 动态填充到批次中的最大长度
- **输出形状**: [batch_size, max_seq_len, 5]
- **包含信息**: 序列实际长度、动作标签、状态标签

## 模型性能分析

### 最佳表现模型
1. **LSTM_AttentionModel v1.1.2** - 动作准确率78.57%，状态准确率85.71%
2. **TransformerModel v2.0** - 动作准确率78.57%，状态准确率85.71%

### 性能特点
- **状态分类**: 所有模型在状态分类上都表现较好（≥78.57%）
- **动作识别**: LSTM_AttentionModel和TransformerModel表现最佳
- **平均性能**: v1.1.2和v2.0模型达到82.14%的平均准确率

## 模型选择建议

### 高精度需求
- **推荐模型**: TransformerModel v2.0 或 LSTM_AttentionModel v1.1.2
- **准确率**: 82.14%
- **特点**: 平衡的动作和状态识别能力

### 平衡性能
- **推荐模型**: LSTM_AttentionModel v1.1.2
- **准确率**: 82.14%
- **特点**: 稳定的性能表现

### 中等性能
- **推荐模型**: CNN_LSTMModel v1.1.3
- **准确率**: 53.57%
- **特点**: 混合架构，适合特定应用场景

## 可视化分析

### 模型架构图
- **位置**: `../models_v1/`目录
- **包含**: 各版本的模型架构示意图

### 训练过程可视化
每个模型目录包含以下训练分析图片：
- `complexity_vs_performance.png` - 复杂度与性能关系
- `confusion_matrix_action.png` - 动作分类混淆矩阵
- `confusion_matrix_status.png` - 状态分类混淆矩阵
- `loss_curves.png` - 损失曲线
- `accuracy_curves.png` - 准确率曲线
- `training_time_comparison.png` - 训练时间对比

## 训练配置

### 超参数设置
- **学习率**: 自适应调整
- **优化器**: Adam
- **损失函数**: 交叉熵损失
- **训练轮数**: 根据验证集性能早停

### 数据划分
- **训练集**: 80%
- **测试集**: 20%
- **随机种子**: 42（确保可重复性）

## 使用说明

### 模型加载
```python
from factory_model import create_model

# 加载指定版本的模型
model = create_model('v2.0', num_classes=3)  # 动作分类
model.load_state_dict(torch.load('models/v2.0/best_model.pth'))
```

### 数据预处理
```python
from dataset import EMGDataset

# 创建数据集实例
dataset = EMGDataset(
    root_dir='./gaits_data',
    seq_length=2000,  # 窗口大小
    apply_filtering=True,
    apply_standardization=True
)
```

### 预测示例
```python
# 获取单个样本
window, action_label, status_label = dataset[0]

# 模型预测
model.eval()
with torch.no_grad():
    action_pred, status_pred = model(window.unsqueeze(0))
```

## 技术参数

### 输入要求
- **序列长度**: 支持变长序列，最大2000个时间点
- **特征维度**: 5个通道（4肌电 + 1角度）
- **数据类型**: float32

### 输出格式
- **动作预测**: 3个类别的概率分布
- **状态预测**: 2个类别的概率分布

## 注意事项

1. **数据质量**: 确保输入数据的采样频率为1000Hz
2. **特征顺序**: 严格按照5个通道的顺序输入数据
3. **模型版本**: 不同版本的模型架构和参数不同
4. **性能基准**: 准确率基于测试集评估，实际应用可能有所差异

## 更新日志

- **v1.1.1**: 基础LSTM模型，建立性能基准
- **v1.1.2**: 引入注意力机制，显著提升动作识别性能
- **v1.1.3**: CNN+LSTM混合架构，探索不同特征提取方式
- **v2.0**: Transformer架构，利用自注意力机制处理长序列

---

*注：所有准确率数据基于实际测试结果，模型文件保存在各自版本目录中。*