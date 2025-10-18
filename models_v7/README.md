# 下肢肌电信号分类模型库 (v7版本)

本目录包含v7版本的下肢肌电信号分类深度学习模型，支持动作识别（行走、坐姿、站立）和状态分类（正常、异常）。

## 模型概览

| 模型版本 | 模型架构 | 动作准确率 | 状态准确率 | 平均准确率 |
|---------|---------|-----------|-----------|-----------|
| v1.1.1 | LSMModel | 21.43% | 85.71% | 53.57% |
| v1.1.2 | LSTM_AttentionModel | 78.57% | 85.71% | 82.14% |
| v1.1.3 | CNN_LSTMModel | 21.43% | 85.71% | 53.57% |
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
SEQ_LENGTH = 1000  # 序列长度（窗口大小）
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
if data.shape[0] >= self.seq_length:  # 1000个时间点
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

## 学习率调度策略

### 当前配置
- **优化器**: Adam
- **初始学习率**: 0.005
- **调度器类型**: 步长调度器 (StepLR)
- **调度参数**: 每50个epoch学习率乘以0.1

### 可用的学习率调度器
```python
# 在factory_model.py中可切换的调度器
schedulers = {
    'plateau': '基于验证损失的平台检测调度器（耐心30，因子0.1）',
    'step': '步长调度器（每50个epoch学习率乘以0.1）',
    'cosine': '余弦退火调度器（周期100个epoch）',
    'exponential': '指数衰减调度器（每个epoch乘以0.95）',
    'cosine_warm_restarts': '余弦退火热重启调度器（T0=50，T_mult=2）',
    'cyclic': '循环学习率调度器（三角模式，基础学习率0.001，最大0.01）',
    'one_cycle': '单周期学习率调度器（最大学习率0.01）',
    'linear_warmup': '线性预热调度器（从10%线性增加到100%）',
    'multi_step': '多步长调度器（在30、60、90epoch时衰减）',
    'lambda': 'Lambda调度器（每个epoch乘以0.95）',
    'none': '不使用学习率调度'
}
```

### 切换调度器方法
```python
from factory_model import set_scheduler

# 切换到余弦退火调度器
set_scheduler('cosine')

# 查看当前调度器信息
from factory_model import get_scheduler_info
info = get_scheduler_info()
print(f"当前调度器: {info['name']}")
print(f"描述: {info['description']}")
print(f"当前学习率: {info['current_lr']}")
```

## 模型性能分析

### 最佳表现模型
1. **LSTM_AttentionModel v1.1.2** - 动作准确率78.57%，状态准确率85.71%
2. **TransformerModel v2.0** - 动作准确率78.57%，状态准确率85.71%

### 性能特点
- **状态分类**: 所有模型在状态分类上都表现较好（≥85.71%）
- **动作识别**: LSTM_AttentionModel和TransformerModel表现最佳（78.57%）
- **平均性能**: v1.1.2和v2.0模型达到82.14%的平均准确率

### 性能对比分析
- **v1.1.1 vs v1.1.3**: 两者性能相同（53.57%），但架构不同
- **v1.1.2 vs v2.0**: 性能完全相同（82.14%），但架构原理不同
- **注意力机制优势**: v1.1.2相比v1.1.1性能提升显著（82.14% vs 53.57%）

## 训练配置

### 超参数设置
- **训练轮数**: 5000 epochs
- **测试轮数**: 100 epochs
- **优化器**: Adam (lr=0.005)
- **损失函数**: 
  - 动作分类: 交叉熵损失
  - 状态分类: 二元交叉熵损失

### 数据划分
- **训练集**: 80%
- **测试集**: 20%
- **随机种子**: 42（确保可重复性）

## 可视化分析

### 训练过程可视化
每个模型目录包含以下训练分析图片：
- `learning_rate_curve.png` - 学习率变化曲线
- `train_loss_curve.png` - 训练损失曲线
- `val_loss_curve.png` - 验证损失曲线
- `train_action_accuracy_curve.png` - 训练动作准确率曲线
- `val_action_accuracy_curve.png` - 验证动作准确率曲线
- `train_status_accuracy_curve.png` - 训练状态准确率曲线
- `val_status_accuracy_curve.png` - 验证状态准确率曲线

### 性能分析可视化
- `complexity_vs_performance.png` - 复杂度与性能关系
- `confusion_matrix_action.png` - 动作分类混淆矩阵
- `confusion_matrix_status.png` - 状态分类混淆矩阵
- `error_analysis.png` - 错误分析
- `feature_importance_pca.png` - 特征重要性PCA分析

## 使用说明

### 模型加载
```python
from factory_model import create_model

# 加载指定版本的模型
model = create_model('v2.0', input_size=5)  # 输入特征维度为5
model.load_state_dict(torch.load('models_v7/v2.0/best_model.pth'))
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
    
# 获取预测结果
action_class = torch.argmax(action_pred, dim=1).item()
status_class = (status_pred > 0.5).float().item()
```

## 技术参数

### 输入要求
- **序列长度**: 支持变长序列，最大2000个时间点
- **特征维度**: 5个通道（4肌电 + 1角度）
- **数据类型**: float32

### 输出格式
- **动作预测**: 3个类别的概率分布
- **状态预测**: 2个类别的概率分布

## 模型选择建议

### 高精度需求
- **推荐模型**: TransformerModel v2.0 或 LSTM_AttentionModel v1.1.2
- **准确率**: 82.14%
- **特点**: 平衡的动作和状态识别能力

### 平衡性能
- **推荐模型**: LSTM_AttentionModel v1.1.2
- **准确率**: 82.14%
- **特点**: 稳定的性能表现，架构相对简单

### 基础研究
- **推荐模型**: LSMModel v1.1.1 或 CNN_LSTMModel v1.1.3
- **准确率**: 53.57%
- **特点**: 适合算法验证和性能基准测试

## 注意事项

1. **数据质量**: 确保输入数据的采样频率为1000Hz
2. **特征顺序**: 严格按照5个通道的顺序输入数据
3. **窗口大小**: 当前配置为2000个时间点，可根据需要调整
4. **调度器选择**: 可根据训练效果切换不同的学习率调度器
5. **性能基准**: 准确率基于测试集评估，实际应用可能有所差异

## 版本说明

- **v7版本特点**: 使用步长学习率调度器，训练轮数5000
- **数据窗口**: 1000个时间点
- **调度策略**: StepLR (每50epoch衰减0.1倍)
- **性能表现**: 最佳模型达到82.14%平均准确率

---

*注：所有准确率数据基于实际测试结果，模型文件保存在各自版本目录中。学习率调度器可在factory_model.py中灵活切换。*