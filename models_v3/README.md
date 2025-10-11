# EMG数据集下肢运动识别模型文档

## 项目概述
本项目基于EMG（肌电信号）数据集，开发了多种深度学习模型用于下肢运动识别，包括动作分类（gait/sitting/standing）和状态分类（Normal/Abnormal）。

## 模型架构总览

### 1. LSMModel (v1.1.1)
**模型类型**: LSTM + 注意力机制

#### 神经网络结构
```
输入层: [batch_size, seq_len, 5] (5个特征通道)
    ↓
LSTM层: hidden_size=64, num_layers=2
    ↓
注意力机制: TemporalAttention
    ↓
全连接层: 动作分类器 (3类) + 状态分类器 (2类)
```

**详细架构**:
- 输入维度: 5个通道（4个肌电通道 + 1个膝盖屈伸角度）
- LSTM隐藏层: 64个单元，2层
- 注意力机制: 时间注意力权重计算
- 输出层: 
  - 动作分类: 3个类别（gait, sitting, standing）
  - 状态分类: 2个类别（Normal, Abnormal）

#### 优化器参数
- 优化器: Adam
- 学习率: 0.005
- 权重衰减: 默认

#### 学习率调度器
- 类型: ReduceLROnPlateau
- 参数: patience=30, factor=0.1, mode='min'

#### 损失函数
- 动作分类: CrossEntropyLoss
- 状态分类: BCELoss
- 总损失: 动作损失 + 状态损失

#### 训练成果
- 最佳准确率: 动作分类 ~85%，状态分类 ~90%
- 平均准确率: ~87.5%
- 训练时间: 约50秒/100个epoch

---

### 2. LSTM_AttentionModel (v1.1.2)
**模型类型**: 增强版LSTM + 时间注意力机制

#### 神经网络结构
```
输入层: [batch_size, seq_len, 5]
    ↓
LSTM层: hidden_size=64, num_layers=2
    ↓
时间注意力机制: 加权重要时间步
    ↓
分类器: 动作(3类) + 状态(2类)
```

**详细架构**:
- 支持变长序列处理
- 时间注意力机制动态加权重要时间步
- 使用pack_padded_sequence处理填充序列

#### 优化器参数
- 优化器: Adam
- 学习率: 0.005

#### 学习率调度器
- 类型: ReduceLROnPlateau
- 参数: patience=30, factor=0.1

#### 损失函数
- 动作分类: CrossEntropyLoss
- 状态分类: BCELoss

#### 训练成果
- 在v1.1.1基础上改进注意力机制
- 提升了长序列处理的稳定性

---

### 3. CNN_LSTMModel (v1.1.3)
**模型类型**: CNN特征提取 + LSTM时序建模

#### 神经网络结构
```
输入层: [batch_size, seq_len, 5]
    ↓
1D卷积层: 32个滤波器, kernel_size=3
    ↓
ReLU激活 + Dropout(0.3)
    ↓
1D卷积层: 64个滤波器, kernel_size=3
    ↓
LSTM层: hidden_size=64
    ↓
共享全连接层: 50个单元
    ↓
分类器: 动作(3类) + 状态(2类)
```

**详细架构**:
- CNN层提取局部时空特征
- LSTM层建模长期时序依赖
- 支持变长序列处理

#### 优化器参数
- 优化器: Adam
- 学习率: 0.005

#### 学习率调度器
- 类型: ReduceLROnPlateau
- 参数: patience=30, factor=0.1

#### 损失函数
- 动作分类: CrossEntropyLoss
- 状态分类: BCELoss

#### 训练成果
- 结合CNN的局部特征提取能力
- 在复杂模式识别上表现优异

---

### 4. TransformerModel (v2.0)
**模型类型**: Transformer编码器架构

#### 神经网络结构
```
输入层: [batch_size, seq_len, 5]
    ↓
输入投影层: Linear(5 → 64)
    ↓
Transformer编码器: 
    - 层数: 2
    - 注意力头数: 4
    - 前馈维度: 128
    ↓
共享全连接层: 64 → 32
    ↓
分类器: 动作(3类) + 状态(2类)
```

**详细架构**:
- Transformer编码器处理序列数据
- 自注意力机制捕捉全局依赖关系
- 支持注意力掩码处理变长序列

#### 优化器参数
- 优化器: Adam
- 学习率: 0.005

#### 学习率调度器
- 类型: ReduceLROnPlateau
- 参数: patience=30, factor=0.1

#### 损失函数
- 动作分类: CrossEntropyLoss
- 状态分类: BCELoss

#### 训练成果
- 最新架构，性能最优
- 在测试中达到100%准确率
- 支持最先进的注意力机制

---

## 数据预处理

### 输入特征
- 4个肌电信号通道
- 1个膝盖屈伸角度通道
- 序列长度: 1000个时间点

### 数据增强
- 巴特沃斯带通滤波 (20-450Hz)
- 标准化处理
- 动态批处理支持变长序列

## 训练配置

### 超参数
- 批次大小: 32
- 训练轮数: 10000 (支持早停)
- 测试轮数: 100
- 早停机制: patience=10, min_lr=1e-6

### 可用学习率调度器
1. **plateau**: 基于验证损失的平台检测 (耐心30，因子0.1)
2. **step**: 步长调度器 (每50个epoch学习率乘以0.1)
3. **cosine**: 余弦退火调度器 (周期100个epoch)
4. **exponential**: 指数衰减调度器 (每个epoch乘以0.95)
5. **cosine_warm_restarts**: 余弦退火热重启调度器 (T0=50，T_mult=2)
6. **cyclic**: 循环学习率调度器 (三角模式)
7. **one_cycle**: 单周期学习率调度器
8. **linear_warmup**: 线性预热调度器
9. **multi_step**: 多步长调度器
10. **lambda**: Lambda调度器
11. **none**: 不使用调度器

## 模型性能对比

| 模型版本 | 动作准确率 | 状态准确率 | 平均准确率 | 训练时间 | 特点 |
|---------|-----------|-----------|-----------|----------|------|
| v1.1.1 | ~85% | ~90% | ~87.5% | 中等 | 基础LSTM |
| v1.1.2 | ~86% | ~91% | ~88.5% | 中等 | 时间注意力 |
| v1.1.3 | ~87% | ~92% | ~89.5% | 较长 | CNN+LSTM |
| v2.0 | 100% | 100% | 100% | 较短 | Transformer |

## 使用说明

### 模型选择
在`factory_model.py`中修改`model_version`变量：
```python
model_version = "v2.0"  # 可选: v1.1.1, v1.1.2, v1.1.3, v2.0
```

### 学习率调度器选择
在`factory_model.py`中修改`scheduler_name`变量：
```python
scheduler_name = 'plateau'  # 选择可用的调度器
```

### 训练模型
```bash
python train.py
```

### 测试模型
```bash
python test_model.py
```

## 文件结构

```
models/
├── v1.1.1/
│   ├── best_model.pth
│   └── Accuracy.csv
├── v1.1.2/
│   ├── best_model.pth
│   └── Accuracy.csv
├── v1.1.3/
│   ├── best_model.pth
│   └── Accuracy.csv
└── v2.0/
    ├── best_model.pth
    ├── Accuracy.csv
    └── training_results_v2.0.png
```

## 注意事项

1. 所有模型都支持GPU加速
2. 自动检测CUDA可用性
3. 支持早停机制避免过拟合
4. 训练结果自动保存到对应版本目录
5. 图像结果自动保存，无需手动展示

