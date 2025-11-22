import torch
import torch.nn as nn
import torch.optim as optim
from importlib import import_module
import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_version = "v1"
test_epoches_num=100
train_epoches_num=10

output_dir = f"./models/{model_version}"


MODEL_CLASSES = {
    'v1': models.LSMModel,          
    'v2':models.LSTM_AttentionModel,
    'v3':models.CNN_LSTMModel,
    'v4': models.TransformerModel,     

}

def get_model(version, **kwargs):
    """
    根据版本名返回模型实例
    支持传参，如 hidden_dim=64
    """
    if version not in MODEL_CLASSES:
        available = list(MODEL_CLASSES.keys())
        raise ValueError(f"Unknown model version: {version}. Available: {available}")
    
    model_class = MODEL_CLASSES[version]
    model_instance = model_class(**kwargs)  # 实例化
    return model_instance

# 创建模型实例，指定输入特征维度为5（4个肌电通道 + 1个膝盖屈伸角度通道）
model = get_model(model_version, input_size=5)        
model = model.to(device)               


criterion_action = nn.CrossEntropyLoss()  
criterion_status = nn.BCELoss()          
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器配置
schedulers = {
    'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.1),
    'step': optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
    'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
    'exponential': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2),
    'cyclic': optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000, mode='triangular'),
    'one_cycle': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=train_epoches_num, steps_per_epoch=10),
    'linear_warmup': optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100),
    'multi_step': optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5),
    'lambda': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch),
    'none': None  # 不使用调度器
}

# 调度器描述信息
scheduler_descriptions = {
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

# 默认使用plateau调度器
# 如需修改调度器，请将下面的 'plateau' 替换为其他调度器名称
# 可用的调度器: ['plateau', 'step', 'cosine', 'exponential', 'cosine_warm_restarts', 'cyclic', 'one_cycle', 'linear_warmup', 'multi_step', 'lambda', 'none']
scheduler_name = 'none'  # 修改此行来切换调度器
scheduler = schedulers[scheduler_name]
print(f"当前使用的学习率调度器: {scheduler_name} - {scheduler_descriptions[scheduler_name]}")

# 调度器选择函数
def set_scheduler(name):
    """设置学习率调度器"""
    global scheduler, scheduler_name
    if name in schedulers:
        scheduler_name = name
        scheduler = schedulers[name]
        print(f"已切换到调度器: {name} - {scheduler_descriptions[name]}")
    else:
        available = list(schedulers.keys())
        raise ValueError(f"未知的调度器: {name}。可用的调度器: {available}")

def get_available_schedulers():
    """获取可用的调度器列表"""
    return list(schedulers.keys())

def get_scheduler_info():
    """获取当前调度器信息"""
    return {
        'name': scheduler_name,
        'description': scheduler_descriptions[scheduler_name],
        'current_lr': optimizer.param_groups[0]['lr'] if scheduler else optimizer.param_groups[0]['lr']
    }

# 学习率历史记录
learning_rate_history = []

