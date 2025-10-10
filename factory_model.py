import torch
import torch.nn as nn
import torch.optim as optim
from importlib import import_module
import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_version = "v1.1.1"
test_epoches_num=10
train_epoches_num=10000


MODEL_CLASSES = {
    'v1.1.1': models.LSMModel,          
    'v1.1.2':models.LSTM_AttentionModel,
    'v1.1.3':models.CNN_LSTMModel,
    'v2.0': models.TransformerModel,     

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
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 学习率调度器配置
schedulers = {
    'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.1),
    'step': optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
    'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
    'exponential': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
}

# 默认使用plateau调度器
scheduler = schedulers['plateau']

# 学习率历史记录
learning_rate_history = []

