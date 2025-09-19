import torch
import torch.nn as nn
import torch.optim as optim
from importlib import import_module
import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_version = "v1.1.2"
test_epoches_num=10
train_epoches_num=10000


MODEL_CLASSES = {
    'v1.1.1': models.LSMModel,          
    'v1.1.2':models.LSTM_AttentionModel,
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

model = get_model(model_version)        
model = model.to(device)               


criterion_action = nn.CrossEntropyLoss()  
criterion_status = nn.BCELoss()          
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.1)

