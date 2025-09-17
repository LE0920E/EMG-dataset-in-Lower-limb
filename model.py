import torch
import torch.nn as nn
import torch.optim as optim

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
print(device)

model_version="v1.1.1"

class MultiTaskModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=200, num_actions=3,dropout_rate=0.5):
        super(MultiTaskModel, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=False,batch_first=True)
        
        # 共享全连接层
        self.shared_fc = nn.Linear(hidden_size, 50)
        
        # 动作分类分支
        self.action_classifier = nn.Linear(50, num_actions)
        
        # 状态分类分支
        self.status_classifier = nn.Linear(50, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 获取最后一个时间步的输出
        
        # 共享层前向传播
        shared_rep = torch.relu(self.shared_fc(lstm_out))
        shared_rep = self.dropout(shared_rep)
        
        # 分支输出
        action_output = self.action_classifier(shared_rep)
        status_output = torch.sigmoid(self.status_classifier(shared_rep))
        
        return action_output, status_output

# 模型实例化
model = MultiTaskModel().to(device)

# 定义损失函数和优化器
criterion_action = nn.CrossEntropyLoss()  # 对于动作分类
criterion_status = nn.BCELoss()  # 对于二分类问题（正常/异常）
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.1)
