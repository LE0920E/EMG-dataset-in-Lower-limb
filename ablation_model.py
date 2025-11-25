import torch
import torch.nn as nn

class EMGAblationModel(nn.Module):
    def __init__(self, input_dim=4, hidden_size=200, num_layers=2, 
                 num_classes_action=3, num_classes_status=2, nhead=2, dropout=0.3):
        """
        EMG消融实验模型，使用Transformer架构
        支持变长序列输入
        
        Args:
            input_dim: 输入特征维度（默认4个肌肉通道）
            hidden_size: Transformer隐藏层大小
            num_layers: Transformer层数
            num_classes_action: 动作分类类别数（3类：gait, sitting, standing）
            num_classes_status: 状态分类类别数（2类：normal, abnormal）
            nhead: 注意力头数
            dropout: Dropout率
        """
        super(EMGAblationModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            activation='relu',
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 共享特征层
        self.shared_fc = nn.Linear(hidden_size, 50)
        
        # 动作分类器
        self.action_classifier = nn.Linear(50, num_classes_action)
        
        # 状态分类器
        self.status_classifier = nn.Linear(50, num_classes_status)
        
    def forward(self, x, lengths=None):
        """
        前向传播，支持变长序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            lengths: 序列实际长度（可选）
        
        Returns:
            action_output: 动作分类输出 [batch_size, num_classes_action]
            status_output: 状态分类输出 [batch_size, num_classes_status]
        """
        # 输入形状: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        
        # 创建注意力掩码（可选，用于变长序列）
        if lengths is not None:
            # 创建注意力掩码，屏蔽填充部分
            max_len = x.size(0)
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        else:
            mask = None
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 使用最后一个有效时间步
        if lengths is not None:
            # 获取每个序列的最后一个有效时间步
            idx = (lengths - 1).view(1, -1, 1).expand(1, len(lengths), x.size(2))
            transformer_out = x.gather(0, idx).squeeze(0)
        else:
            transformer_out = x[-1, :, :]  # 使用最后一个时间步
        
        # 共享特征提取
        shared_rep = torch.relu(self.shared_fc(transformer_out))
        shared_rep = self.dropout(shared_rep)
        
        # 动作分类
        action_output = self.action_classifier(shared_rep)
        
        # 状态分类
        status_output = torch.sigmoid(self.status_classifier(shared_rep))
        
        return action_output, status_output
