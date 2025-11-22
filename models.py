import torch
import torch.nn as nn


# LSTM-based model with shared representation for multi-task learning
class LSMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=200, num_actions=3, dropout_rate=0.5):
        """
        LSTM模型，支持5通道输入（4个肌电通道+1个膝盖屈伸角度通道）
        支持变长序列输入
        
        Args:
            input_size: 输入特征维度（默认5）
            hidden_size: LSTM隐藏层大小
            num_actions: 动作类别数
            dropout_rate: Dropout率
        """
        super(LSMModel, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=False, batch_first=True)
        self.shared_fc = nn.Linear(hidden_size, 50)
        self.action_classifier = nn.Linear(50, num_actions)
        self.status_classifier = nn.Linear(50, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths=None):
        """
        前向传播，支持变长序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            lengths: 序列实际长度（可选）
        """
        # 输入形状: [batch_size, seq_len, input_size]
        if lengths is not None:
            # 确保所有长度都大于0
            lengths = torch.clamp(lengths, min=1)
            
            # 按序列长度降序排列，以便pack_padded_sequence正常工作
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            
            # 打包序列
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM处理
            lstm_out_packed, _ = self.lstm(x_packed)
            
            # 解包序列
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True, total_length=x.size(1)
            )
            
            # 恢复原始顺序
            _, unsort_idx = sort_idx.sort()
            lstm_out = lstm_out[unsort_idx]
            
            # 获取每个序列的最后一个有效时间步
            batch_size = x.size(0)
            seq_lengths = lengths - 1  # 转换为0-based索引
            batch_indices = torch.arange(batch_size)
            lstm_out = lstm_out[batch_indices, seq_lengths]
        else:
            # 固定长度序列处理
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]  # 使用最后一个时间步
        
        shared_rep = torch.relu(self.shared_fc(lstm_out))
        shared_rep = self.dropout(shared_rep)
        action_output = self.action_classifier(shared_rep)
        status_output = torch.sigmoid(self.status_classifier(shared_rep))
        return action_output, status_output


# Transformer-based model using encoder layers for sequence modeling
class TransformerModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=200, num_actions=3, num_layers=2, nhead=2):
        """
        Transformer模型，支持5通道输入
        支持变长序列输入
        
        Args:
            input_size: 输入特征维度（默认5）
            hidden_size: Transformer隐藏层大小
            num_actions: 动作类别数
            num_layers: Transformer层数
            nhead: 注意力头数
        """
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shared_fc = nn.Linear(hidden_size, 50)
        self.action_classifier = nn.Linear(50, num_actions)
        self.status_classifier = nn.Linear(50, 1)

    def forward(self, x, lengths=None):
        """
        前向传播，支持变长序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            lengths: 序列实际长度（可选）
        """
        # 输入形状: [batch_size, seq_len, input_size]
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        
        # 创建注意力掩码（可选，用于变长序列）
        if lengths is not None:
            # 创建注意力掩码，屏蔽填充部分
            max_len = x.size(0)
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        else:
            mask = None
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 使用最后一个有效时间步
        if lengths is not None:
            # 获取每个序列的最后一个有效时间步
            idx = (lengths - 1).view(1, -1, 1).expand(1, len(lengths), x.size(2))
            transformer_out = x.gather(0, idx).squeeze(0)
        else:
            transformer_out = x[-1, :, :]  # 使用最后一个时间步
        
        shared_rep = torch.relu(self.shared_fc(transformer_out))
        action_output = self.action_classifier(shared_rep)
        status_output = torch.sigmoid(self.status_classifier(shared_rep))
        return action_output, status_output


# Temporal attention mechanism to weight important time steps in sequence
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.weight_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, lstm_output):
        attention_scores = torch.matmul(lstm_output, self.weight_vector)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sum, attention_weights


# LSTM with temporal attention for enhanced feature focusing
class LSTM_AttentionModel(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=200,
        num_classes_action=3,
        num_classes_status=1,
        num_layers=2
    ):
        """
        带注意力机制的LSTM模型，支持5通道输入
        支持变长序列输入
        
        Args:
            input_size: 输入特征维度（默认5）
            hidden_size: LSTM隐藏层大小
            num_classes_action: 动作类别数
            num_classes_status: 状态类别数
            num_layers: LSTM层数
        """
        super(LSTM_AttentionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        self.classifier_action = nn.Linear(hidden_size, num_classes_action)
        self.classifier_status = nn.Linear(hidden_size, num_classes_status)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        """
        前向传播，支持变长序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            lengths: 序列实际长度（可选）
        """
        # 输入形状: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        if lengths is not None:
            # 使用pack_padded_sequence处理变长序列
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            # 固定长度序列处理
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            lstm_out, _ = self.lstm(x, (h0, c0))
        
        attended_feature, attn_weights = self.attention(lstm_out)
        output_action = self.classifier_action(attended_feature)
        output_status = self.sigmoid(self.classifier_status(attended_feature))
        return output_action, output_status, attn_weights
    

# CNN-LSTM model: uses 1D convolution to extract local features, followed by LSTM for temporal modeling
class CNN_LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=200,
        num_classes_action=3,
        num_classes_status=1,
        num_layers=2,
        num_filters=32,
        kernel_size=3
    ):
        """
        CNN-LSTM模型，支持5通道输入
        支持变长序列输入
        
        Args:
            input_size: 输入特征维度（默认5）
            hidden_size: LSTM隐藏层大小
            num_classes_action: 动作类别数
            num_classes_status: 状态类别数
            num_layers: LSTM层数
            num_filters: 卷积滤波器数量
            kernel_size: 卷积核大小
        """
        super(CNN_LSTMModel, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # LSTM layer
        self.lstm = nn.LSTM(num_filters*2, hidden_size, num_layers, batch_first=True)

        # Fully connected layers for classification
        self.shared_fc = nn.Linear(hidden_size, 50)
        self.action_classifier = nn.Linear(50, num_classes_action)
        self.status_classifier = nn.Linear(50, num_classes_status)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        """
        前向传播，支持变长序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            lengths: 序列实际长度（可选）
        """
        # 输入形状: [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, input_size, seq_len]
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 转回 [batch_size, seq_len, num_filters*2]

        # LSTM forward
        if lengths is not None:
            # 使用pack_padded_sequence处理变长序列
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
            # 获取每个序列的最后一个有效时间步
            idx = (lengths - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
            idx = idx.unsqueeze(1)
            last_hidden = lstm_out.gather(1, idx).squeeze(1)
        else:
            # 固定长度序列处理
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # 使用最后一个时间步

        # Shared and output layers
        shared_rep = torch.relu(self.shared_fc(last_hidden))
        action_output = self.action_classifier(shared_rep)
        status_output = self.sigmoid(self.status_classifier(shared_rep))

        return action_output, status_output