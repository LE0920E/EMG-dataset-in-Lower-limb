import torch
import torch.nn as nn



class LSMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=200, num_actions=3,dropout_rate=0.5):
        super(LSMModel, self).__init__()
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
    
class TransformerModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=200, num_actions=3, num_layers=2, nhead=2):
        super(TransformerModel, self).__init__()
        
        # 线性层将输入映射到 Transformer 的 hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            activation='relu'
        )
        
        # Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 共享全连接层
        self.shared_fc = nn.Linear(hidden_size, 50)
        
        # 动作分类分支
        self.action_classifier = nn.Linear(50, num_actions)
        
        # 状态分类分支
        self.status_classifier = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # 投影到 hidden_size
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)

        # Transformer 要求 shape 为 (seq_len, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, hidden_size)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_size)
        
        # 取最后一个时间步的输出
        transformer_out = x[-1, :, :]  # (batch_size, hidden_size)

        # 共享层前向传播
        shared_rep = torch.relu(self.shared_fc(transformer_out))
        
        # 分支输出
        action_output = self.action_classifier(shared_rep)
        status_output = torch.sigmoid(self.status_classifier(shared_rep))
        
        return action_output, status_output
    

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.weight_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, lstm_output):
        attention_scores = torch.matmul(lstm_output, self.weight_vector)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sum, attention_weights


class LSTM_AttentionModel(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_classes_action=3,
        num_classes_status=1,
        num_layers=2
    ):
        super(LSTM_AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = TemporalAttention(hidden_size)

        self.classifier_action = nn.Linear(hidden_size, num_classes_action)
        self.classifier_status = nn.Linear(hidden_size, num_classes_status)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        attended_feature, attn_weights = self.attention(lstm_out)

        output_action = self.classifier_action(attended_feature)
        output_status = self.sigmoid(self.classifier_status(attended_feature))

        return output_action, output_status, attn_weights




