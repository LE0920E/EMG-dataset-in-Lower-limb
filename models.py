import torch
import torch.nn as nn


# LSTM-based model with shared representation for multi-task learning
class LSMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=200, num_actions=3, dropout_rate=0.5):
        super(LSMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=False, batch_first=True)
        self.shared_fc = nn.Linear(hidden_size, 50)
        self.action_classifier = nn.Linear(50, num_actions)
        self.status_classifier = nn.Linear(50, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        shared_rep = torch.relu(self.shared_fc(lstm_out))
        shared_rep = self.dropout(shared_rep)
        action_output = self.action_classifier(shared_rep)
        status_output = torch.sigmoid(self.status_classifier(shared_rep))
        return action_output, status_output


# Transformer-based model using encoder layers for sequence modeling
class TransformerModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=200, num_actions=3, num_layers=2, nhead=2):
        super(TransformerModel, self).__init__()
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

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        transformer_out = x[-1, :, :]
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
    

# CNN-LSTM model: uses 1D convolution to extract local features, followed by LSTM for temporal modeling
class CNN_LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_classes_action=3,
        num_classes_status=1,
        num_layers=2,
        num_filters=32,
        kernel_size=3
    ):
        super(CNN_LSTMModel, self).__init__()
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

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # Convert to (batch_size, input_size, seq_len)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_filters*2)

        # LSTM forward
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Use last time step

        # Shared and output layers
        shared_rep = torch.relu(self.shared_fc(last_hidden))
        action_output = self.action_classifier(shared_rep)
        status_output = self.sigmoid(self.status_classifier(shared_rep))

        return action_output, status_output