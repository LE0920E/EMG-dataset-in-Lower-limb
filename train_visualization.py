import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os
import time
from factory_model import output_dir,model, criterion_action, criterion_status, optimizer, scheduler, device, model_version, train_epoches_num, learning_rate_history
from dataset import train_loader, test_loader, ACTION_LABELS, STATUS_LABELS

# Set matplotlib font for English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelTrainerWithVisualization:
    def __init__(self, model, train_loader, test_loader, model_version):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_version = model_version
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练历史记录
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_action_history = []
        self.train_acc_status_history = []
        self.val_acc_action_history = []
        self.val_acc_status_history = []
        self.learning_rate_history = []
        
        # 预测结果存储
        self.all_predictions = []
        self.all_true_labels = []
        self.all_features = []
        self.attention_weights_history = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct_action = 0
        correct_status = 0
        total_samples = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 动态批处理返回 (padded_sequences, lengths, action_labels, status_labels)
            if len(batch_data) == 4:
                X, lengths, y_action, y_status = batch_data
            else:
                X, y_action, y_status = batch_data
            X = X.to(device)
            y_action = y_action.to(device)
            y_status = y_status.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(X)
            
            # 计算损失
            loss1 = criterion_action(outputs[0], y_action.long())
            loss2 = criterion_status(outputs[1].squeeze(), y_status.float())
            loss = loss1 + loss2
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted_action = torch.max(outputs[0], 1)
            predicted_status = (outputs[1].squeeze() > 0.5).float()
            
            correct_action += (predicted_action == y_action).sum().item()
            correct_status += (predicted_status == y_status).sum().item()
            total_samples += y_action.size(0)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc_action = correct_action / total_samples
        epoch_acc_status = correct_status / total_samples
        
        return epoch_loss, epoch_acc_action, epoch_acc_status

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        val_loss = 0.0
        correct_action = 0
        correct_status = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                # 动态批处理返回 (padded_sequences, lengths, action_labels, status_labels)
                if len(batch_data) == 4:
                    X, lengths, y_action, y_status = batch_data
                else:
                    X, y_action, y_status = batch_data
                X = X.to(device)
                y_action = y_action.to(device)
                y_status = y_status.to(device)
                
                outputs = self.model(X)
                
                loss1 = criterion_action(outputs[0], y_action.long())
                loss2 = criterion_status(outputs[1].squeeze(), y_status.float())
                loss = loss1 + loss2
                
                val_loss += loss.item()
                
                _, predicted_action = torch.max(outputs[0], 1)
                predicted_status = (outputs[1].squeeze() > 0.5).float()
                
                correct_action += (predicted_action == y_action).sum().item()
                correct_status += (predicted_status == y_status).sum().item()
                total_samples += y_action.size(0)
        
        val_loss /= len(self.test_loader)
        val_acc_action = correct_action / total_samples
        val_acc_status = correct_status / total_samples
        
        return val_loss, val_acc_action, val_acc_status

    def collect_predictions(self):
        """收集预测结果用于可视化"""
        self.model.eval()
        all_true_action = []
        all_pred_action = []
        all_true_status = []
        all_pred_status = []
        all_prob_status = []
        all_features = []
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                # 动态批处理返回 (padded_sequences, lengths, action_labels, status_labels)
                if len(batch_data) == 4:
                    X, lengths, y_action, y_status = batch_data
                else:
                    X, y_action, y_status = batch_data
                X = X.to(device)
                outputs = self.model(X)
                
                # 动作预测
                _, predicted_action = torch.max(outputs[0], 1)
                all_true_action.extend(y_action.cpu().numpy())
                all_pred_action.extend(predicted_action.cpu().numpy())
                
                # 状态预测
                prob_status = outputs[1].squeeze().cpu().numpy()
                predicted_status = (prob_status > 0.5).astype(int)
                all_true_status.extend(y_status.cpu().numpy())
                all_pred_status.extend(predicted_status)
                all_prob_status.extend(prob_status)
                
                # 特征提取（使用最后一个隐藏层）
                if hasattr(self.model, 'shared_fc'):
                    # 对于有共享层的模型
                    if hasattr(self.model, 'lstm'):
                        # 检查是否是CNN_LSTMModel
                        if hasattr(self.model, 'conv1'):
                            # CNN_LSTMModel的处理
                            x_conv = X.permute(0, 2, 1)  # Convert to (batch_size, input_size, seq_len)
                            x_conv = self.model.relu(self.model.conv1(x_conv))
                            x_conv = self.model.relu(self.model.conv2(x_conv))
                            x_conv = x_conv.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_filters*2)
                            lstm_out, _ = self.model.lstm(x_conv)
                            features = self.model.shared_fc(lstm_out[:, -1, :])
                        else:
                            # 普通LSTM模型的处理
                            lstm_out, _ = self.model.lstm(X)
                            features = self.model.shared_fc(lstm_out[:, -1, :])
                    elif hasattr(self.model, 'transformer_encoder'):
                        x_proj = self.model.input_projection(X)
                        x_trans = x_proj.permute(1, 0, 2)
                        transformer_out = self.model.transformer_encoder(x_trans)
                        features = self.model.shared_fc(transformer_out[-1, :, :])
                    else:
                        features = torch.zeros(X.size(0), 50)  # 默认特征维度
                else:
                    features = torch.zeros(X.size(0), 50)
                
                all_features.extend(features.cpu().numpy())
        
        return (all_true_action, all_pred_action, 
                all_true_status, all_pred_status, all_prob_status,
                all_features)

    def plot_training_curves(self):
        """绘制训练曲线 - 七张图分开显示，使用实际训练轮数"""
        actual_epochs = len(self.train_loss_history)
        epochs_range = range(1, actual_epochs + 1)
        
        # 1. 训练损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_loss_history, label='Training Loss', color='blue', linewidth=2)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/train_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 验证损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.val_loss_history, label='Validation Loss', color='red', linewidth=2)
        plt.title('Validation Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/val_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 训练动作准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_acc_action_history, label='Training Action Accuracy', color='green', linewidth=2)
        plt.title('Training Action Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/train_action_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 验证动作准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.val_acc_action_history, label='Validation Action Accuracy', color='orange', linewidth=2)
        plt.title('Validation Action Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/val_action_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 训练状态准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.train_acc_status_history, label='Training Status Accuracy', color='cyan', linewidth=2)
        plt.title('Training Status Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/train_status_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 验证状态准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.val_acc_status_history, label='Validation Status Accuracy', color='magenta', linewidth=2)
        plt.title('Validation Status Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/val_status_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. 学习率曲线（单独保存）
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.learning_rate_history)
        plt.title('Learning Rate Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/learning_rate_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, true_labels, pred_labels, task_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{task_name.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.output_dir}/confusion_matrix_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, true_labels, prob_scores, task_name):
        """绘制ROC曲线"""
        # 对于多分类问题，只对状态分类（二分类）绘制ROC曲线
        if task_name == 'status':
            fpr, tpr, _ = roc_curve(true_labels, prob_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{task_name.capitalize()} ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/roc_curve_{task_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return roc_auc
        else:
            # 对于动作分类（多分类），跳过ROC曲线绘制
            print(f"Skipping ROC curve for {task_name} (multi-class problem)")
            return 0.0

    def plot_feature_importance(self, features, labels):
        """绘制特征重要性图"""
        # 使用PCA进行特征降维可视化
        if len(features) > 1:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
            plt.title('PCA Feature Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(f'{self.output_dir}/feature_importance_pca.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_attention_weights(self):
        """绘制注意力权重图（如果模型有注意力机制）"""
        if hasattr(self.model, 'attention') and len(self.attention_weights_history) > 0:
            # 取最后一个batch的注意力权重
            attention_weights = self.attention_weights_history[-1]
            
            plt.figure(figsize=(12, 6))
            plt.imshow(attention_weights.cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('Attention Weights Heatmap')
            plt.xlabel('Time Steps')
            plt.ylabel('Samples')
            plt.savefig(f'{self.output_dir}/attention_weights.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_error_analysis(self, true_action, pred_action, true_status, pred_status):
        """绘制错误分析图"""
        # 计算错误类型
        action_correct = (np.array(true_action) == np.array(pred_action))
        status_correct = (np.array(true_status) == np.array(pred_status))
        
        # 错误类型统计
        error_types = {
            'Action Correct-Status Wrong': np.sum(action_correct & ~status_correct),
            'Action Wrong-Status Correct': np.sum(~action_correct & status_correct),
            'Both Wrong': np.sum(~action_correct & ~status_correct),
            'Both Correct': np.sum(action_correct & status_correct)
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(error_types.keys(), error_types.values())
        plt.title('Error Type Analysis')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_complexity_vs_performance(self):
        """绘制模型复杂度与性能关系图"""
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 获取最终性能
        final_acc_action = self.val_acc_action_history[-1] if self.val_acc_action_history else 0
        final_acc_status = self.val_acc_status_history[-1] if self.val_acc_status_history else 0
        
        # 这里可以扩展为比较多个模型，目前只绘制当前模型
        plt.figure(figsize=(8, 6))
        plt.scatter(total_params, final_acc_action, label='Action Accuracy', s=100)
        plt.scatter(total_params, final_acc_status, label='Status Accuracy', s=100)
        plt.xlabel('Number of Model Parameters')
        plt.ylabel('Accuracy')
        plt.title('Model Complexity vs Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/complexity_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_accuracy_to_csv(self, train_acc_action, train_acc_status, test_acc_action, test_acc_status):
        """保存准确率到CSV文件"""
        csv_file = f"{self.output_dir}/Accuracy.csv"
        
        # 计算平均准确率（四个准确率的平均值）
        avg_accuracy = (train_acc_action + train_acc_status + test_acc_action + test_acc_status) / 4
        
        # 创建DataFrame
        data = {
            'Version': [self.model_version],
            'train_action_accuracy': [train_acc_action],
            'train_status_accuracy': [train_acc_status],
            'test_action_accuracy': [test_acc_action],
            'test_status_accuracy': [test_acc_status],
            'average_accuracy': [avg_accuracy]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"Accuracy saved to: {csv_file}")
        
        # 保存训练历史数据用于恢复训练曲线
        self.save_training_history()
    
    def save_training_history(self):
        """保存训练历史数据到CSV文件"""
        history_file = f"{self.output_dir}/training_history.csv"
        
        # 创建训练历史DataFrame
        history_data = {
            'epoch': list(range(1, len(self.train_loss_history) + 1)),
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'train_acc_action': self.train_acc_action_history,
            'train_acc_status': self.train_acc_status_history,
            'val_acc_action': self.val_acc_action_history,
            'val_acc_status': self.val_acc_status_history,
            'learning_rate': self.learning_rate_history
        }
        
        df_history = pd.DataFrame(history_data)
        df_history.to_csv(history_file, index=False)
        print(f"Training history saved to: {history_file}")
    
    def load_training_history(self):
        """从CSV文件加载训练历史数据"""
        history_file = f"{self.output_dir}/training_history.csv"
        
        if os.path.exists(history_file):
            df_history = pd.read_csv(history_file)
            
            # 恢复训练历史数据
            self.train_loss_history = df_history['train_loss'].tolist()
            self.val_loss_history = df_history['val_loss'].tolist()
            self.train_acc_action_history = df_history['train_acc_action'].tolist()
            self.train_acc_status_history = df_history['train_acc_status'].tolist()
            self.val_acc_action_history = df_history['val_acc_action'].tolist()
            self.val_acc_status_history = df_history['val_acc_status'].tolist()
            self.learning_rate_history = df_history['learning_rate'].tolist()
            
            print(f"Training history loaded from: {history_file}")
            print(f"Loaded {len(self.train_loss_history)} epochs of training history")
            return True
        
        return False

    def check_existing_results(self):
        """检查是否已存在训练结果"""
        model_path = f'{self.output_dir}/best_model.pth'
        accuracy_path = f'{self.output_dir}/Accuracy.csv'
        
        # 检查模型文件和准确率文件是否存在
        if os.path.exists(model_path) and os.path.exists(accuracy_path):
            print(f"Found existing training results, loading model and accuracy data...")
            
            # 加载模型
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            
            # 读取准确率数据
            df = pd.read_csv(accuracy_path)
            if not df.empty:
                # 检查CSV格式（新格式包含训练集准确率）
                if 'train_action_accuracy' in df.columns:
                    # 新格式：包含训练集准确率
                    train_acc_action = df['train_action_accuracy'].iloc[0]
                    train_acc_status = df['train_status_accuracy'].iloc[0]
                    test_acc_action = df['test_action_accuracy'].iloc[0]
                    test_acc_status = df['test_status_accuracy'].iloc[0]
                    avg_accuracy = df['average_accuracy'].iloc[0]
                    
                    print(f"Loaded accuracy data (new format):")
                    print(f"  Train Action accuracy: {train_acc_action:.4f}")
                    print(f"  Train Status accuracy: {train_acc_status:.4f}")
                    print(f"  Test Action accuracy: {test_acc_action:.4f}")
                    print(f"  Test Status accuracy: {test_acc_status:.4f}")
                    print(f"  Average accuracy: {avg_accuracy:.4f}")
                    
                    # 设置最终准确率用于后续可视化（使用测试集准确率）
                    self.final_acc_action = test_acc_action
                    self.final_acc_status = test_acc_status
                else:
                    # 旧格式：只包含测试集准确率
                    test_acc_action = df['Action accuracy'].iloc[0]
                    test_acc_status = df['Status accuracy'].iloc[0]
                    avg_accuracy = df['average accuracy'].iloc[0]
                    
                    print(f"Loaded accuracy data (old format):")
                    print(f"  Action accuracy: {test_acc_action:.4f}")
                    print(f"  Status accuracy: {test_acc_status:.4f}")
                    print(f"  Average accuracy: {avg_accuracy:.4f}")
                    
                    # 设置最终准确率用于后续可视化
                    self.final_acc_action = test_acc_action
                    self.final_acc_status = test_acc_status
                
                # 尝试加载训练历史数据
                if self.load_training_history():
                    print("Training curves can be regenerated from saved history data")
                else:
                    print("No training history data found, training curves will not be available")
                
                return True
        
        return False

    def train(self, num_epochs, resume_training=False):
        """完整的训练流程，包含早停机制和继续训练功能"""
        
        # 检查是否已存在训练结果
        if self.check_existing_results():
            if not resume_training:
                print(f"Model {self.model_version} already has training results, skipping training process")
                
                # 如果加载了训练历史数据，重新生成训练曲线
                if hasattr(self, 'train_loss_history') and len(self.train_loss_history) > 0:
                    print("Regenerating training curves from saved history...")
                    predictions = self.collect_predictions()
                    self.generate_all_visualizations(predictions)
                
                # 返回默认的时间统计值（因为跳过了训练过程）
                return 0.0, 0.0, len(self.train_loss_history)
            else:
                 print(f"Model {self.model_version} found, resuming training from epoch {len(self.train_loss_history) + 1}")
                 
                 # 恢复优化器状态
                 optimizer_path = f'{self.output_dir}/optimizer_state.pth'
                 if os.path.exists(optimizer_path):
                     optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                     print("Optimizer state loaded successfully")
                 
                 # 简化调度器处理：直接使用默认配置继续训练
                 if scheduler is not None:
                     print("Using default scheduler configuration for continued training")
                     # 不需要重置调度器，直接使用当前配置继续训练
                     # 调度器会自动从当前状态继续
        else:
            # 如果没有现有结果，初始化训练历史
            self.train_loss_history = []
            self.val_loss_history = []
            self.train_acc_action_history = []
            self.train_acc_status_history = []
            self.val_acc_action_history = []
            self.val_acc_status_history = []
            self.learning_rate_history = []
            
            print(f"Starting new training for model {self.model_version}...")
        
        print(f"Training model {self.model_version}...")
        
        # 初始化训练状态
        start_epoch = len(self.train_loss_history)
        best_val_acc = 0.0
        best_model_state = None
        total_time = 0.0
        epoch_times = []
        
        # 初始化最佳准确率记录
        best_train_acc_action = 0.0
        best_train_acc_status = 0.0
        
        # 早停机制参数
        patience = 10  # 容忍轮数
        min_lr = 1e-6  # 最小学习率阈值
        no_improve_count = 0  # 性能未提升计数
        best_avg_acc = 0.0  # 最佳四个准确率平均值
        actual_epochs = num_epochs  # 实际训练轮数
        
        # 如果继续训练，恢复最佳验证准确率
        if resume_training and len(self.val_acc_action_history) > 0:
            best_avg_acc = (self.train_acc_action_history[-1] + self.train_acc_status_history[-1] + 
                           self.val_acc_action_history[-1] + self.val_acc_status_history[-1]) / 4
            print(f"Resuming from best 4-accuracy average: {best_avg_acc:.4f}")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc_action, train_acc_status = self.train_epoch()
            
            # 验证
            val_loss, val_acc_action, val_acc_status = self.validate_epoch()
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rate_history.append(current_lr)
            learning_rate_history.append(current_lr)  # 更新全局记录
            
            # 更新学习率调度器（如果存在）
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 记录历史
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_action_history.append(train_acc_action)
            self.train_acc_status_history.append(train_acc_status)
            self.val_acc_action_history.append(val_acc_action)
            self.val_acc_status_history.append(val_acc_status)
            
            # 保存最佳模型（基于四个准确率的平均值：训练集动作、训练集状态、测试集动作、测试集状态）
            current_avg_acc = (train_acc_action + train_acc_status + val_acc_action + val_acc_status) / 4
            if current_avg_acc > best_avg_acc:
                best_avg_acc = current_avg_acc
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, f'{self.output_dir}/best_model.pth')
                
                # 记录训练集最佳准确率
                best_train_acc_action = train_acc_action
                best_train_acc_status = train_acc_status
                
                print(f'  New best model saved! (4-Acc Avg: {current_avg_acc:.4f}, Train Action: {train_acc_action:.4f}, Train Status: {train_acc_status:.4f}, Val Action: {val_acc_action:.4f}, Val Status: {val_acc_status:.4f})')
                no_improve_count = 0  # 重置未提升计数
            else:
                no_improve_count += 1
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            total_time += epoch_time
            
            # 打印进度
            current_epoch = epoch + 1
            total_epochs = start_epoch + num_epochs
            if current_epoch % 10 == 0 or current_epoch == start_epoch + 1:
                print(f'Epoch {current_epoch}/{total_epochs}, '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'LR: {current_lr:.8f}, Time: {epoch_time:.2f}s')
                print(f'Action Acc - Train: {train_acc_action:.4f}, Val: {val_acc_action:.4f}')
                print(f'Status Acc - Train: {train_acc_status:.4f}, Val: {val_acc_status:.4f}')
                print(f'No improvement count: {no_improve_count}/{patience}')
            
            # 检查早停条件：学习率过低且性能不再提升
            if current_lr < min_lr and no_improve_count >= patience:
                actual_epochs = epoch + 1
                print(f"🚨 Early stopping triggered at epoch {actual_epochs}")
                print(f"Learning rate {current_lr:.8f} < {min_lr} and no improvement for {no_improve_count} epochs")
                break
        
        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # 保存优化器状态
        torch.save(optimizer.state_dict(), f'{self.output_dir}/optimizer_state.pth')
        
        # 注意：不保存调度器状态，因为每次继续训练都会重置调度器
        
        # 收集最终预测结果
        predictions = self.collect_predictions()
        
        # 生成所有可视化图表 - 使用实际训练轮数
        self.generate_all_visualizations(predictions)
        
        # 保存准确率
        self.save_accuracy_to_csv(best_train_acc_action, best_train_acc_status, self.val_acc_action_history[-1], self.val_acc_status_history[-1])
        
        avg_time = total_time / actual_epochs
        print(f"Training completed! Actual training epochs: {actual_epochs}, Total time: {total_time:.2f}s, Average epoch time: {avg_time:.2f}s")
        print(f"Model and visualization results saved to: {self.output_dir}")
        
        return total_time, avg_time, actual_epochs

    def generate_all_visualizations(self, predictions):
        """生成所有可视化图表"""
        true_action, pred_action, true_status, pred_status, prob_status, features = predictions
        
        # 1. 训练曲线
        self.plot_training_curves()
        
        # 2. 混淆矩阵
        self.plot_confusion_matrix(true_action, pred_action, 'action')
        self.plot_confusion_matrix(true_status, pred_status, 'status')
        
        # 3. ROC曲线
        # 修复警告：先将features转换为numpy数组，再转换为tensor
        features_array = np.array(features)
        roc_auc_action = self.plot_roc_curve(true_action, 
                                            [p[pred_action[i]] for i, p in enumerate(
                                                torch.softmax(torch.tensor(features_array), dim=1).numpy())], 
                                            'action')
        roc_auc_status = self.plot_roc_curve(true_status, prob_status, 'status')
        
        # 4. 特征重要性
        self.plot_feature_importance(features, true_action)
        
        # 5. 注意力权重（如果适用）
        self.plot_attention_weights()
        
        # 6. 错误分析
        self.plot_error_analysis(true_action, pred_action, true_status, pred_status)
        
        # 7. 模型复杂度与性能
        self.plot_model_complexity_vs_performance()
        
        print(f"ROC AUC - Action: {roc_auc_action:.4f}, Status: {roc_auc_status:.4f}")

# 主训练函数
def main(resume_training=False):
    # 创建训练器
    trainer = ModelTrainerWithVisualization(model, train_loader, test_loader, model_version)
    
    # 开始训练并获取时间统计
    total_time, avg_time, actual_epochs = trainer.train(train_epoches_num, resume_training)
    
    # 打印详细的时间统计
    print("\n" + "="*50)
    print("Training Time Statistics:")
    print(f"Planned training epochs: {train_epoches_num}")
    print(f"Actual training epochs: {actual_epochs}")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average epoch time: {avg_time:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main(resume_training=True)