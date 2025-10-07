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
from factory_model import model, criterion_action, criterion_status, optimizer, scheduler, device, model_version, train_epoches_num, learning_rate_history
from dataset import train_loader, test_loader, ACTION_LABELS, STATUS_LABELS

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelTrainerWithVisualization:
    def __init__(self, model, train_loader, test_loader, model_version):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_version = model_version
        self.output_dir = f"./models_v2/{model_version}"
        
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
        
        for batch_idx, (X, y_action, y_status) in enumerate(self.train_loader):
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
            for X, y_action, y_status in self.test_loader:
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
            for X, y_action, y_status in self.test_loader:
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
        """绘制训练曲线 - 七张图分开显示"""
        # 1. 训练损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='训练损失', color='blue', linewidth=2)
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/train_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 验证损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.val_loss_history, label='验证损失', color='red', linewidth=2)
        plt.title('验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/val_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 训练动作准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_acc_action_history, label='训练动作准确率', color='green', linewidth=2)
        plt.title('训练动作准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/train_action_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 验证动作准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.val_acc_action_history, label='验证动作准确率', color='orange', linewidth=2)
        plt.title('验证动作准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/val_action_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 训练状态准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_acc_status_history, label='训练状态准确率', color='cyan', linewidth=2)
        plt.title('训练状态准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/train_status_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 验证状态准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.val_acc_status_history, label='验证状态准确率', color='magenta', linewidth=2)
        plt.title('验证状态准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/val_status_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 学习率曲线（单独保存）
        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_rate_history)
        plt.title('学习率变化曲线')
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
        plt.title(f'{task_name}混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(f'{self.output_dir}/confusion_matrix_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, true_labels, prob_scores, task_name):
        """绘制ROC曲线"""
        # 对于多分类问题，只对状态分类（二分类）绘制ROC曲线
        if task_name == '状态':
            fpr, tpr, _ = roc_curve(true_labels, prob_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率')
            plt.ylabel('真正率')
            plt.title(f'{task_name} ROC曲线')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f'{self.output_dir}/roc_curve_{task_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return roc_auc
        else:
            # 对于动作分类（多分类），跳过ROC曲线绘制
            print(f"跳过{task_name}的ROC曲线绘制（多分类问题）")
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
            plt.title('PCA特征可视化')
            plt.xlabel('主成分1')
            plt.ylabel('主成分2')
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
            plt.title('注意力权重热图')
            plt.xlabel('时间步')
            plt.ylabel('样本')
            plt.savefig(f'{self.output_dir}/attention_weights.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_error_analysis(self, true_action, pred_action, true_status, pred_status):
        """绘制错误分析图"""
        # 计算错误类型
        action_correct = (np.array(true_action) == np.array(pred_action))
        status_correct = (np.array(true_status) == np.array(pred_status))
        
        # 错误类型统计
        error_types = {
            '动作正确-状态错误': np.sum(action_correct & ~status_correct),
            '动作错误-状态正确': np.sum(~action_correct & status_correct),
            '两者都错误': np.sum(~action_correct & ~status_correct),
            '两者都正确': np.sum(action_correct & status_correct)
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(error_types.keys(), error_types.values())
        plt.title('错误类型分析')
        plt.ylabel('样本数量')
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
        plt.scatter(total_params, final_acc_action, label='动作准确率', s=100)
        plt.scatter(total_params, final_acc_status, label='状态准确率', s=100)
        plt.xlabel('模型参数数量')
        plt.ylabel('准确率')
        plt.title('模型复杂度与性能关系')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/complexity_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_accuracy_to_csv(self, final_acc_action, final_acc_status):
        """保存准确率到CSV文件"""
        csv_file = f"./models_v2/Accuracy_{self.model_version}.csv"
        
        # 计算平均准确率
        avg_accuracy = (final_acc_action + final_acc_status) / 2
        
        # 创建DataFrame
        data = {
            'Version': [self.model_version],
            'Action accuracy': [final_acc_action],
            'Status accuracy': [final_acc_status],
            'average accuracy': [avg_accuracy]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"准确率已保存到: {csv_file}")

    def train(self, num_epochs):
        """完整的训练流程"""
        print(f"开始训练模型 {self.model_version}...")
        
        best_val_acc = 0.0
        best_model_state = None
        total_time = 0.0
        epoch_times = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc_action, train_acc_status = self.train_epoch()
            
            # 验证
            val_loss, val_acc_action, val_acc_status = self.validate_epoch()
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rate_history.append(current_lr)
            learning_rate_history.append(current_lr)  # 更新全局记录
            
            # 更新学习率调度器
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
            
            # 保存最佳模型
            current_val_acc = (val_acc_action + val_acc_status) / 2
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, f'{self.output_dir}/best_model.pth')
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            total_time += epoch_time
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')
                print(f'Action Acc - Train: {train_acc_action:.4f}, Val: {val_acc_action:.4f}')
                print(f'Status Acc - Train: {train_acc_status:.4f}, Val: {val_acc_status:.4f}')
        
        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # 收集最终预测结果
        predictions = self.collect_predictions()
        
        # 生成所有可视化图表
        self.generate_all_visualizations(predictions)
        
        # 保存准确率
        self.save_accuracy_to_csv(self.val_acc_action_history[-1], self.val_acc_status_history[-1])
        
        avg_time = total_time / num_epochs
        print(f"训练完成！总时间: {total_time:.2f}s, 平均每轮时间: {avg_time:.2f}s")
        print(f"模型和可视化结果保存在: {self.output_dir}")
        
        return total_time, avg_time

    def generate_all_visualizations(self, predictions):
        """生成所有可视化图表"""
        true_action, pred_action, true_status, pred_status, prob_status, features = predictions
        
        # 1. 训练曲线
        self.plot_training_curves()
        
        # 2. 混淆矩阵
        self.plot_confusion_matrix(true_action, pred_action, 'action')
        self.plot_confusion_matrix(true_status, pred_status, 'status')
        
        # 3. ROC曲线
        roc_auc_action = self.plot_roc_curve(true_action, 
                                            [p[pred_action[i]] for i, p in enumerate(
                                                torch.softmax(torch.tensor(features), dim=1).numpy())], 
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
def main():
    # 创建训练器
    trainer = ModelTrainerWithVisualization(model, train_loader, test_loader, model_version)
    
    # 开始训练并获取时间统计
    total_time, avg_time = trainer.train(train_epoches_num)
    
    # 打印详细的时间统计
    print("\n" + "="*50)
    print("训练时间统计:")
    print(f"总训练轮次: {train_epoches_num}")
    print(f"总训练时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"平均每轮时间: {avg_time:.2f} 秒")
    print("="*50)

if __name__ == "__main__":
    main()