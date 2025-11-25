import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import time
from datetime import datetime
from ablation_dataset import create_ablation_datasets, dynamic_collate_fn, BATCH_SIZE,DEFAULT_PARAMETERS, AVAILABLE_PARAMETERS
from ablation_model import EMGAblationModel

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 训练参数
EPOCHS = 500
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_CLASSES_ACTION = 3  # gait, sitting, standing
NUM_CLASSES_STATUS = 2  # normal, abnormal
INPUT_DIM = len(DEFAULT_PARAMETERS)  # 根据DEFAULT_PARAMETERS动态设置输入维度

# 早停机制配置
EARLY_STOPPING_ENABLED = True  # 是否启用早停机制
EARLY_STOPPING_PATIENCE = 10   # 容忍轮数
EARLY_STOPPING_MIN_LR = 1e-6   # 最小学习率阈值

# 输出目录
MODELS_DIR = './models'
RESULTS_DIR = './models'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = EMGAblationModel(
        input_dim=INPUT_DIM,  # 4个肌肉通道
        hidden_size=200,  # Transformer隐藏层大小
        num_layers=2,
        num_classes_action=NUM_CLASSES_ACTION,
        num_classes_status=NUM_CLASSES_STATUS,
        nhead=2,  # 注意力头数
        dropout=0.3
    ).to(device)
scheduler_name = 'cosine'  # 修改此行来切换调度器
optimizer = optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )

schedulers = {
    'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.1),
    'step': optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
    'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
    'exponential': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2),
    'cyclic': optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000, mode='triangular'),
    'one_cycle': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=EPOCHS, steps_per_epoch=10),
    'linear_warmup': optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100),
    'multi_step': optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5),
    'lambda': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch),
    'none': None  # 不使用调度器
}
scheduler_set = schedulers[scheduler_name]
class AblationTrainer:
    def __init__(self, model, train_loader, test_loader, device='cuda', 
                 early_stopping_enabled=EARLY_STOPPING_ENABLED,
                 early_stopping_patience=EARLY_STOPPING_PATIENCE,
                 early_stopping_min_lr=EARLY_STOPPING_MIN_LR):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 早停机制配置
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_lr = early_stopping_min_lr
        
        # 优化器和损失函数
        self.optimizer = optimizer
        
        # Cosine学习率调度器
        self.scheduler = scheduler_set
        
        # 损失函数
        self.criterion_action = nn.CrossEntropyLoss()
        self.criterion_status = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.test_losses = []
        self.train_accs_action = []
        self.test_accs_action = []
        self.train_accs_status = []
        self.test_accs_status = []
        
        # 创建输出目录
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # 版本信息
        self.version = "v1"  # 固定版本号
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_action = 0
        correct_status = 0
        total_samples = 0
        
        for batch_idx, (emg_seq, seq_lengths, action_labels, status_labels) in enumerate(self.train_loader):
            emg_seq = emg_seq.to(self.device)
            action_labels = action_labels.to(self.device)
            status_labels = status_labels.to(self.device)
            seq_lengths = seq_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            action_output, status_output = self.model(emg_seq, seq_lengths)
            
            # 计算损失
            loss_action = self.criterion_action(action_output, action_labels)
            loss_status = self.criterion_status(status_output, status_labels)
            loss = loss_action + loss_status
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, action_pred = torch.max(action_output, 1)
            _, status_pred = torch.max(status_output, 1)
            correct_action += (action_pred == action_labels).sum().item()
            correct_status += (status_pred == status_labels).sum().item()
            total_samples += action_labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        acc_action = 100. * correct_action / total_samples
        acc_status = 100. * correct_status / total_samples
        
        return avg_loss, acc_action, acc_status
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct_action = 0
        correct_status = 0
        total_samples = 0
        
        all_action_preds = []
        all_action_labels = []
        all_status_preds = []
        all_status_labels = []
        
        with torch.no_grad():
            for emg_seq, seq_lengths, action_labels, status_labels in self.test_loader:
                emg_seq = emg_seq.to(self.device)
                action_labels = action_labels.to(self.device)
                status_labels = status_labels.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                
                # 前向传播
                action_output, status_output = self.model(emg_seq, seq_lengths)
                
                # 计算损失
                loss_action = self.criterion_action(action_output, action_labels)
                loss_status = self.criterion_status(status_output, status_labels)
                loss = loss_action + loss_status
                
                total_loss += loss.item()
                
                # 预测
                _, action_pred = torch.max(action_output, 1)
                _, status_pred = torch.max(status_output, 1)
                
                correct_action += (action_pred == action_labels).sum().item()
                correct_status += (status_pred == status_labels).sum().item()
                total_samples += action_labels.size(0)
                
                # 收集预测结果用于分析
                all_action_preds.extend(action_pred.cpu().numpy())
                all_action_labels.extend(action_labels.cpu().numpy())
                all_status_preds.extend(status_pred.cpu().numpy())
                all_status_labels.extend(status_labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        acc_action = 100. * correct_action / total_samples
        acc_status = 100. * correct_status / total_samples
        
        return avg_loss, acc_action, acc_status, all_action_preds, all_action_labels, all_status_preds, all_status_labels
    
    def train(self):
        """训练模型"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Weight decay: {WEIGHT_DECAY}")
        print(f"Early stopping enabled: {self.early_stopping_enabled}")
        if self.early_stopping_enabled:
            print(f"Early stopping patience: {self.early_stopping_patience}")
            print(f"Early stopping min LR: {self.early_stopping_min_lr}")
        
        # 显示模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        best_test_acc_action = 0
        best_test_acc_status = 0
        best_train_acc_action = 0
        best_train_acc_status = 0
        
        # 早停机制参数
        patience = self.early_stopping_patience if self.early_stopping_enabled else EPOCHS
        min_lr = self.early_stopping_min_lr if self.early_stopping_enabled else 0
        no_improve_count = 0  # 性能未提升计数
        best_test_loss = float('inf')  # 最佳测试损失
        best_avg_accuracy = 0  # 最佳平均准确率
        actual_epochs = EPOCHS  # 实际训练轮数
        
        # 训练历史记录
        training_history = []
        
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc_action, train_acc_status = self.train_epoch()
            
            # 评估
            test_loss, test_acc_action, test_acc_status, _, _, _, _ = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs_action.append(train_acc_action)
            self.test_accs_action.append(test_acc_action)
            self.train_accs_status.append(train_acc_status)
            self.test_accs_status.append(test_acc_status)
            
            # 记录训练过程
            epoch_time = time.time() - start_time
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc_action': train_acc_action,
                'test_acc_action': test_acc_action,
                'train_acc_status': train_acc_status,
                'test_acc_status': test_acc_status,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            })
            
            # 早停机制检查 - 基于损失和准确率综合判断
            if self.early_stopping_enabled:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # 检查是否达到双100%准确率
            if test_acc_action == 100.0 and test_acc_status == 100.0:
                print("🎯 达到双100%准确率！保存最佳模型")
                self.save_model('perfect_model.pth')
                best_test_acc_action = 100.0
                best_test_acc_status = 100.0
            
            # 打印进度
            print(f'Epoch {epoch+1}/{EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            print(f'  Train Acc Action: {train_acc_action:.2f}%, Test Acc Action: {test_acc_action:.2f}%')
            print(f'  Train Acc Status: {train_acc_status:.2f}%, Test Acc Status: {test_acc_status:.2f}%')
            print(f'  Learning Rate: {current_lr:.6f}, Time: {epoch_time:.2f}s')
            if self.early_stopping_enabled:
                print(f'  No improvement count: {no_improve_count}/{patience}')
            
            # 计算四个准确率的平均值（训练集动作、训练集状态、测试集动作、测试集状态）
            current_avg_accuracy = (train_acc_action + train_acc_status + test_acc_action + test_acc_status) / 4
            
            # 保存最佳模型（基于四个准确率的平均值）
            if current_avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = current_avg_accuracy
                best_test_acc_action = test_acc_action
                best_test_acc_status = test_acc_status
                best_train_acc_action = train_acc_action
                best_train_acc_status = train_acc_status
                self.save_model('best_model.pth')
                print(f'  New best model saved! (4-Acc Avg: {current_avg_accuracy:.2f}%, Train Action: {train_acc_action:.2f}%, Train Status: {train_acc_status:.2f}%, Test Action: {test_acc_action:.2f}%, Test Status: {test_acc_status:.2f}%)')
            
            # 单独记录最佳动作和状态准确率（仅用于显示，不保存模型）
            # 使用临时变量记录，避免覆盖基于平均准确率的最佳值
            temp_best_action = best_test_acc_action
            temp_best_status = best_test_acc_status
            
            if test_acc_action > temp_best_action:
                temp_best_action = test_acc_action
                print(f'  New best action accuracy: {test_acc_action:.2f}%')
            
            if test_acc_status > temp_best_status:
                temp_best_status = test_acc_status
                print(f'  New best status accuracy: {test_acc_status:.2f}%')
            
            print('-' * 60)
            
            # 检查早停条件：学习率过低且性能不再提升
            if self.early_stopping_enabled and current_lr < min_lr and no_improve_count >= patience:
                actual_epochs = epoch + 1
                print(f"🚨 Early stopping triggered at epoch {actual_epochs}")
                print(f"Learning rate {current_lr:.8f} < {min_lr} and no improvement for {no_improve_count} epochs")
                break
        
        # 保存训练历史
        self.save_training_history(training_history)
        
        # 保存准确率CSV
        self.save_accuracy_csv(best_train_acc_action, best_train_acc_status, best_test_acc_action, best_test_acc_status)
        
        print(f"Training completed! Actual training epochs: {actual_epochs}")
        print(f"Best Action Accuracy: {best_test_acc_action:.2f}%")
        print(f"Best Status Accuracy: {best_test_acc_status:.2f}%")
        
        # 显示最终模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Final Model Parameters: {total_params:,}")
        print(f"Final Trainable Parameters: {trainable_params:,}")
    
    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(MODELS_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, model_path)
        
        # 单独保存优化器状态
        optimizer_path = os.path.join(MODELS_DIR, 'optimizer_state.pth')
        torch.save(self.optimizer.state_dict(), optimizer_path)
    
    def save_training_history(self, history):
        """保存训练历史到CSV"""
        df = pd.DataFrame(history)
        csv_path = os.path.join(RESULTS_DIR, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        print(f"Training history saved to: {csv_path}")
    
    def save_accuracy_csv(self, train_acc_action, train_acc_status, test_acc_action, test_acc_status):
        """保存准确率到CSV"""
        accuracy_data = {
            'version': [self.version],
            'train_action_accuracy': [train_acc_action],
            'train_status_accuracy': [train_acc_status],
            'test_action_accuracy': [test_acc_action],
            'test_status_accuracy': [test_acc_status],
            'average_accuracy': [(train_acc_action + train_acc_status + test_acc_action + test_acc_status) / 4]
        }
        df = pd.DataFrame(accuracy_data)
        csv_path = os.path.join(RESULTS_DIR, 'Accuracy.csv')
        df.to_csv(csv_path, index=False)
        print(f"Accuracy data saved to: {csv_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        # 训练损失曲线
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.test_losses, label='Test Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'train_loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 动作分类准确率曲线
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_accs_action, label='Train Accuracy', color='blue')
        plt.plot(self.test_accs_action, label='Test Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Action Classification Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'train_action_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 状态分类准确率曲线
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_accs_status, label='Train Accuracy', color='blue')
        plt.plot(self.test_accs_status, label='Test Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Status Classification Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'train_status_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 学习率曲线
        plt.figure(figsize=(8, 6))
        lr_values = [h['learning_rate'] for h in self._get_training_history()]
        plt.plot(lr_values, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training curves saved as separate images")
    
    def plot_validation_curves(self):
        """绘制验证集曲线"""
        # 验证集损失曲线
        plt.figure(figsize=(8, 6))
        plt.plot(self.test_losses, label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, 'validation_loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 验证集动作分类准确率曲线
        plt.figure(figsize=(8, 6))
        plt.plot(self.test_accs_action, label='Validation Action Accuracy', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Action Classification Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, 'validation_action_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 验证集状态分类准确率曲线
        plt.figure(figsize=(8, 6))
        plt.plot(self.test_accs_status, label='Validation Status Accuracy', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Status Classification Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, 'validation_status_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 综合验证集曲线（损失和准确率在同一图中）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 损失曲线
        ax1.plot(self.test_losses, label='Validation Loss', color='red', linewidth=2)
        ax1.set_ylabel('Loss')
        ax1.set_title('Validation Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.test_accs_action, label='Action Accuracy', color='blue', linewidth=2)
        ax2.plot(self.test_accs_status, label='Status Accuracy', color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'validation_comprehensive_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Validation curves saved as separate images")
    
    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        # 获取最终预测结果
        _, _, _, action_preds, action_labels, status_preds, status_labels = self.evaluate()
        
        # 动作分类混淆矩阵
        plt.figure(figsize=(8, 6))
        cm_action = confusion_matrix(action_labels, action_preds)
        sns.heatmap(cm_action, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Gait', 'Sitting', 'Standing'],
                   yticklabels=['Gait', 'Sitting', 'Standing'])
        plt.title('Action Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_action.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 状态分类混淆矩阵
        plt.figure(figsize=(8, 6))
        cm_status = confusion_matrix(status_labels, status_preds)
        sns.heatmap(cm_status, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'])
        plt.title('Status Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_status.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Confusion matrices saved as separate images")
    
    def plot_roc_curves(self):
        """绘制ROC曲线"""
        # 获取模型预测概率
        self.model.eval()
        all_action_probs = []
        all_action_labels = []
        all_status_probs = []
        all_status_labels = []
        
        with torch.no_grad():
            for emg_seq, seq_lengths, action_labels, status_labels in self.test_loader:
                emg_seq = emg_seq.to(self.device)
                action_labels = action_labels.to(self.device)
                status_labels = status_labels.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                
                # 前向传播获取概率
                action_output, status_output = self.model(emg_seq, seq_lengths)
                action_probs = torch.softmax(action_output, dim=1)
                status_probs = torch.softmax(status_output, dim=1)
                
                all_action_probs.extend(action_probs.cpu().numpy())
                all_action_labels.extend(action_labels.cpu().numpy())
                all_status_probs.extend(status_probs.cpu().numpy())
                all_status_labels.extend(status_labels.cpu().numpy())
        
        # 转换为numpy数组
        action_probs = np.array(all_action_probs)
        action_labels = np.array(all_action_labels)
        status_probs = np.array(all_status_probs)
        status_labels = np.array(all_status_labels)
        
        # 绘制动作分类ROC曲线（多分类）
        plt.figure(figsize=(10, 8))
        
        # 动作分类：多类别ROC曲线
        action_classes = ['Gait', 'Sitting', 'Standing']
        action_labels_bin = label_binarize(action_labels, classes=[0, 1, 2])
        
        # 计算每个类别的ROC曲线和AUC
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(len(action_classes)):
            fpr[i], tpr[i], _ = roc_curve(action_labels_bin[:, i], action_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制每个类别的ROC曲线
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(len(action_classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{action_classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Action Classification ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_action.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制状态分类ROC曲线（二分类）
        plt.figure(figsize=(10, 8))
        
        # 状态分类：二分类ROC曲线
        status_classes = ['Normal', 'Abnormal']
        
        # 计算ROC曲线和AUC
        fpr_status, tpr_status, _ = roc_curve(status_labels, status_probs[:, 1])
        roc_auc_status = auc(fpr_status, tpr_status)
        
        plt.plot(fpr_status, tpr_status, color='darkorange', lw=2,
                label=f'Status Classification (AUC = {roc_auc_status:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Status Classification ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_status.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制综合ROC曲线（动作和状态在同一图中）
        plt.figure(figsize=(12, 10))
        
        # 动作分类曲线
        for i, color in zip(range(len(action_classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Action {action_classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        # 状态分类曲线
        plt.plot(fpr_status, tpr_status, color='darkorange', lw=2,
                label=f'Status Classification (AUC = {roc_auc_status:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Comprehensive ROC Curves - Action and Status Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存AUC数据到CSV
        auc_data = {
            'class_type': ['Action_Gait', 'Action_Sitting', 'Action_Standing', 'Status_Classification'],
            'auc_score': [roc_auc[0], roc_auc[1], roc_auc[2], roc_auc_status]
        }
        df_auc = pd.DataFrame(auc_data)
        df_auc.to_csv(os.path.join(RESULTS_DIR, 'auc_scores.csv'), index=False)
        
        print("ROC curves and AUC scores saved successfully!")
        print(f"Action Gait AUC: {roc_auc[0]:.3f}")
        print(f"Action Sitting AUC: {roc_auc[1]:.3f}")
        print(f"Action Standing AUC: {roc_auc[2]:.3f}")
        print(f"Status Classification AUC: {roc_auc_status:.3f}")
    
    def generate_error_analysis(self):
        """生成错误类型分析"""
        _, _, _, action_preds, action_labels, status_preds, status_labels = self.evaluate()
        
        # 动作分类错误分析
        action_errors = []
        for i, (pred, true) in enumerate(zip(action_preds, action_labels)):
            if pred != true:
                action_errors.append({
                    'sample_id': i,
                    'predicted': pred,
                    'actual': true,
                    'error_type': f'{self._get_action_name(true)} -> {self._get_action_name(pred)}'
                })
        
        # 状态分类错误分析
        status_errors = []
        for i, (pred, true) in enumerate(zip(status_preds, status_labels)):
            if pred != true:
                status_errors.append({
                    'sample_id': i,
                    'predicted': pred,
                    'actual': true,
                    'error_type': f'{self._get_status_name(true)} -> {self._get_status_name(pred)}'
                })
        
        # 保存错误分析
        if action_errors:
            df_action_errors = pd.DataFrame(action_errors)
            df_action_errors.to_csv(os.path.join(RESULTS_DIR, 'action_errors.csv'), index=False)
        
        if status_errors:
            df_status_errors = pd.DataFrame(status_errors)
            df_status_errors.to_csv(os.path.join(RESULTS_DIR, 'status_errors.csv'), index=False)
        
        # 生成分类报告
        action_report = classification_report(action_labels, action_preds, 
                                             target_names=['Gait', 'Sitting', 'Standing'],
                                             output_dict=True)
        status_report = classification_report(status_labels, status_preds,
                                            target_names=['Normal', 'Abnormal'],
                                            output_dict=True)
        
        # 保存分类报告
        df_action_report = pd.DataFrame(action_report).transpose()
        df_status_report = pd.DataFrame(status_report).transpose()
        
        df_action_report.to_csv(os.path.join(RESULTS_DIR, 'action_classification_report.csv'))
        df_status_report.to_csv(os.path.join(RESULTS_DIR, 'status_classification_report.csv'))
        
        print(f"Error analysis saved to: {RESULTS_DIR}")
    
    def _get_action_name(self, label):
        """获取动作名称"""
        action_names = {0: 'Gait', 1: 'Sitting', 2: 'Standing'}
        return action_names.get(label, 'Unknown')
    
    def _get_status_name(self, label):
        """获取状态名称"""
        status_names = {0: 'Normal', 1: 'Abnormal'}
        return status_names.get(label, 'Unknown')
    
    def _get_training_history(self):
        """获取训练历史"""
        # 从CSV文件读取训练历史
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('training_history') and f.endswith('.csv')]
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(RESULTS_DIR, x)))
            df = pd.read_csv(os.path.join(RESULTS_DIR, latest_csv))
            return df.to_dict('records')
        return []
    
    def generate_final_report(self):
        """生成最终报告"""
        # 加载最佳模型进行评估
        best_model_path = os.path.join(MODELS_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model for final evaluation")
        
        # 获取最终评估结果
        test_loss, test_acc_action, test_acc_status, action_preds, action_labels, status_preds, status_labels = self.evaluate()
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 动态获取实际使用的参数信息
        selected_parameters = self.train_loader.dataset.selected_parameters
        parameter_names = [AVAILABLE_PARAMETERS[param] for param in selected_parameters]
        
        # 统计肌电信号和关节角度参数
        emg_signals = [name for param, name in zip(selected_parameters, parameter_names) if param != 'flexo_extension']
        joint_angles = [name for param, name in zip(selected_parameters, parameter_names) if param == 'flexo_extension']
        
        # 构建输入描述
        input_description = []
        if emg_signals:
            input_description.append(f"{len(emg_signals)} muscle EMG signals ({', '.join(emg_signals)})")
        if joint_angles:
            input_description.append(f"{len(joint_angles)} joint angle ({', '.join(joint_angles)})")
        
        input_description_str = " + ".join(input_description)
        
        # 创建报告文本
        report = f"""
=== Ablation Experiment Final Report ===
Version: {self.version}

Model Architecture:
- Input: {input_description_str}
- Output: Action classification (3 classes) + Status classification (2 classes)
- Sequence length: 1000 (1 second window)
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}

Training Configuration:
- Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})
- Loss function: CrossEntropyLoss
- Scheduler: CosineAnnealingLR (T_max={EPOCHS})
- Epochs: {EPOCHS}
- Batch size: {BATCH_SIZE}

Final Results:
- Test Loss: {test_loss:.4f}
- Action Classification Accuracy: {test_acc_action:.2f}%
- Status Classification Accuracy: {test_acc_status:.2f}%

Dataset Information:
- Training samples: {len(self.train_loader.dataset)}
- Test samples: {len(self.test_loader.dataset)}
- Action classes: Gait, Sitting, Standing
- Status classes: Normal, Abnormal
- Selected parameters: {', '.join(parameter_names)}

Generated Files:
- Training curves: train_loss_curve.png, train_action_accuracy_curve.png, train_status_accuracy_curve.png, learning_rate_curve.png
- Validation curves: validation_loss_curve.png, validation_action_accuracy_curve.png, validation_status_accuracy_curve.png, validation_comprehensive_curve.png
- ROC curves: roc_curve_action.png, roc_curve_status.png, roc_curve_comprehensive.png
- Confusion matrices: confusion_matrix_action.png, confusion_matrix_status.png
- Training history: training_history.csv
- Error analysis: action_errors.csv, status_errors.csv
- Classification reports: action_classification_report.csv, status_classification_report.csv
- AUC scores: auc_scores.csv
- Accuracy data: Accuracy.csv

=== End of Report ===
"""
        
        # 保存报告
        report_path = os.path.join(RESULTS_DIR, 'final_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Final report saved to: {report_path}")
        print(report)


def main():
    """主函数"""
    print("=== EMG Ablation Experiment Training ===")
    
    # 检查设备
    
    
    # 创建数据集
    print("Creating datasets...")
    train_dataset, test_dataset = create_ablation_datasets()
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        collate_fn=dynamic_collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        collate_fn=dynamic_collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 创建模型
    print("Creating model...")
    
    
    # 计算并显示模型参数详细信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # 创建训练器
    trainer = AblationTrainer(model, train_loader, test_loader, device)
    
    # 训练模型
    trainer.train()
    
    # 生成可视化结果
    print("Generating visualizations...")
    trainer.plot_training_curves()
    trainer.plot_validation_curves()
    trainer.plot_roc_curves()
    trainer.plot_confusion_matrices()
    trainer.generate_error_analysis()
    trainer.generate_final_report()
    
    print("Training completed successfully!")
    print(f"All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()