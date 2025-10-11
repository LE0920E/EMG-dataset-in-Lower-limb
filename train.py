from factory_model import output_dir,model,criterion_action,criterion_status,optimizer,scheduler,model_version,train_epoches_num
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\n=== 开始训练 ===")
from dataset import train_loader,test_loader,ACTION_LABELS, STATUS_LABELS
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import time
import pandas as pd



def train_model(model, data_loader, criterion_action, criterion_status, optimizer, num_epochs=50):
    model.train()
    loss_history = []  # 记录每个 epoch 的平均损失
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_data in data_loader:
            # 动态批处理返回 (padded_sequences, lengths, action_labels, status_labels)
            if len(batch_data) == 4:
                X, lengths, y_action, y_status = batch_data
            else:
                X, y_action, y_status = batch_data
                lengths = None
            
            X = X.to(device)
            y_action = y_action.to(device)
            y_status = y_status.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            # 根据是否有长度信息调用模型
            if lengths is not None:
                outputs = model(X, lengths)
            else:
                outputs = model(X)
            

            
            loss1 = criterion_action(outputs[0], y_action.long())
            loss2 = criterion_status(outputs[1].squeeze(), y_status.float())
            
            loss = loss1 + loss2  # 可根据需要调整两个任务的权重
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_epoch_loss = running_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")
    
    return loss_history

def validate_model(model, data_loader, criterion_action, criterion_status):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_data in data_loader:
            # 动态批处理返回 (padded_sequences, lengths, action_labels, status_labels)
            if len(batch_data) == 4:
                X, lengths, y_action, y_status = batch_data
            else:
                X, y_action, y_status = batch_data
                lengths = None
            
            X = X.to(device)
            y_action = y_action.to(device)
            y_status = y_status.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            
            # 根据是否有长度信息调用模型
            if lengths is not None:
                outputs = model(X, lengths)
            else:
                outputs = model(X)

            loss1 = criterion_action(outputs[0], y_action.long())
            loss2 = criterion_status(outputs[1].squeeze(), y_status.float())
            loss = loss1 + loss2

            val_loss += loss.item()

    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss

from sklearn.metrics import accuracy_score

def test_model(model, data_loader):
    model.eval()
    correct_action = 0
    correct_status = 0
    total = 0

    with torch.no_grad():
        for batch_data in data_loader:
            # 动态批处理返回 (padded_sequences, lengths, action_labels, status_labels)
            if len(batch_data) == 4:
                X, lengths, y_action, y_status = batch_data
            else:
                X, y_action, y_status = batch_data
                lengths = None
            
            X = X.to(device)
            y_action = y_action.to(device)
            y_status = y_status.to(device)
            if lengths is not None:
                lengths = lengths.to(device)
            
            # 根据是否有长度信息调用模型
            if lengths is not None:
                outputs = model(X, lengths)
            else:
                outputs = model(X)

            _, predicted_action = torch.max(outputs[0].data, 1)
            predicted_status = (outputs[1].squeeze() > 0.5).float()

            correct_action += (predicted_action == y_action).sum().item()
            correct_status += (predicted_status == y_status).sum().item()
            total += y_action.size(0)

    accuracy_action = correct_action / total
    accuracy_status = correct_status / total
    average_accuracy = (accuracy_action + accuracy_status) / 2

    return accuracy_action, accuracy_status, average_accuracy

def plot_training_results(loss_history, test_accuracies_action, test_accuracies_status, num_epochs):
    epochs = range(1, num_epochs + 1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制训练损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制测试准确率
    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_accuracies_action, 'g-', label='Action Accuracy')
    plt.title('Test Accuracy per Epoch(action)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies_status, 'r-', label='Status Accuracy')
    plt.title('Test Accuracy per Epoch(station)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像到output_dir目录
    image_path = os.path.join(output_dir, f'training_results_{model_version}.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像，避免内存泄漏
    print(f"训练结果图像已保存到: {image_path}")

def get_accuracy_by_version(csv_file, version):
    """
    从CSV文件中读取指定版本的三项准确率，并返回三个变量。

    :param csv_file: str, CSV文件路径
    :param version: str, 要查询的版本号，如 'v1.1.1'
    :return: tuple (action_acc, status_acc, average_acc)，如果未找到则返回 (None, None, None)
    """
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        return 0,0,0

    try:
        # 读取CSV
        df = pd.read_csv(csv_file)
        
        # 去除列名前后空格
        df.columns = df.columns.str.strip()
        
        # 查找对应版本
        result = df[df['Version'] == version]
        
        if result.empty:
            return 0,0,0
        
        # 提取第一行数据（假设版本唯一）
        action_acc = float(result['Action accuracy'].iloc[0])
        status_acc = float(result['Status accuracy'].iloc[0])
        average_acc = float(result['average accuracy'].iloc[0])
        
        return action_acc, status_acc, average_acc

    except Exception as e:
        print(f" 读取或解析文件时出错: {e}")
        return None, None, None

def update_accuracy_by_version(csv_file, version, action_acc=None, status_acc=None, average_acc=None):
    """
    更新CSV文件中指定版本的准确率数据。如果版本不存在，可选择添加新行。

    :param csv_file: str, CSV文件路径
    :param version: str, 要更新的版本号，如 'v1.1.1'
    :param action_acc: float, 动作准确率（可选）
    :param status_acc: float, 状态准确率（可选）
    :param average_acc: float, 平均准确率（可选）
    :return: bool, 是否更新成功
    """
    # 检查参数
    if all(v is None for v in [action_acc, status_acc, average_acc]):
        return False

    # 检查文件是否存在，若不存在则创建带表头的空DataFrame
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
    else:
        df = pd.DataFrame(columns=['Version', 'Action accuracy', 'Status accuracy', 'average accuracy'])

    # 检查是否已有该版本
    if version in df['Version'].values:
        row_idx = df[df['Version'] == version].index[0]

        if action_acc is not None:
            df.loc[row_idx, 'Action accuracy'] = action_acc
        if status_acc is not None:
            df.loc[row_idx, 'Status accuracy'] = status_acc
        if average_acc is not None:
            df.loc[row_idx, 'average accuracy'] = average_acc
    else:
        new_row = {
            'Version': version,
            'Action accuracy': action_acc if action_acc is not None else 0.0,
            'Status accuracy': status_acc if status_acc is not None else 0.0,
            'average accuracy': average_acc if average_acc is not None else 0.0
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 保存回CSV
    try:
        df.to_csv(csv_file, index=False)
        print(f" 数据已成功保存到 '{csv_file}'")
        return True
    except Exception as e:
        print(f" 保存文件失败: {e}")
        return False

# 初始化空列表存储结果
loss_history = []
test_accuracies_action = []
test_accuracies_status = []

# 设置训练轮数
num_epochs = train_epoches_num
MODEL_VERSION = model_version
os.makedirs(output_dir, exist_ok=True)
accuracy_file=f"{output_dir}/Accuracy.csv"
# 初始化最佳准确率和保存路径
best_acc = 0.0
best_acc_action=0.0
best_acc_status=0.0
model_save_path = f"{output_dir}/best_model.pth"
best_acc_action,best_acc_status,best_acc=get_accuracy_by_version(accuracy_file,MODEL_VERSION)

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f"Best model saved with average accuracy: {best_acc:.2f}")
else:
    print("未找到模型文件，将使用新结构初始化模型")

start_time = time.time()
is_update=0

# 早停机制参数
patience = 10  # 容忍轮数
min_lr = 1e-6  # 最小学习率阈值
no_improve_count = 0  # 性能未提升计数
best_val_loss = float('inf')  # 最佳验证损失

# 模型训练并记录损失
for epoch in range(num_epochs):
    start_time_epoch=time.time()
    
    loss_per_epoch = train_model(model, train_loader, criterion_action, criterion_status, optimizer, num_epochs=1)
    loss_history.extend(loss_per_epoch)  # 每个 epoch 只训练一次

    # 验证模型并记录损失
    val_loss = validate_model(model, test_loader, criterion_action, criterion_status)
    
    # 测试模型并记录准确率
    acc_action, acc_status, avg_acc = test_model(model, test_loader)
    test_accuracies_action.append(acc_action)
    test_accuracies_status.append(acc_status)

    # 打印当前准确率
    end_time_epoch=time.time()
    elapsed_time_epoch=end_time_epoch-start_time_epoch
    print(f"\n--- Epoch {epoch+1} ---")
    print(f"Action Accuracy: {acc_action:.4f}, Status Accuracy: {acc_status:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Running time: {elapsed_time_epoch:.2f} s")

    # 更新学习率
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr:.8f}")

    # 判断是否为最佳模型
    current_acc = (acc_action + acc_status) / 2  # 可以根据需要调整权重
    if current_acc > best_acc:
        is_update+=1
        best_acc_action = acc_action
        best_acc_status = acc_status
        best_acc = current_acc
        torch.save(model.state_dict(),model_save_path)
        print(f"✅ Best model saved with average accuracy: {best_acc:.2f}")
    
    # 早停机制检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    # 检查早停条件：学习率过低且性能不再提升
    if current_lr < min_lr and no_improve_count >= patience:
        print(f"🚨 Early stopping triggered at epoch {epoch+1}")
        print(f"Learning rate {current_lr:.8f} < {min_lr} and no improvement for {no_improve_count} epochs")
        break

end_time = time.time()
elapsed_time = end_time - start_time
actual_epochs = epoch + 1  # 实际训练的轮数

if(is_update>0):
    update_accuracy_by_version(accuracy_file,MODEL_VERSION,best_acc_action,best_acc_status,best_acc)
print(f"\n---------Training summary----------")
print(f"Total number of training epochs: {actual_epochs} (early stopped at epoch {actual_epochs})")
print(f"Best test accuracy (Action): {best_acc_action*100:.4f}%")
print(f"Best test accuracy (Status): {best_acc_status*100:.4f}%")
print(f"Best model saved with average accuracy: {best_acc*100:.4f}%")
print(f"Total Running time {elapsed_time:.2f} s")
print(f"Average Running time {elapsed_time/actual_epochs:.2f} s per epoch")
print(f"Update number: {is_update}")
print(f"Model saved to: {model_save_path}")

# 可视化结果 - 使用实际训练轮数
plot_training_results(loss_history, test_accuracies_action, test_accuracies_status, num_epochs=actual_epochs)
