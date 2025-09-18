from factory_model import model,criterion_action,criterion_status,optimizer,scheduler,device,model_version,test_epoches_num
from dataset import full_loader,test_loader,ACTION_LABELS, STATUS_LABELS
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import accuracy_score


import numpy as np
def test_model(model, data_loader, criterion_action, criterion_status):
    model.eval()
    all_y_action_true = []
    all_y_action_pred = []
    all_y_status_true = []
    all_y_status_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for X, y_action, y_status in data_loader:
            X = X.to(device)
            y_action = y_action.to(device)
            y_status = y_status.to(device)
            outputs = model(X)

            action_preds = torch.argmax(outputs[0], dim=1).cpu().numpy()
            status_preds = (outputs[1].squeeze() > 0.5).float().cpu().numpy()

            all_y_action_true.extend(y_action.cpu().numpy())
            all_y_action_pred.extend(action_preds)
            all_y_status_true.extend(y_status.cpu().numpy())
            all_y_status_pred.extend(status_preds)

            loss1 = criterion_action(outputs[0], y_action.long())
            loss2 = criterion_status(outputs[1].squeeze(), y_status.float())
            loss = loss1 + loss2
            total_loss += loss.item()

    acc_action = accuracy_score(all_y_action_true, all_y_action_pred)
    acc_status = accuracy_score(all_y_status_true, all_y_status_pred)

    avg_val_loss = total_loss / len(data_loader)

    print(f"Test Accuracy Action: {acc_action}, Test Accuracy Status: {acc_status}, Val Loss: {avg_val_loss:.4f}")

    return acc_action, acc_status,avg_val_loss

# 初始化空列表存储结果
loss_history = []
test_accuracies_action = []
test_accuracies_status = []
test_accuracies_average=[]

# 设置训练轮数
num_epochs = test_epoches_num
MODEL_VERSION = model_version

# 初始化最佳准确率和保存路径
best_acc = 0.0
best_acc_action=0.0
best_acc_status=0.0
model_save_path = r"./models/best_model.pth"
model_save_path = model_save_path.replace(".pth", f"_{MODEL_VERSION}.pth")

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
else:
    print("未找到模型文件，将使用新结构初始化模型")

start_time = time.time()

# 模型训练并记录损失
for epoch in range(num_epochs):
    start_time_epoch=time.time()

    # 测试模型并记录准确率
    acc_action, acc_status, val_loss = test_model(model, full_loader, criterion_action, criterion_status)
    test_accuracies_action.append(acc_action)
    test_accuracies_status.append(acc_status)

    # 打印当前准确率
    end_time_epoch=time.time()
    elapsed_time_epoch=end_time_epoch-start_time_epoch
    print(f"\n--- Epoch {epoch+1} ---")
    print(f"Action Accuracy: {acc_action:.4f}, Status Accuracy: {acc_status:.4f}")
    current_acc = (acc_action + acc_status) / 2  # 可以根据需要调整权重
    test_accuracies_average.append(current_acc)
    print(f"Average accuracy: {current_acc:.4f}")
    print(f"Running time: {elapsed_time_epoch:.2f} s")
    
    if current_acc > best_acc:
        best_acc_action = acc_action
        best_acc_status = acc_status
        best_acc = current_acc
        

end_time = time.time()

test_accuracies_action = np.array(test_accuracies_action)
test_accuracies_status = np.array(test_accuracies_status)
test_accuracies_average=np.array(test_accuracies_average)

average_action=np.mean(test_accuracies_action)
average_status=np.mean(test_accuracies_status)
average_general=np.mean(test_accuracies_average)

elapsed_time = end_time - start_time
print(f"\n---------Training summary----------")
print(f"Total number of training epochs:{num_epochs}")
print(f"Best test accuracy (Action): {best_acc_action*100:.4f}%")
print(f"Best test accuracy (Status): {best_acc_status*100:.4f}%")
print(f"Best average accuracy: {best_acc*100:.4f}%")
print(f"Average test accuracy (Action): {average_action*100:.4f}%")
print(f"Average test accuracy (Status): {average_status*100:.4f}%")
print(f"Average accuracy: {average_general*100:.4f}%")
print(f"Total Running time {elapsed_time:.2f} s")
print(f"Average Running time {elapsed_time/num_epochs:.2f} s per epoch")






