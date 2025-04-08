# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
# import matplotlib.pyplot as plt
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# # --------------------------
# # 新增残差块模块
# # --------------------------
# class ResidualBlock(nn.Module):
#     """残差块结构"""
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# # --------------------------
# # 改进的CNN模型架构（含残差块）
# # --------------------------
# class DynamicCNN(nn.Module):
#     """支持残差连接和深度特征提取的CNN"""
#
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = input_dim
#
#         # 特征预处理层
#         self.preprocess = nn.Sequential(
#             nn.Linear(input_dim, 8),  # 扩展到8维特征
#             nn.BatchNorm1d(8),
#             nn.ReLU()
#         )
#
#         # 残差卷积模块
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(1, 8, kernel_size=3, padding=1),  # 初始卷积
#             ResidualBlock(8, 8),  # 残差块1
#             nn.MaxPool1d(kernel_size=2, stride=1),  # 保持特征维度
#             ResidualBlock(8, 16),  # 残差块2
#             nn.AdaptiveAvgPool1d(4)  # 自适应池化
#         )
#
#         # 动态计算全连接层输入维度
#         with torch.no_grad():
#             dummy = torch.randn(1, 1, 8)  # 预处理后的特征维度
#             dummy_out = self.conv_layers(dummy)
#             self.fc_input = dummy_out.view(-1).shape[0]
#
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(self.fc_input, 32),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         x = self.preprocess(x)  # (batch, input_dim) -> (batch, 8)
#         x = x.unsqueeze(1)  # 增加通道维度 (batch, 1, 8)
#         features = self.conv_layers(x)
#         features = features.view(features.size(0), -1)
#         return self.classifier(features).squeeze(1)
#
#
# # --------------------------
# # 类别平衡处理（修改训练函数）
# # --------------------------
# def create_weighted_sampler(y):
#     """创建加权采样器"""
#     class_counts = np.bincount(y.astype(int))
#     class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
#     sample_weights = class_weights[y]
#     return WeightedRandomSampler(sample_weights, len(y))
# # --------------------------
# # 早停机制类
# # --------------------------
#
# class EarlyStopper:
#     def __init__(self, patience=5, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_score = None
#
#     def __call__(self, score):
#         if self.best_score is None:
#             self.best_score = score
#         elif score < self.best_score + self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         else:
#             self.best_score = score
#             self.counter = 0
#         return False
# # --------------------------
# # 修改后的训练评估模块
# # --------------------------
# def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs=30):
#     """支持类别平衡的训练循环"""
#     early_stopper = EarlyStopper(patience=5, min_delta=0.001)
#     best_auc = 0
#     history = {'train_loss': [], 'val_auc': [], 'val_f1': [],
#                'val_accuracy': [], 'val_precision': []}
#
#     for epoch in range(epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         # 验证阶段
#         model.eval()
#         y_true, y_pred_proba = [], []
#         with torch.no_grad():
#             for X, y in test_loader:
#                 X, y = X.to(device), y.to(device)
#                 outputs = model(X)
#                 y_true.extend(y.cpu().numpy())
#                 y_pred_proba.extend(torch.sigmoid(outputs).cpu().numpy())
#
#         # 计算指标
#         auc = roc_auc_score(y_true, y_pred_proba)
#         y_pred = np.round(y_pred_proba)
#         f1 = f1_score(y_true, y_pred)
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred)
#
#         # 记录历史
#         history['train_loss'].append(train_loss / len(train_loader))
#         history['val_auc'].append(auc)
#         history['val_f1'].append(f1)
#         history['val_accuracy'].append(accuracy)
#         history['val_precision'].append(precision)
#
#         # 早停检查
#         if early_stopper(auc):
#             print(f"Early stopping at epoch {epoch + 1}")
#             break
#
#         # 保存最佳模型
#         if auc > best_auc:
#             best_auc = auc
#             torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
#
#         print(f"Epoch {epoch + 1}/{epochs}")
#         print(f"Train Loss: {train_loss / len(train_loader):.4f}")
#         print(f"Validation AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
#
#     return model, history
#
#
# # --------------------------
# # 修改后的交叉验证模块
# # --------------------------
# def cross_validate(model_cls, X, y, n_splits=5, epochs=30):
#     """支持类别平衡的交叉验证"""
#     skf = StratifiedKFold(n_splits=n_splits)
#     fold_metrics = {'auc': [], 'f1': [], 'accuracy': [], 'precision': []}
#
#     for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#         print(f"\nFold {fold + 1}/{n_splits}")
#
#         # 数据分割
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#
#         # 计算类别权重
#         pos = sum(y_train)
#         neg = len(y_train) - pos
#         pos_weight = torch.tensor([neg / pos if pos != 0 else 1.0]).to(device)
#         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
#         # 创建带平衡采样的DataLoader
#         sampler = create_weighted_sampler(y_train)
#         train_loader = DataLoader(
#             TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
#             batch_size=64, sampler=sampler
#         )
#         test_loader = DataLoader(
#             TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
#             batch_size=64
#         )
#
#         # 模型初始化
#         model = model_cls(input_dim=X_train.shape[1]).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
#
#         # 训练验证
#         model, hist = train_and_evaluate(
#             model, train_loader, test_loader, optimizer,
#             criterion, epochs=epochs
#         )
#
#         # 记录最佳指标
#         fold_metrics['auc'].append(max(hist['val_auc']))
#         fold_metrics['f1'].append(max(hist['val_f1']))
#         fold_metrics['accuracy'].append(max(hist['val_accuracy']))
#         fold_metrics['precision'].append(max(hist['val_precision']))
#
#     print("\nCross Validation Results:")
#     print(f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
#     print(f"F1: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
#     print(f"Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
#     print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
#
# def main():
#     # 加载数据
#     df1 = pd.read_csv('./data/cleaned_jigan.csv')
#     df2 = pd.read_csv('./data/cleaned_labs_first_day.csv')
#     target = 'match_flag'
#
#     # 交叉验证第一个模型
#     print("Cross Validating Model 1...")
#     X1 = df1.drop(columns=[target]).values.astype(np.float32)
#     y1 = df1[target].values.astype(np.float32)
#     cross_validate(DynamicCNN, X1, y1)
#
#     # 交叉验证第二个模型
#     print("\nCross Validating Model 2...")
#     X2 = df2.drop(columns=[target]).values.astype(np.float32)
#     y2 = df2[target].values.astype(np.float32)
#     cross_validate(DynamicCNN, X2, y2)
#
#     # 初始化最佳模型
#     cnn1 = DynamicCNN(input_dim=X1.shape[1]).to(device)
#     cnn1.load_state_dict(torch.load('best_model_DynamicCNN.pth'))
#
#     cnn2 = DynamicCNN(input_dim=X2.shape[1]).to(device)
#     cnn2.load_state_dict(torch.load('best_model_DynamicCNN.pth'))
#
#     # 训练融合权重
#     ensemble = AutoWeightEnsemble(cnn1, cnn2).to(device)
#     ensemble_optimizer = optim.Adam(ensemble.parameters(), lr=0.001)
#
#     # 创建联合验证集
#     min_len = min(len(X1), len(X2))
#     X1_joint, X2_joint = X1[:min_len], X2[:min_len]
#     y_joint = y1[:min_len]
#
#     joint_loader = DataLoader(
#         TensorDataset(
#             torch.FloatTensor(X1_joint),
#             torch.FloatTensor(X2_joint),
#             torch.FloatTensor(y_joint)
#         ),
#         batch_size=64, shuffle=True
#     )
#
#     # 训练融合模型
#     print("\nTraining Ensemble Weights:")
#     for epoch in range(15):
#         total_loss = 0
#         ensemble.train()
#         for X1, X2, y in joint_loader:
#             X1, X2, y = X1.to(device), X2.to(device), y.to(device)
#             ensemble_optimizer.zero_grad()
#             outputs = ensemble(X1, X2)
#             loss = nn.BCELoss()(outputs, y)
#             loss.backward()
#             ensemble_optimizer.step()
#             total_loss += loss.item()
#
#         # 显示当前权重
#         current_weights = torch.softmax(ensemble.weights, dim=0).detach().cpu().numpy()
#         print(f"Epoch {epoch + 1}: Loss {total_loss / len(joint_loader):.4f} | "
#               f"Weights: ModelA={current_weights[0]:.3f}, ModelB={current_weights[1]:.3f}")
#
#     # 可视化权重变化
#     ensemble.plot_weights()
#
#     # 最终评估
#     ensemble.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for X1, X2, y in joint_loader:
#             X1, X2 = X1.to(device), X2.to(device)
#             prob = ensemble(X1, X2)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(prob.cpu().numpy() > 0.5)
#
#     print("\nFinal Ensemble Performance:")
#     print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_true, y_pred):.4f}")
#     print(f"Recall: {recall_score(y_true, y_pred):.4f}")
#     print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
#     print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")
#
#
# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             roc_curve)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, TensorDataset,
                              WeightedRandomSampler)
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------
# 时间序列增强模块
# --------------------------
class TemporalAugmentation:
    """时间序列数据增强"""

    def __init__(self, sigma=0.1, p=0.5):
        self.sigma = sigma  # 噪声强度
        self.p = p  # 应用概率

    def __call__(self, x):
        if np.random.rand() < self.p:
            # 添加高斯噪声
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


# --------------------------
# 残差块模块（改进版）
# --------------------------
class ResidualBlock(nn.Module):
    """带通道注意力机制的残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 通道注意力机制
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 通道注意力
        ca_weight = self.ca(out)
        out = out * ca_weight

        out += identity
        out = self.relu(out)
        return out


# --------------------------
# 改进的CNN模型
# --------------------------
class DynamicCNN(nn.Module):
    """带数据增强和时间感知的CNN"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # 数据增强层
        self.augment = Compose([
            TemporalAugmentation(sigma=0.05, p=0.3),
        ])

        # 特征预处理
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.GELU()
        )

        # 残差卷积模块
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            ResidualBlock(16, 16),
            nn.MaxPool1d(2),
            ResidualBlock(16, 32),
            nn.AdaptiveAvgPool1d(8)
        )

        # 动态计算全连接输入维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, 16)
            self.fc_input = self.conv_layers(dummy).view(-1).shape[0]

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, augment=True):
        # 训练时应用数据增强
        if self.training and augment:
            x = self.augment(x)

        x = self.preprocess(x)
        x = x.unsqueeze(1)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features).squeeze(1)


# --------------------------
# 注意力融合模块
# --------------------------
class AttentionFusion(nn.Module):
    """基于注意力的模型融合"""

    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # 获取两个模型的原始输出
        with torch.no_grad():
            logitA = self.modelA(x1)
            logitB = self.modelB(x2)

        # 计算注意力权重
        concat_logits = torch.stack([logitA, logitB], dim=1)
        weights = self.attention(concat_logits)

        # 加权融合
        probA = torch.sigmoid(logitA)
        probB = torch.sigmoid(logitB)
        return (weights[:, 0] * probA) + (weights[:, 1] * probB)


# --------------------------
# 模型解释模块
# --------------------------
def feature_importance(model, X, feature_names, n_samples=1000):
    """特征重要性分析"""
    model.eval()
    baseline = torch.mean(X, dim=0, keepdim=True)

    delta_list = []
    with torch.no_grad():
        for i in tqdm(range(X.shape[1])):
            perturbed = X.clone()
            perturbed[:, i] = baseline[0, i]
            orig_output = torch.sigmoid(model(X))
            perturbed_output = torch.sigmoid(model(perturbed))
            delta = torch.mean(torch.abs(orig_output - perturbed_output)).item()
            delta_list.append(delta)

    # 可视化
    indices = np.argsort(delta_list)[::-1]
    plt.figure(figsize=(12, 6))
    plt.barh(range(20), [delta_list[i] for i in indices[:20]][::-1])
    plt.yticks(range(20), [feature_names[i] for i in indices[:20]][::-1])
    plt.xlabel('Importance Score')
    plt.title('Top 20 Important Features')
    plt.show()


# --------------------------
# 训练评估模块（优化版）
# --------------------------
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=30):
    history = {'train_loss': [], 'val_auc': [], 'val_f1': [],
               'val_accuracy': [], 'val_precision': []}
    early_stop = EarlyStopper(patience=5)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                outputs = model(X)
                y_true.extend(y.cpu().numpy())
                y_probs.extend(torch.sigmoid(outputs).cpu().numpy())

        # 计算指标
        auc = roc_auc_score(y_true, y_probs)
        f1 = f1_score(y_true, np.round(y_probs))
        accuracy = accuracy_score(y_true, np.round(y_probs))
        precision = precision_score(y_true, np.round(y_probs))
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_auc'].append(auc)
        history['val_f1'].append(f1)
        history['val_accuracy'].append(auc)
        history['val_precision'].append(f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Val AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")

        if early_stop(auc):
            print("Early stopping triggered!")
            break

    return model, history


# --------------------------
# 主流程（无索引对齐版）
# --------------------------
def main():
    # 加载数据
    df1 = pd.read_csv('./data/cleaned_jigan.csv')
    df2 = pd.read_csv('./data/cleaned_labs_first_day.csv')
    target = 'match_flag'

    # 确保数据对齐（假设样本顺序一致）
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len].reset_index(drop=True)
    df2 = df2.iloc[:min_len].reset_index(drop=True)

    # 特征工程
    X1 = df1.drop(columns=target).values.astype(np.float32)
    X2 = df2.drop(columns=target).values.astype(np.float32)
    y = df1[target].values.astype(np.float32)

    # 交叉验证两个基础模型
    print("Training Model 1...")
    cross_validate(DynamicCNN, X1, y)
    print("\nTraining Model 2...")
    cross_validate(DynamicCNN, X2, y)

    # 加载最佳模型
    cnn1 = DynamicCNN(X1.shape[1]).to(device)
    cnn1.load_state_dict(torch.load('best_model_DynamicCNN.pth'))
    cnn2 = DynamicCNN(X2.shape[1]).to(device)
    cnn2.load_state_dict(torch.load('best_model_DynamicCNN.pth'))

    # 训练注意力融合模型
    fusion_model = AttentionFusion(cnn1, cnn2).to(device)
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

    dataset = TensorDataset(torch.FloatTensor(X1),
                            torch.FloatTensor(X2),
                            torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    print("\nTraining Fusion Model:")
    for epoch in range(20):
        total_loss = 0
        fusion_model.train()
        for X1, X2, y_batch in loader:
            X1, X2, y_batch = X1.to(device), X2.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = fusion_model(X1, X2)
            loss = nn.BCELoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss={total_loss / len(loader):.4f}")

    # 模型解释
    print("\nAnalyzing Feature Importance for Model 1:")
    feature_names = df1.drop(columns=target).columns.tolist()
    feature_importance(cnn1, torch.FloatTensor(X1[:1000]), feature_names)

    print("\nAnalyzing Feature Importance for Model 2:")
    feature_names = df2.drop(columns=target).columns.tolist()
    feature_importance(cnn2, torch.FloatTensor(X2[:1000]), feature_names)

    # 最终评估
    evaluate_ensemble(fusion_model, X1, X2, y)


# --------------------------
# 辅助类和函数
# --------------------------
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


# --------------------------
# 修改后的交叉验证模块
# --------------------------
def cross_validate(model_cls, X, y, n_splits=5, epochs=30):
    """支持类别平衡的交叉验证"""
    skf = StratifiedKFold(n_splits=n_splits)
    fold_metrics = {'auc': [], 'f1': [], 'accuracy': [], 'precision': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # 数据分割
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 计算类别权重
        pos = sum(y_train)
        neg = len(y_train) - pos
        pos_weight = torch.tensor([neg / pos if pos != 0 else 1.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 创建带平衡采样的DataLoader
        sampler = create_weighted_sampler(y_train)
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=64, sampler=sampler
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=64
        )

        # 模型初始化
        model = model_cls(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 训练验证
        model, hist = train_and_evaluate(
            model, train_loader, test_loader, optimizer,
            criterion, epochs=epochs
        )

        # 记录最佳指标
        fold_metrics['auc'].append(max(hist['val_auc']))
        fold_metrics['f1'].append(max(hist['val_f1']))
        fold_metrics['accuracy'].append(max(hist['val_accuracy']))
        fold_metrics['precision'].append(max(hist['val_precision']))

    print("\nCross Validation Results:")
    print(f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
    print(f"F1: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
    print(f"Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
    print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")


# --------------------------
# 类别平衡处理（修改训练函数）
# --------------------------
def create_weighted_sampler(y):
    """创建加权采样器"""
    class_counts = np.bincount(y.astype(int))
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(sample_weights, len(y))


def evaluate_ensemble(model, X1, X2, y):
    loader = DataLoader(TensorDataset(torch.FloatTensor(X1),
                                      torch.FloatTensor(X2)),
                        batch_size=256)

    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for X1_batch, X2_batch in loader:
            X1_batch, X2_batch = X1_batch.to(device), X2_batch.to(device)
            batch_probs = model(X1_batch, X2_batch).cpu().numpy()
            probs.extend(batch_probs)
            truths.extend(y[:len(X1_batch)])

    # 计算指标
    preds = (np.array(probs) > 0.5).astype(int)
    print("\nFinal Performance:")
    print(f"AUC: {roc_auc_score(truths, probs):.4f}")
    print(f"Accuracy: {accuracy_score(truths, preds):.4f}")
    print(f"Precision: {precision_score(truths, preds):.4f}")
    print(f"Recall: {recall_score(truths, preds):.4f}")
    print(f"F1: {f1_score(truths, preds):.4f}")

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(truths, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(truths, probs):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()