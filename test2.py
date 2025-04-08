import random
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
from scipy.interpolate import interp1d

# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x



class TimeWarp:
    def __call__(self, x):
        x_cpu = x.cpu().numpy()  # 先移动到 CPU 并转换为 NumPy 数组
        original_indices = np.arange(x_cpu.shape[-1])
        new_indices = np.linspace(0, original_indices.max(), num=len(original_indices))
        interpolator = interp1d(original_indices, x_cpu, axis=-1, kind='linear')
        warped = interpolator(new_indices)
        return torch.tensor(warped, dtype=torch.float32, device=x.device)  # 还原到原设备



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
        self.augment = Compose([TemporalAugmentation(sigma=0.05, p=0.3),TimeWarp()])

        # 特征预处理
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, 16),
            # nn.BatchNorm1d(16),
            nn.LayerNorm(16),
            nn.GELU()
        )

        # 残差卷积模块
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            ResidualBlock(16, 16),
            nn.Dropout(0.2),
            nn.MaxPool1d(2),
            ResidualBlock(16, 32),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1),
            ResidualBlock(64, 128),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(8)
        )

        # 动态计算全连接输入维度
        with torch.no_grad():
            dummy = torch.randn(2, input_dim)
            dummy = self.preprocess(dummy).unsqueeze(1)
            dummy = self.conv_layers(dummy)
            self.fc_input = dummy.view(dummy.size(0), -1).shape[1]

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, augment=True):
        if self.training and augment:
            x = self.augment(x)
        x = self.preprocess(x).unsqueeze(1)
        features = self.conv_layers(x).view(x.size(0), -1)
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
        self.attention = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        with torch.no_grad():
            logitA = self.modelA(x1)
            logitB = self.modelB(x2)
        concat_logits = torch.stack([logitA, logitB], dim=1)
        weights = self.attention(concat_logits)
        probA = torch.sigmoid(logitA)
        probB = torch.sigmoid(logitB)
        return (weights[:, 0] * probA) + (weights[:, 1] * probB)


# --------------------------
# 模型解释模块
# --------------------------
def feature_importance(model, X, feature_names, n_samples=1000):
    model.eval()
    baseline = torch.mean(X, dim=0, keepdim=True)
    delta_list = []

    # 确保特征数量与特征名称数量一致
    if X.shape[1] != len(feature_names):
        print(f"警告：特征数量 ({X.shape[1]}) 与特征名称数量 ({len(feature_names)}) 不匹配")
        print("可能原因：数据预处理时某些列未被正确移除或加载")
        return

    with torch.no_grad():
        for i in tqdm(range(X.shape[1])):
            perturbed = X.clone()
            perturbed[:, i] = baseline[0, i]
            orig_output = torch.sigmoid(model(X))
            perturbed_output = torch.sigmoid(model(perturbed))
            delta = torch.mean(torch.abs(orig_output - perturbed_output)).item()
            delta_list.append(delta)

    # 动态确定显示数量
    display_num = min(20, len(delta_list))  # 取特征数量和前20中的较小值
    indices = np.argsort(delta_list)[::-1][:display_num]  # 只取实际存在的索引

    plt.figure(figsize=(12, 6))
    plt.barh(range(display_num), [delta_list[i] for i in indices][::-1])
    plt.yticks(range(display_num), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {display_num} Important Features')
    plt.tight_layout()
    plt.show()


# --------------------------
# 训练评估模块（优化版）
# --------------------------
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion,
                       scheduler=None, epochs=30, save_path='best_model.pth'):
    history = {'train_loss': [], 'val_auc': [], 'val_f1': [],
               'val_accuracy': [], 'val_precision': []}
    best_auc = 0
    early_stop = EarlyStopper(patience=5,min_delta=0.01)

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
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

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
        preds = np.round(y_probs)
        f1 = f1_score(y_true, preds)
        accuracy = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds)

        history['val_auc'].append(auc)
        history['val_f1'].append(f1)
        history['val_accuracy'].append(accuracy)
        history['val_precision'].append(precision)

        # 学习率调度
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(auc)
            else:
                scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")

        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with AUC: {auc:.4f}")

        if early_stop(auc):
            print("Early stopping triggered!")
            break

    model.load_state_dict(torch.load(save_path))
    return model, history


# --------------------------
# 主流程
# --------------------------
def main():
    # 加载数据
    df1 = pd.read_csv('./data/cleaned_microbiologyevents.csv')
    df2 = pd.read_csv('./data/cleaned_labs_first_day_lgbm.csv')
    target = 'match_flag'

    # 数据对齐
    common_ids = np.intersect1d(df1['hadm_id'], df2['hadm_id'])  # 假设存在唯一标识列
    df1 = df1[df1['hadm_id'].isin(common_ids)].sort_values('hadm_id').reset_index(drop=True)
    df2 = df2[df2['hadm_id'].isin(common_ids)].sort_values('hadm_id').reset_index(drop=True)

    # 特征工程
    feature_names1 = df1.drop(columns=[target, 'hadm_id']).columns.tolist()  # 提取特征名称
    feature_names2 = df2.drop(columns=[target, 'hadm_id']).columns.tolist()
    X1 = df1.drop(columns=[target, 'hadm_id']).values.astype(np.float32)  # 移除标识列和目标列
    X2 = df2.drop(columns=[target, 'hadm_id']).values.astype(np.float32)
    y = df1[target].values.astype(np.float32)

    # 添加数据完整性检查
    print("\n数据完整性验证：")
    print(f"X1样本数: {len(X1)}, 特征数: {X1.shape[1]}, 特征名称数: {len(feature_names1)}")
    print(f"X2样本数: {len(X2)}, 特征数: {X2.shape[1]}, 特征名称数: {len(feature_names2)}")
    print(f"正类比例: {np.mean(y):.2%}")
    assert len(X1) == len(X2) == len(y), "特征与标签数量不匹配"
    assert X1.shape[1] == len(feature_names1), "X1 特征数量与特征名称不匹配"
    assert X2.shape[1] == len(feature_names2), "X2 特征数量与特征名称不匹配"

    # 训练模型1
    print("\nTraining Model 1...")
    dataset1 = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(y))
    train_loader1, val_loader1 = create_loaders(dataset1)
    model1 = DynamicCNN(X1.shape[1]).to(device)
    optimizer1 = optim.AdamW(model1.parameters(), lr=0.001, weight_decay=1e-4)
    criterion1 = nn.BCEWithLogitsLoss(pos_weight=calc_pos_weight(y))
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='max', factor=0.1, patience=3)
    model1, hist1 = train_and_evaluate(model1, train_loader1, val_loader1, optimizer1,
                                       criterion1, scheduler=scheduler1, epochs=100, save_path='best_model1.pth')

    # 训练模型2
    print("\nTraining Model 2...")
    dataset2 = TensorDataset(torch.FloatTensor(X2), torch.FloatTensor(y))
    train_loader2, val_loader2 = create_loaders(dataset2)
    model2 = DynamicCNN(X2.shape[1]).to(device)
    optimizer2 = optim.AdamW(model2.parameters(), lr=0.001, weight_decay=1e-4)
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=calc_pos_weight(y))
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.1, patience=3)
    model2, hist2 = train_and_evaluate(model2, train_loader2, val_loader2, optimizer2,
                                       criterion2, scheduler=scheduler2, save_path='best_model2.pth')

    # 加载最佳模型
    model1.load_state_dict(torch.load('best_model1.pth'))
    model2.load_state_dict(torch.load('best_model2.pth'))

    # 训练融合模型
    print("\nTraining Fusion Model...")
    fusion_model = AttentionFusion(model1, model2).to(device)
    optimizer = optim.AdamW(fusion_model.parameters(), lr=0.001)
    scheduler_fusion = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    dataset = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(X2), torch.FloatTensor(y))
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    epoches = 20
    for epoch in range(epoches):
        total_loss = 0
        fusion_model.train()
        for X1_batch, X2_batch, y_batch in train_loader:
            X1_batch, X2_batch = X1_batch.to(device), X2_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            probs = fusion_model(X1_batch, X2_batch)
            loss = nn.BCELoss()(probs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler_fusion.step()
        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

    # 模型解释与评估
    print("\nFeature Importance for Model 1:")
    feature_importance(model1, torch.FloatTensor(X1[:1000]).to(device), feature_names1)
    print("\nFeature Importance for Model 2:")
    feature_importance(model2, torch.FloatTensor(X2[:1000]).to(device), feature_names2)
    evaluate_ensemble(fusion_model, X1, X2, y)


# --------------------------
# 辅助函数
# --------------------------

import shap

shap.initjs()


def shap_evaluation(model, background_data, evaluation_data, feature_names, model_name=""):
    """
    SHAP模型解释模块
    参数:
        model: 待分析的PyTorch模型
        background_data: 用于计算基准分布的背景数据(Tensor)
        evaluation_data: 待解释的样本数据(Tensor)
        feature_names: 特征名称列表
        model_name: 模型名称(用于保存结果)
    """
    # 确保模型在CPU且处于评估模式
    model.to('cpu').eval()

    # 定义模型的前向传播函数适配器
    def model_wrapper(x):
        tensor_x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(tensor_x).detach().numpy()
        if outputs.ndim == 1:
            return np.vstack([1 - outputs, outputs]).T  # 转换为概率矩阵
        return outputs

    # 数据转换
    background = background_data.numpy() if not isinstance(background_data, np.ndarray) else background_data
    samples = evaluation_data.numpy() if not isinstance(evaluation_data, np.ndarray) else evaluation_data

    # 创建解释器（不同方法选择）
    if len(background) > 100:  # 使用KernelExplainer处理大数据
        explainer = shap.KernelExplainer(model_wrapper, shap.kmeans(background, 100))
    else:  # 使用DeepExplainer
        explainer = shap.DeepExplainer(model, background_data)

    # 计算SHAP值
    shap_values = explainer.shap_values(samples)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    if isinstance(shap_values, list):  # 分类任务多类别输出
        shap.summary_plot(shap_values[1], samples, feature_names=feature_names, show=False)
    else:  # 回归或二分类任务
        shap.summary_plot(shap_values, samples, feature_names=feature_names, show=False)
    plt.title(f'SHAP Feature Importance ({model_name})')
    plt.tight_layout()
    plt.savefig(f'shap_{model_name}.png', dpi=300)
    plt.show()

    # 返回样本级解释
    return shap_values


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.005):
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


def create_loaders(dataset, val_ratio=0.2):
    # 使用分层划分
    y = dataset.tensors[1].numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(np.zeros(len(y)), y))

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    # 类别平衡采样
    y_train = y[train_idx]
    sampler = WeightedRandomSampler(
        weights=calc_sample_weights(y_train),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64)
    return train_loader, val_loader


def calc_pos_weight(y):
    pos = np.sum(y)
    neg = len(y) - pos
    return torch.tensor([neg / pos]).to(device) if pos > 0 else torch.tensor([1.0]).to(device)


def calc_sample_weights(y):
    class_counts = np.bincount(y.astype(int))
    class_weights = 1. / class_counts
    return torch.tensor([class_weights[int(label)] for label in y])


def evaluate_ensemble(model, X1, X2, y):
    # 创建数据集时确保顺序一致
    dataset = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(X2), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)  # 必须关闭shuffle

    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for X1_batch, X2_batch, y_batch in loader:
            X1_batch, X2_batch = X1_batch.to(device), X2_batch.to(device)
            batch_probs = model(X1_batch, X2_batch).cpu().numpy()
            probs.extend(batch_probs)
            truths.extend(y_batch.cpu().numpy())  # 直接使用loader提供的标签

    # 转换为numpy数组并验证
    truths = np.array(truths)
    probs = np.array(probs)
    print(f"\n最终验证集类别分布 - 负类: {np.sum(truths == 0)}, 正类: {np.sum(truths == 1)}")

    # 检查类别分布
    unique_classes = np.unique(truths)
    if len(unique_classes) == 1:
        print("\n警告：验证集只包含单一类别，无法计算AUC")
        class_dist = {0: np.sum(truths == 0), 1: np.sum(truths == 1)}
        print(f"类别分布: {class_dist}")
        return

    # 计算评估指标
    preds = np.round(probs)
    print("\nFinal Ensemble Performance:")
    try:
        auc = roc_auc_score(truths, probs)
        print(f"AUC: {auc:.4f}")
    except ValueError as e:
        print(f"AUC计算失败: {str(e)}")
        auc = 0

    print(f"Accuracy: {accuracy_score(truths, preds):.4f}")
    print(f"Precision: {precision_score(truths, preds):.4f}")
    print(f"Recall: {recall_score(truths, preds):.4f}")
    print(f"F1: {f1_score(truths, preds):.4f}")

    # 绘制ROC曲线
    if auc > 0:
        fpr, tpr, _ = roc_curve(truths, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()