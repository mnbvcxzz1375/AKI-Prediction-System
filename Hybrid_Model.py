# import random
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import (accuracy_score, precision_score,
#                              recall_score, f1_score, roc_auc_score,
#                              roc_curve)
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import (DataLoader, TensorDataset,
#                               WeightedRandomSampler)
# import matplotlib.pyplot as plt
# from torchvision.transforms import Compose
# from tqdm import tqdm
# import torch.nn.functional as F
# from functools import partial
#
# # 固定随机种子
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# # --------------------------
# # 时间序列增强模块
# # --------------------------
# class TemporalAugmentation:
#     """时间序列数据增强"""
#
#     def __init__(self, sigma=0.1, p=0.5):
#         self.sigma = sigma  # 噪声强度
#         self.p = p  # 应用概率
#
#     def __call__(self, x):
#         if np.random.rand() < self.p:
#             noise = torch.randn_like(x) * self.sigma
#             return x + noise
#         return x
#
#
# # --------------------------
# # 深度残差块（带瓶颈结构）
# # --------------------------
# class BottleneckResidualBlock(nn.Module):
#     """带瓶颈结构的深度残差块"""
#     expansion = 4  # 通道扩展系数
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#
#         self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         # 跳跃连接
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels * self.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels * self.expansion)
#             )
#
#         # self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()
#         # 通道注意力（改进版SE模块）
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(out_channels * self.expansion, out_channels * self.expansion // 16, 1),
#             nn.ReLU(),
#             nn.Conv1d(out_channels * self.expansion // 16, out_channels * self.expansion, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         # 通道注意力应用
#         se_weight = self.se(out)
#         out = out * se_weight
#
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# # --------------------------
# # 深度残差网络架构
# # --------------------------
# class DeepResNet(nn.Module):
#     """深度残差网络（>50层配置）"""
#
#     def __init__(self, input_dim, block=BottleneckResidualBlock, layers=[3, 4, 23, 3]):
#         """
#         参数:
#             input_dim: 输入特征维度
#             block: 残差块类型
#             layers: 各阶段残差块数量 [stage1, stage2, stage3, stage4]
#         """
#         super().__init__()
#         self.in_channels = 64
#
#         # 初始卷积层
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         )
#
#         # 残差阶段
#         self.stage1 = self._make_stage(block, 64, layers[0], stride=1)
#         self.stage2 = self._make_stage(block, 128, layers[1], stride=2)
#         self.stage3 = self._make_stage(block, 256, layers[2], stride=2)
#         self.stage4 = self._make_stage(block, 512, layers[3], stride=2)
#
#         # 自适应池化
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#
#         # 全连接层
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(512 * block.expansion, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1)
#         )
#
#         # 参数初始化
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_stage(self, block, out_channels, blocks, stride=1):
#         """构建网络阶段"""
#         layers = []
#         # 第一个块处理下采样
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion
#         # 后续块保持通道数不变
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # 初始特征提取
#         x = x.unsqueeze(1)  # 添加通道维度 [batch, 1, seq_len]
#         x = self.conv1(x)
#
#         # 残差阶段
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#
#         # 池化和分类
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x.squeeze(1)
#
#
# # --------------------------
# # 改进的CNN模型
# # --------------------------
# class DynamicCNN(nn.Module):
#     """带数据增强和时间感知的CNN"""
#
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = input_dim
#
#         # 数据增强层
#         self.augment = Compose([TemporalAugmentation(sigma=0.05, p=0.3)])
#
#         # 特征预处理
#         self.preprocess = nn.Sequential(
#             nn.Linear(input_dim, 16),
#             nn.BatchNorm1d(16),
#             nn.GELU()
#         )
#
#         # 残差卷积模块
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(1, 16, 3, padding=1),
#             BottleneckResidualBlock(16, 16),
#             nn.MaxPool1d(2),
#             BottleneckResidualBlock(16, 32),
#             nn.AdaptiveAvgPool1d(8)
#         )
#
#         # 动态计算全连接输入维度
#         with torch.no_grad():
#             dummy = torch.randn(2, input_dim)
#             dummy = self.preprocess(dummy).unsqueeze(1)
#             dummy = self.conv_layers(dummy)
#             self.fc_input = dummy.view(dummy.size(0), -1).shape[1]
#
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(self.fc_input, 64),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x, augment=True):
#         if self.training and augment:
#             x = self.augment(x)
#         x = self.preprocess(x).unsqueeze(1)
#         features = self.conv_layers(x).view(x.size(0), -1)
#         return self.classifier(features).squeeze(1)
#
#
# # --------------------------
# # 注意力融合模块
# # --------------------------
# class AttentionFusion(nn.Module):
#     """基于注意力的模型融合"""
#
#     def __init__(self, modelA, modelB):
#         super().__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.attention = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.Tanh(),
#             nn.Linear(8, 2),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x1, x2):
#         with torch.no_grad():
#             logitA = self.modelA(x1)
#             logitB = self.modelB(x2)
#         concat_logits = torch.stack([logitA, logitB], dim=1)
#         weights = self.attention(concat_logits)
#         probA = torch.sigmoid(logitA)
#         probB = torch.sigmoid(logitB)
#         return (weights[:, 0] * probA) + (weights[:, 1] * probB)
#
#
# class HierarchicalAttentionFusion(nn.Module):
#     """多层次特征融合"""
#
#     def __init__(self, modelA, modelB, feat_layers=['layer3', 'layer4']):
#         super().__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#
#         # 注册钩子获取中间特征
#         self.featA = {}
#         self.featB = {}
#         for layer in feat_layers:
#             getattr(modelA, layer).register_forward_hook(
#                 lambda m, inp, out, layer=layer: self._save_feat('A', layer, out)
#             )
#             getattr(modelB, layer).register_forward_hook(
#                 lambda m, inp, out, layer=layer: self._save_feat('B', layer, out)
#             )
#
#         # 交叉注意力机制
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=512, num_heads=8, batch_first=True)
#
#         # 动态融合门控
#         self.gate = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 2),
#             nn.Softmax(dim=1)
#         )
#
#     def _save_feat(self, model_id, layer, output):
#         self.feats[f'{model_id}_{layer}'] = output.mean(dim=[2])  # 全局平均池化
#
#     def forward(self, x1, x2):
#         # 获取基础预测
#         with torch.no_grad():
#             logitA = self.modelA(x1)
#             logitB = self.modelB(x2)
#
#         # 特征对齐与融合
#         fused_feats = []
#         for layer in self.feat_layers:
#             featA = self.featA[layer]
#             featB = self.featB[layer]
#
#             # 交叉注意力
#             attn_feat, _ = self.cross_attn(featA.unsqueeze(1),
#                                            featB.unsqueeze(1),
#                                            featB.unsqueeze(1))
#             fused_feats.append(attn_feat.squeeze())
#
#         # 门控融合
#         global_feat = torch.cat(fused_feats, dim=1)
#         weights = self.gate(global_feat)
#
#         # 概率融合
#         probA = torch.sigmoid(logitA)
#         probB = torch.sigmoid(logitB)
#         return (weights[:, 0] * probA) + (weights[:, 1] * probB)
#
#
# # --------------------------
# # 模型解释模块
# # --------------------------
# def feature_importance(model, X, feature_names, n_samples=1000):
#     model.eval()
#     baseline = torch.mean(X, dim=0, keepdim=True)
#     delta_list = []
#
#     # 确保特征数量与特征名称数量一致
#     if X.shape[1] != len(feature_names):
#         print(f"警告：特征数量 ({X.shape[1]}) 与特征名称数量 ({len(feature_names)}) 不匹配")
#         print("可能原因：数据预处理时某些列未被正确移除或加载")
#         return
#
#     with torch.no_grad():
#         for i in tqdm(range(X.shape[1])):
#             perturbed = X.clone()
#             perturbed[:, i] = baseline[0, i]
#             orig_output = torch.sigmoid(model(X))
#             perturbed_output = torch.sigmoid(model(perturbed))
#             delta = torch.mean(torch.abs(orig_output - perturbed_output)).item()
#             delta_list.append(delta)
#
#     # 动态确定显示数量
#     display_num = min(20, len(delta_list))  # 取特征数量和前20中的较小值
#     indices = np.argsort(delta_list)[::-1][:display_num]  # 只取实际存在的索引
#
#     plt.figure(figsize=(12, 6))
#     plt.barh(range(display_num), [delta_list[i] for i in indices][::-1])
#     plt.yticks(range(display_num), [feature_names[i] for i in indices][::-1])
#     plt.xlabel('Importance Score')
#     plt.title(f'Top {display_num} Important Features')
#     plt.tight_layout()
#     plt.show()
#
#
# # --------------------------
# # 训练评估模块（优化版）
# # --------------------------
# def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion,
#                        scheduler=None, epochs=30, save_path='best_model.pth'):
#     history = {'train_loss': [], 'val_auc': [], 'val_f1': [],
#                'val_accuracy': [], 'val_precision': []}
#     best_auc = 0
#     early_stop = EarlyStopper(patience=10)
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
#         train_loss /= len(train_loader)
#         history['train_loss'].append(train_loss)
#
#         # 验证阶段
#         model.eval()
#         y_true, y_probs = [], []
#         with torch.no_grad():
#             for X, y in val_loader:
#                 X = X.to(device)
#                 outputs = model(X)
#                 y_true.extend(y.cpu().numpy())
#                 y_probs.extend(torch.sigmoid(outputs).cpu().numpy())
#
#         # 计算指标
#         auc = roc_auc_score(y_true, y_probs)
#         preds = np.round(y_probs)
#         f1 = f1_score(y_true, preds)
#         accuracy = accuracy_score(y_true, preds)
#         precision = precision_score(y_true, preds)
#
#         history['val_auc'].append(auc)
#         history['val_f1'].append(f1)
#         history['val_accuracy'].append(accuracy)
#         history['val_precision'].append(precision)
#
#         # 学习率调度
#         if scheduler:
#             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                 scheduler.step(auc)
#             else:
#                 scheduler.step()
#
#         print(f"Epoch {epoch + 1}/{epochs}")
#         print(f"Train Loss: {train_loss:.4f}")
#         print(f"Val AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
#
#         # 保存最佳模型
#         if auc > best_auc:
#             best_auc = auc
#             torch.save(model.state_dict(), save_path)
#             print(f"Saved new best model with AUC: {auc:.4f}")
#
#         if early_stop(auc):
#             print("Early stopping triggered!")
#             break
#
#     model.load_state_dict(torch.load(save_path))
#     return model, history
#
#
# # --------------------------
# # 辅助函数
# # --------------------------
#
# import shap
#
# shap.initjs()
#
#
# def shap_feature_analysis(model, background_data, evaluation_data, feature_names, model_name="", max_samples=200):
#     """
#     改进的SHAP特征分析模块
#     参数:
#         model: 训练好的PyTorch模型
#         background_data: 用于解释的基准数据 (Tensor)
#         evaluation_data: 待分析的数据 (Tensor)
#         feature_names: 特征名称列表
#         model_name: 模型标识符
#         max_samples: 最大解释样本数
#     """
#     # 设备切换和模型模式设置
#     original_device = next(model.parameters()).device
#     model.to('cpu').eval()
#
#     # 数据预处理
#     background = background_data.cpu().numpy()[:500]  # 限制背景数据量
#     samples = evaluation_data.cpu().numpy()[:max_samples]
#
#     # 动态选择解释器
#     if len(background) > 300:
#         explainer = shap.KernelExplainer(
#             lambda x: torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).detach().numpy(),
#             shap.kmeans(background, 100)
#         )
#     else:
#         explainer = shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32))
#
#     # 计算SHAP值
#     shap_values = explainer.shap_values(samples)
#
#     # 可视化解释
#     plt.figure(figsize=(12, 6))
#     shap.summary_plot(
#         shap_values if isinstance(shap_values, list) else shap_values[1],
#         samples,
#         feature_names=feature_names,
#         plot_type="bar",
#         show=False
#     )
#     plt.title(f'SHAP Feature Importance - {model_name}')
#     plt.tight_layout()
#     plt.savefig(f'shap_{model_name}.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # 恢复模型原始状态
#     model.to(original_device)
#     return shap_values
#
#
# class EarlyStopper:
#     def __init__(self, patience=10, min_delta=0.005):
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
#
#
# def create_loaders(dataset, val_ratio=0.2):
#     # 使用分层划分
#     y = dataset.tensors[1].numpy()
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     train_idx, val_idx = next(skf.split(np.zeros(len(y)), y))
#
#     train_dataset = torch.utils.data.Subset(dataset, train_idx)
#     val_dataset = torch.utils.data.Subset(dataset, val_idx)
#
#     # 类别平衡采样
#     y_train = y[train_idx]
#     sampler = WeightedRandomSampler(
#         weights=calc_sample_weights(y_train),
#         num_samples=len(train_dataset),
#         replacement=True
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
#     val_loader = DataLoader(val_dataset, batch_size=64)
#     return train_loader, val_loader
#
#
# def calc_pos_weight(y):
#     pos = np.sum(y)
#     neg = len(y) - pos
#     return torch.tensor([neg / pos]).to(device) if pos > 0 else torch.tensor([1.0]).to(device)
#
#
# def calc_sample_weights(y):
#     class_counts = np.bincount(y.astype(int))
#     class_weights = 1. / class_counts
#     return torch.tensor([class_weights[int(label)] for label in y])
#
#
# def evaluate_ensemble(model, X1, X2, y):
#     # 创建数据集时确保顺序一致
#     dataset = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(X2), torch.FloatTensor(y))
#     loader = DataLoader(dataset, batch_size=256, shuffle=False)  # 必须关闭shuffle
#
#     model.eval()
#     probs, truths = [], []
#     with torch.no_grad():
#         for X1_batch, X2_batch, y_batch in loader:
#             X1_batch, X2_batch = X1_batch.to(device), X2_batch.to(device)
#             batch_probs = model(X1_batch, X2_batch).cpu().numpy()
#             probs.extend(batch_probs)
#             truths.extend(y_batch.cpu().numpy())  # 直接使用loader提供的标签
#
#     # 转换为numpy数组并验证
#     truths = np.array(truths)
#     probs = np.array(probs)
#     print(f"\n最终验证集类别分布 - 负类: {np.sum(truths == 0)}, 正类: {np.sum(truths == 1)}")
#
#     # 检查类别分布
#     unique_classes = np.unique(truths)
#     if len(unique_classes) == 1:
#         print("\n警告：验证集只包含单一类别，无法计算AUC")
#         class_dist = {0: np.sum(truths == 0), 1: np.sum(truths == 1)}
#         print(f"类别分布: {class_dist}")
#         return
#
#     # 计算评估指标
#     preds = np.round(probs)
#     print("\nFinal Ensemble Performance:")
#     try:
#         auc = roc_auc_score(truths, probs)
#         print(f"AUC: {auc:.4f}")
#     except ValueError as e:
#         print(f"AUC计算失败: {str(e)}")
#         auc = 0
#
#     print(f"Accuracy: {accuracy_score(truths, preds):.4f}")
#     print(f"Precision: {precision_score(truths, preds):.4f}")
#     print(f"Recall: {recall_score(truths, preds):.4f}")
#     print(f"F1: {f1_score(truths, preds):.4f}")
#
#     # 绘制ROC曲线
#     if auc > 0:
#         fpr, tpr, _ = roc_curve(truths, probs)
#         plt.figure(figsize=(8, 6))
#         plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
#         plt.plot([0, 1], [0, 1], 'k--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC Curve')
#         plt.legend()
#         plt.show()
#
#
#
# class HierarchicalAttentionFusion(nn.Module):
#     """最终稳定版（解决KeyError问题）"""
#
#     def __init__(self, modelA, modelB, feat_layers=['stage2', 'stage3']):
#         super().__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.feat_layers = feat_layers
#
#         # 特征存储结构
#         self.feats = {'A': {}, 'B': {}}
#         self.feat_dims = {}
#
#         # 先注册钩子
#         self._register_dim_hooks()
#
#         # 运行虚拟输入以触发维度检测
#         self._detect_feature_dims()
#
#         # 动态构建网络组件
#         self.projections = nn.ModuleDict()
#         self.cross_attns = nn.ModuleDict()
#         for layer in feat_layers:
#             dim = self.feat_dims[layer]
#             self.projections[layer] = nn.Linear(dim, 512)
#             self.cross_attns[layer] = nn.MultiheadAttention(
#                 embed_dim=512, num_heads=8, batch_first=True)
#
#         # 融合门控
#         self.gate = nn.Sequential(
#             nn.Linear(len(feat_layers) * 512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 2),
#             nn.Softmax(dim=1)
#         )
#
#     def _register_dim_hooks(self):
#         """注册维度检测钩子"""
#
#         def dim_hook(module, inp, out, layer):
#             # 确保获取的是卷积层后的通道维度
#             if out.ndim == 3:  # [batch, channels, seq_len]
#                 self.feat_dims[layer] = out.shape[1]
#             else:
#                 raise RuntimeError(f"特征层 {layer} 的输出维度异常，应为3维张量")
#
#         # 验证层是否存在
#         for layer in self.feat_layers:
#             if not hasattr(self.modelA, layer):
#                 raise AttributeError(f"模型A不存在 {layer} 层")
#             if not hasattr(self.modelB, layer):
#                 raise AttributeError(f"模型B不存在 {layer} 层")
#
#             # 注册模型A的钩子
#             getattr(self.modelA, layer).register_forward_hook(
#                 partial(dim_hook, layer=layer)
#             )
#             # 注册模型B的钩子
#             getattr(self.modelB, layer).register_forward_hook(
#                 partial(dim_hook, layer=layer)
#             )
#
#     def _detect_feature_dims(self):
#         """通过虚拟输入触发维度检测"""
#         with torch.no_grad():
#             # 生成符合模型输入的虚拟数据
#             dummy_input = torch.randn(2, 100)  # 假设输入维度为100
#
#             # 确保执行到所有目标层
#             try:
#                 _ = self.modelA(dummy_input.to(next(self.modelA.parameters()).device))
#                 _ = self.modelB(dummy_input.to(next(self.modelB.parameters()).device))
#             except Exception as e:
#                 raise RuntimeError("虚拟数据前向传播失败，请检查模型输入维度") from e
#
#         # 验证所有层维度已检测
#         missing = [layer for layer in self.feat_layers if layer not in self.feat_dims]
#         if missing:
#             raise KeyError(f"以下层未检测到维度信息：{missing}，请检查：\n"
#                            f"1. 层名称是否正确\n"
#                            f"2. 模型结构是否包含这些层\n"
#                            f"3. 虚拟数据是否成功通过所有目标层")
#
#     def _save_feat(self, model_id, layer, output):
#         """保存中间特征"""
#         if output.ndim == 3:
#             pooled = F.adaptive_avg_pool1d(output, 16)
#             self.feats[model_id][layer] = pooled.permute(0, 2, 1)
#         else:
#             raise ValueError(f"特征层 {layer} 的输出维度应为3维，实际为 {output.shape}")
#
#     def forward(self, x1, x2):
#         # 清空特征缓存
#         self.feats = {'A': {}, 'B': {}}
#
#         # 前向传播获取中间特征
#         with torch.no_grad():
#             _ = self.modelA(x1)
#             _ = self.modelB(x2)
#
#         # 特征融合
#         fused = []
#         for layer in self.feat_layers:
#             # 特征投影
#             featA = self.projections[layer](self.feats['A'][layer])
#             featB = self.projections[layer](self.feats['B'][layer])
#
#             # 交叉注意力
#             attn_out, _ = self.cross_attns[layer](
#                 query=featA,
#                 key=featB,
#                 value=featB
#             )
#             fused.append(attn_out.mean(dim=1))
#
#         # 门控融合
#         weights = self.gate(torch.cat(fused, dim=1))
#
#         # 获取最终预测
#         with torch.no_grad():
#             probA = torch.sigmoid(self.modelA(x1))
#             probB = torch.sigmoid(self.modelB(x2))
#
#         return weights[:, 0] * probA + weights[:, 1] * probB
#
#
#
# # --------------------------
# # 主流程
# # --------------------------
# # 加载数据
# df1 = pd.read_csv('./data/cleaned_microbiologyevents_plus.csv')
# df2 = pd.read_csv('./data/cleaned_labs_first_day_lgbm.csv')
# target = 'match_flag'
#
# # 数据对齐
# common_ids = np.intersect1d(df1['hadm_id'], df2['hadm_id'])  # 假设存在唯一标识列
# df1 = df1[df1['hadm_id'].isin(common_ids)].sort_values('hadm_id').reset_index(drop=True)
# df2 = df2[df2['hadm_id'].isin(common_ids)].sort_values('hadm_id').reset_index(drop=True)
#
# # 特征工程
# feature_names1 = df1.drop(columns=[target, 'hadm_id']).columns.tolist()  # 提取特征名称
# feature_names2 = df2.drop(columns=[target, 'hadm_id']).columns.tolist()
# X1 = df1.drop(columns=[target, 'hadm_id']).values.astype(np.float32)  # 移除标识列和目标列
# X2 = df2.drop(columns=[target, 'hadm_id']).values.astype(np.float32)
# y = df1[target].values.astype(np.float32)
#
# # 添加数据完整性检查
# print("\n数据完整性验证：")
# print(f"X1样本数: {len(X1)}, 特征数: {X1.shape[1]}, 特征名称数: {len(feature_names1)}")
# print(f"X2样本数: {len(X2)}, 特征数: {X2.shape[1]}, 特征名称数: {len(feature_names2)}")
# print(f"正类比例: {np.mean(y):.2%}")
# assert len(X1) == len(X2) == len(y), "特征与标签数量不匹配"
# assert X1.shape[1] == len(feature_names1), "X1 特征数量与特征名称不匹配"
# assert X2.shape[1] == len(feature_names2), "X2 特征数量与特征名称不匹配"
#
# # 训练模型1
# print("\nTraining Model 1...")
# dataset1 = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(y))
# train_loader1, val_loader1 = create_loaders(dataset1)
# model1 = DeepResNet(X1.shape[1]).to(device)
# optimizer1 = optim.AdamW(model1.parameters(), lr=0.001, weight_decay=1e-4)
# criterion1 = nn.BCEWithLogitsLoss(pos_weight=calc_pos_weight(y))
# scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='max', factor=0.1, patience=3)
# model1, hist1 = train_and_evaluate(model1, train_loader1, val_loader1, optimizer1,
#                                  criterion1, scheduler=scheduler1, epochs=40,save_path='best_model1.pth')
#
# # 训练模型2
# print("\nTraining Model 2...")
# dataset2 = TensorDataset(torch.FloatTensor(X2), torch.FloatTensor(y))
# train_loader2, val_loader2 = create_loaders(dataset2)
# model2 = DeepResNet(X2.shape[1]).to(device)
# optimizer2 = optim.AdamW(model2.parameters(), lr=0.001, weight_decay=1e-4)
# criterion2 = nn.BCEWithLogitsLoss(pos_weight=calc_pos_weight(y))
# scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.1, patience=3)
# model2, hist2 = train_and_evaluate(model2, train_loader2, val_loader2, optimizer2,
#                                  criterion2, scheduler=scheduler2, save_path='best_model2.pth')
#
#
# from torch.nn import functional as F
# # 加载最佳模型
# model1.load_state_dict(torch.load('best_model1.pth'))
# model2.load_state_dict(torch.load('best_model2.pth'))
#
#     # 在模型训练后添加
# background_samples = 300  # 控制计算时间
#
# # # 对模型1的分析
# # shap_feature_analysis(
# #     model1,
# #     torch.FloatTensor(X1[:background_samples]),
# #     torch.FloatTensor(X1[:5]),  # 解释前100个样本
# #     feature_names1,
# #     model_name="Microbiology_Model"
# # )
#
# # # 对模型2的分析
# # shap_feature_analysis(
# #     model2,
# #     torch.FloatTensor(X2[:background_samples]),
# #     torch.FloatTensor(X2[:10]),
# #     feature_names2,
# #     model_name="Lab_Model"
# # )
#
#
# # 训练融合模型
# print("\nTraining HierarchicalAttentionFusion Model...")
# fusion_model = fusion_model = HierarchicalAttentionFusion(
#     model1,
#     model2,
#     feat_layers=['stage2', 'stage3'] , # 根据实际需要选择层
# ).to(device)
# optimizer = optim.AdamW(fusion_model.parameters(), lr=0.001)
# scheduler_fusion = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# dataset = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(X2), torch.FloatTensor(y))
# train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
# epoches = 20
# for epoch in range(epoches):
#     total_loss = 0
#     fusion_model.train()
#     for X1_batch, X2_batch, y_batch in train_loader:
#         X1_batch, X2_batch = X1_batch.to(device), X2_batch.to(device)
#         y_batch = y_batch.to(device)
#
#         optimizer.zero_grad()
#         probs = fusion_model(X1_batch, X2_batch)
#         loss = nn.BCELoss()(probs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     scheduler_fusion.step()
#     print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
#
# # 模型解释与评估
# print("\nFeature Importance for Model 1:")
# feature_importance(model1, torch.FloatTensor(X1[:1000]).to(device), feature_names1)
# print("\nFeature Importance for Model 2:")
# feature_importance(model2, torch.FloatTensor(X2[:1000]).to(device), feature_names2)
# evaluate_ensemble(fusion_model, X1, X2, y)
# # 训练融合模型
# print("\nTraining Fusion Model...")
# fusion_model = AttentionFusion(model1, model2).to(device)
# optimizer = optim.AdamW(fusion_model.parameters(), lr=0.001)
# scheduler_fusion = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# dataset = TensorDataset(torch.FloatTensor(X1), torch.FloatTensor(X2), torch.FloatTensor(y))
# train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
# epoches = 20
# for epoch in range(epoches):
#     total_loss = 0
#     fusion_model.train()
#     for X1_batch, X2_batch, y_batch in train_loader:
#         X1_batch, X2_batch = X1_batch.to(device), X2_batch.to(device)
#         y_batch = y_batch.to(device)
#
#         optimizer.zero_grad()
#         probs = fusion_model(X1_batch, X2_batch)
#         loss = nn.BCELoss()(probs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     scheduler_fusion.step()
#     print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
#
# # 模型解释与评估
# print("\nFeature Importance for Model 1:")
# feature_importance(model1, torch.FloatTensor(X1[:1000]).to(device), feature_names1)
# print("\nFeature Importance for Model 2:")
# feature_importance(model2, torch.FloatTensor(X2[:1000]).to(device), feature_names2)
# evaluate_ensemble(fusion_model, X1, X2, y)

