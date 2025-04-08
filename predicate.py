import torch
from torch import nn
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             roc_curve)
import torch.optim as optim
from torch.utils.data import (DataLoader, TensorDataset,
                              WeightedRandomSampler)
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from tqdm import tqdm


# --------------------------
# 模型定义（必须与训练时结构一致）
# --------------------------

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
        self.augment = Compose([TemporalAugmentation(sigma=0.05, p=0.3)])

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


class AKIVisualizer:
    """可视化引擎"""
    COLOR_MAP = {
        "non-aki": "#2ecc71",
        "low": "#f1c40f",
        "mid": "#e67e22",
        "high": "#e74c3c"
    }

    @staticmethod
    def plot_risk_distribution(risk_levels, width=400, height=400):
        """风险等级分布饼图"""
        counts = pd.Series(risk_levels).value_counts().reindex(["non-aki", "low", "mid", "high"], fill_value=0)
        fig = px.pie(
            names=counts.index,
            values=counts.values,
            color=counts.index,
            color_discrete_map=AKIVisualizer.COLOR_MAP,
            hole=0.4,
            width=width,
            height=height
        )
        fig.update_layout(
            title="AKI风险等级分布",
            showlegend=False,
            margin=dict(t=40, b=10),
            annotations=[dict(text=f'总计<br>{len(risk_levels)}', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        return fig

    @staticmethod
    def plot_probabilities(probs, threshold=0.5, width=800, height=400):
        """概率条形图"""
        df = pd.DataFrame({
            "病例编号": range(1, len(probs) + 1),
            "AKI概率": probs.flatten(),
            "风险等级": [AKIPredictor._map_prob_to_level(p) for p in probs]
        })
        fig = px.bar(
            df,
            x="AKI概率",
            y="病例编号",
            orientation='h',
            color="风险等级",
            color_discrete_map=AKIVisualizer.COLOR_MAP,
            range_x=[0, 1],
            width=width,
            height=height
        )
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            annotation_text=f"预警阈值 ({threshold * 100}%)",
            annotation_position="top right"
        )
        fig.update_layout(
            title="病例AKI风险概率排序",
            yaxis=dict(title="病例编号", type='category'),
            xaxis=dict(title="AKI发生概率", tickformat=".0%"),
            hovermode="y unified"
        )
        return fig

    @staticmethod
    def plot_gauge(prob, width=300, height=300):
        """单病例风险仪表盘"""
        level = AKIPredictor._map_prob_to_level(prob)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'font': {'size': 24}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': AKIVisualizer.COLOR_MAP[level]},
                'steps': [
                    {'range': [0, 20], 'color': AKIVisualizer.COLOR_MAP["non-aki"]},
                    {'range': [20, 50], 'color': AKIVisualizer.COLOR_MAP["low"]},
                    {'range': [50, 80], 'color': AKIVisualizer.COLOR_MAP["mid"]},
                    {'range': [80, 100], 'color': AKIVisualizer.COLOR_MAP["high"]}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        fig.update_layout(
            title=f"AKI风险等级：{level.upper()}",
            width=width,
            height=height,
            margin=dict(t=60, b=10)
        )
        return fig

    @staticmethod
    def plot_feature_importance(predictor, micro_df, lab_df, top_n=10):
        """特征重要性分析（示例）"""
        # 模拟特征重要性数据
        micro_importance = np.random.randn(len(predictor.micro_features))
        lab_importance = np.random.randn(len(predictor.lab_features))
        micro_fig = px.bar(
            x=predictor.micro_features,
            y=micro_importance,
            title="微生物特征重要性",
            labels={'x': '特征', 'y': '重要性'}
        )
        lab_fig = px.bar(
            x=predictor.lab_features,
            y=lab_importance,
            title="检验指标特征重要性",
            labels={'x': '特征', 'y': '重要性'}
        )
        return micro_fig, lab_fig

    @staticmethod
    def plot_probability_histogram(probs, width=600, height=400):
        """AKI概率直方图"""
        df = pd.DataFrame({"AKI概率": probs.flatten()})
        fig = px.histogram(
            df, x="AKI概率", nbins=20,
            title="AKI概率直方图",
            color_discrete_sequence=["#3498db"],
            width=width, height=height
        )
        fig.update_layout(
            xaxis_title="AKI概率",
            yaxis_title="样本数"
        )
        return fig

    @staticmethod
    def plot_lab_boxplot(lab_df, width=600, height=400):
        """检验指标箱线图"""
        lab_long = lab_df.melt(var_name="指标", value_name="值")
        fig = px.box(
            lab_long, x="指标", y="值",
            title="检验指标分布箱线图",
            color="指标",
            width=width, height=height
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    @staticmethod
    def plot_micro_scatter(micro_df, feature_x, feature_y, width=600, height=400):
        """指定两个微生物特征散点图"""
        fig = px.scatter(
            micro_df, x=feature_x, y=feature_y,
            title=f"{feature_x} vs {feature_y} 散点图",
            color=feature_x,
            width=width, height=height
        )
        return fig

    @staticmethod
    def plot_lab_correlation_heatmap(lab_df, width=600, height=400):
        """检验指标相关性热图"""
        corr = lab_df.corr()
        fig = px.imshow(
            corr, text_auto=True,
            title="检验指标相关性热图",
            width=width, height=height,
            color_continuous_scale="Viridis"
        )
        return fig

    @staticmethod
    def plot_lab_pairplot(lab_df, width=800, height=800):
        """检验指标对角散点图矩阵"""
        fig = px.scatter_matrix(
            lab_df,
            title="检验指标对角散点图矩阵",
            width=width, height=height,
            color=lab_df.columns[0] if lab_df.columns.size > 0 else None
        )
        return fig

    @staticmethod
    def plot_lab_violin(lab_df, width=800, height=400):
        """检验指标小提琴图"""
        lab_long = lab_df.melt(var_name="指标", value_name="值")
        fig = px.violin(
            lab_long, x="指标", y="值", box=True, points="all",
            title="检验指标小提琴图",
            width=width, height=height,
            color="指标"
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    @staticmethod
    def plot_micro_density_contour(micro_df, feature_x, feature_y, width=600, height=400):
        """微生物数据密度轮廓图"""
        fig = px.density_contour(
            micro_df, x=feature_x, y=feature_y,
            title=f"{feature_x} vs {feature_y} 密度轮廓图",
            width=width, height=height,
            color_continuous_scale="Blues"
        )
        return fig

    @staticmethod
    def plot_parallel_coordinates(lab_df, width=800, height=400):
        """平行坐标图展示检验指标样本分布"""
        fig = px.parallel_coordinates(
            lab_df,
            title="检验指标平行坐标图",
            color=lab_df.iloc[:, 0],
            width=width, height=height
        )
        return fig

    @staticmethod
    def plot_radar_chart(probs, width=600, height=400):
        """雷达图：不同风险等级的平均概率分布"""
        df = pd.DataFrame({
            "风险等级": [AKIPredictor._map_prob_to_level(p) for p in probs],
            "AKI概率": probs.flatten()
        })
        # 计算各风险等级的平均概率
        avg_df = df.groupby("风险等级").mean().reset_index()
        # 保证顺序
        avg_df["风险等级"] = pd.Categorical(avg_df["风险等级"], categories=["non-aki", "low", "mid", "high"], ordered=True)
        avg_df = avg_df.sort_values("风险等级")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avg_df["AKI概率"],
            theta=avg_df["风险等级"],
            fill='toself',
            name="平均概率"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title="各风险等级平均AKI概率雷达图",
            width=width, height=height
        )
        return fig

    @staticmethod
    def plot_sunburst_chart(lab_df, width=600, height=400):
        """Sunburst图：分层展示检验指标（示例）"""
        # 示例：假设lab_df中有若干指标，根据指标名称首字母进行分层
        lab_long = lab_df.melt(var_name="指标", value_name="值")
        lab_long["类别"] = lab_long["指标"].str[0]
        fig = px.sunburst(
            lab_long,
            path=["类别", "指标"],
            values="值",
            title="检验指标Sunburst分层图",
            width=width, height=height
        )
        return fig

# 新增数据验证方法
def validate_input(lab_df, micro_df):
    """验证输入数据格式"""
    required_lab_columns = [
        'aniongap_min', 'aniongap_max', 'chloride_min', 'chloride_max', 
        'potassium_min', 'potassium_max', 'sodium_min', 'sodium_max',
        'bicarbonate_min', 'bicarbonate_max', 'creatinine_min', 'creatinine_max',
        'bun_min', 'bun_max', 'albumin_min', 'albumin_max', 'bilirubin_min',
        'bilirubin_max', 'glucose_min', 'glucose_max', 'lactate_min', 'lactate_max',
        'hematocrit_min', 'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max',
        'platelet_min', 'platelet_max', 'wbc_min', 'wbc_max', 'ptt_min', 'ptt_max',
        'inr_min', 'inr_max', 'pt_min', 'pt_max', 'bands_min', 'bands_max', 'gender'
    ]
    
    required_micro_columns = [
        'spec_itemid', 'org_itemid', 'isolate_num', 'ab_itemid',
        'dilution_text', 'dilution_value', 'urineoutput'
    ]
    
    missing_lab = set(required_lab_columns) - set(lab_df.columns)
    missing_micro = set(required_micro_columns) - set(micro_df.columns)
    
    if missing_lab:
        raise ValueError(f"缺少实验室字段: {missing_lab}")
    if missing_micro:
        raise ValueError(f"缺少微生物字段: {missing_micro}")

# --------------------------
# 增强版预测接口
# --------------------------
class AKIPredictor:
    def __init__(self, model_dir="./models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.micro_features = pd.read_csv('./data/cleaned_microbiologyevents_plus.csv', nrows=0).drop(
            columns=['hadm_id', 'match_flag']).columns.tolist()
        self.lab_features = pd.read_csv('./data/cleaned_labs_first_day_lgbm.csv', nrows=0).drop(
            columns=['hadm_id', 'match_flag']).columns.tolist()
        self.micro_imputer = joblib.load(f"{model_dir}/microbiologyevents_plus_lgbm.pkl")
        self.lab_imputer = joblib.load(f"{model_dir}/labs_first_day_lgbm.pkl")
        self.micro_scaler = joblib.load(f"{model_dir}/microbiologyevents_plus_standard_scaler.pkl")
        self.lab_scaler = joblib.load(f"{model_dir}/labs_first_day_lgbm_standard_scaler.pkl")
        self._load_models(model_dir)

    def _load_models(self, model_dir):
        self.model1 = DynamicCNN(len(self.micro_features)).to(self.device)
        self.model2 = DynamicCNN(len(self.lab_features)).to(self.device)
        self.fusion_model = AttentionFusion(self.model1, self.model2).to(self.device)
        self.model1.load_state_dict(torch.load(f"{model_dir}/model1.pth", map_location=self.device))
        self.model2.load_state_dict(torch.load(f"{model_dir}/model2.pth", map_location=self.device))
        self.fusion_model.load_state_dict(torch.load(f"{model_dir}/fusion_model.pth", map_location=self.device))
        self.model1.eval()
        self.model2.eval()
        self.fusion_model.eval()

    def _preprocess(self, micro_df, lab_df):
        self._validate_columns(micro_df, lab_df)
        micro_data = micro_df.drop(columns=['hadm_id'], errors='ignore')
        lab_data = lab_df.drop(columns=['hadm_id'], errors='ignore')
        micro_filled = pd.DataFrame(self.micro_imputer.transform(micro_data), columns=self.micro_features)
        lab_filled = pd.DataFrame(self.lab_imputer.transform(lab_data), columns=self.lab_features)
        micro_scaled = self.micro_scaler.transform(micro_filled)
        lab_scaled = self.lab_scaler.transform(lab_filled)
        return (
            torch.FloatTensor(micro_scaled).to(self.device),
            torch.FloatTensor(lab_scaled).to(self.device)
        )

    def _validate_columns(self, micro_df, lab_df):
        micro_cols = set(micro_df.columns) - {'hadm_id'}
        lab_cols = set(lab_df.columns) - {'hadm_id'}
        if micro_cols != set(self.micro_features):
            raise ValueError(f"微生物数据特征不匹配，需要：{self.micro_features}")
        if lab_cols != set(self.lab_features):
            raise ValueError(f"检验数据特征不匹配，需要：{self.lab_features}")

    def predict_proba(self, micro_df, lab_df):
        with torch.no_grad():
            x1, x2 = self._preprocess(micro_df, lab_df)
            probs = self.fusion_model(x1, x2).cpu().numpy()
        return probs

    def predict_risk_level(self, micro_df, lab_df):
        probs = self.predict_proba(micro_df, lab_df)
        return [self._map_prob_to_level(p) for p in probs]

    @staticmethod
    def _map_prob_to_level(prob):
        if prob < 0.2:
            return "non-aki"
        elif 0.2 <= prob < 0.5:
            return "low"
        elif 0.5 <= prob < 0.8:
            return "mid"
        else:
            return "high"

    def generate_dashboard(self, micro_df, lab_df):
        probs = self.predict_proba(micro_df, lab_df)
        risks = self.predict_risk_level(micro_df, lab_df)
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "pie"}, {"type": "xy"}],
                [{"type": "domain"}, {"type": "xy"}]
            ],
            subplot_titles=(
                "风险等级分布",
                "病例风险概率排序",
                "当前病例风险仪表",
                "特征重要性分析"
            )
        )
        risk_dist = AKIVisualizer.plot_risk_distribution(risks)
        prob_bar = AKIVisualizer.plot_probabilities(probs)
        gauge = AKIVisualizer.plot_gauge(probs[0])
        micro_fig, lab_fig = AKIVisualizer.plot_feature_importance(self, micro_df, lab_df)
        fig.add_trace(risk_dist.data[0], row=1, col=1)
        for trace in prob_bar.data:
            fig.add_trace(trace, row=1, col=2)
        fig.add_trace(gauge.data[0], row=2, col=1)
        for trace in micro_fig.data:
            fig.add_trace(trace, row=2, col=2)
        fig.update_layout(
            height=800,
            showlegend=False,
            margin=dict(t=80),
            title_text="AKI风险预测综合看板"
        )
        return fig

# 输出目录（需要与前端项目静态资源目录一致，这里示例使用绝对路径或相对路径）
output_dir = "./web/web-shower/public/assert/html/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


@staticmethod
def generate_super_dashboard(predictor, micro_df, lab_df):
    """生成超级仪表板，包含更多精美页面"""
    probs = predictor.predict_proba(micro_df, lab_df)
    risks = predictor.predict_risk_level(micro_df, lab_df)

    # 修改后的子图配置，移除可能导致冲突的splom图表
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"type": "domain"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=[
            "风险等级分布", "病例风险概率排序", "AKI概率直方图",
            "检验指标散点图", "检验指标小提琴图", "平行坐标图",  # 将原splom标题改为散点图
            "当前病例风险仪表", "各风险等级平均概率雷达图", "检验指标Sunburst图"
        ]
    )

    # 左上：饼图
    risk_dist = AKIVisualizer.plot_risk_distribution(risks, width=300, height=300)
    # 上中：条形图
    prob_bar = AKIVisualizer.plot_probabilities(probs, width=300, height=300)
    # 上右：直方图
    prob_hist = AKIVisualizer.plot_probability_histogram(probs, width=300, height=300)
    # 中左：替换为普通散点图或热力图（示例使用散点图）
    lab_scatter = AKIVisualizer.plot_lab_scatter(lab_df, 'feature1', 'feature2', width=300, height=300)  # 假设存在该函数
    # 中中：检验指标小提琴图
    lab_violin = AKIVisualizer.plot_lab_violin(lab_df, width=300, height=300)
    # 中右：平行坐标图
    lab_parallel = AKIVisualizer.plot_parallel_coordinates(lab_df, width=300, height=300)
    # 左下：仪表盘
    gauge = AKIVisualizer.plot_gauge(probs[0], width=300, height=300)
    # 右下：雷达图
    radar = AKIVisualizer.plot_radar_chart(probs, width=300, height=300)
    # 下中：Sunburst图
    sunburst = AKIVisualizer.plot_sunburst_chart(lab_df, width=300, height=300)

    # 添加各图表到仪表板
    fig.add_trace(risk_dist.data[0], row=1, col=1)
    for trace in prob_bar.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in prob_hist.data:
        fig.add_trace(trace, row=1, col=3)
    # 添加散点图到中左位置
    for trace in lab_scatter.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in lab_violin.data:
        fig.add_trace(trace, row=2, col=2)
    for trace in lab_parallel.data:
        fig.add_trace(trace, row=2, col=3)
    fig.add_trace(gauge.data[0], row=3, col=1)
    for trace in sunburst.data:
        fig.add_trace(trace, row=3, col=2)
    for trace in radar.data:
        fig.add_trace(trace, row=3, col=3)

    fig.update_layout(
        height=1000,
        showlegend=False,
        margin=dict(t=100),
        title_text="超级版AKI风险预测综合看板"
    )
    return fig


def generate_charts(micro_df, lab_df):
    
    validate_input(lab_df, micro_df)
    # 初始化预测器（确保模型、数据文件存在）
    predictor = AKIPredictor()

    # 基础图表
    risks = predictor.predict_risk_level(micro_df, lab_df)
    probs = predictor.predict_proba(micro_df, lab_df)

    risk_dist = AKIVisualizer.plot_risk_distribution(risks)
    prob_chart = AKIVisualizer.plot_probabilities(probs)
    gauge = AKIVisualizer.plot_gauge(probs[0])
    prob_hist = AKIVisualizer.plot_probability_histogram(probs)
    lab_box = AKIVisualizer.plot_lab_boxplot(new_lab)

    # 其它图表
    # 散点图（示例使用 feature1 与 feature2，如数据中不存在，请替换成真实特征名）
    if 'feature1' in new_micro.columns and 'feature2' in new_micro.columns:
        micro_scatter = AKIVisualizer.plot_micro_scatter(new_micro, 'feature1', 'feature2')
        micro_scatter.write_html(os.path.join(output_dir, "micro_scatter.html"))
        micro_density = AKIVisualizer.plot_micro_density_contour(new_micro, 'feature1', 'feature2')
        micro_density.write_html(os.path.join(output_dir, "micro_density.html"))

    lab_corr = AKIVisualizer.plot_lab_correlation_heatmap(new_lab)
    lab_pair = AKIVisualizer.plot_lab_pairplot(new_lab)
    lab_violin = AKIVisualizer.plot_lab_violin(new_lab)
    lab_parallel = AKIVisualizer.plot_parallel_coordinates(new_lab)
    radar = AKIVisualizer.plot_radar_chart(probs)
    sunburst = AKIVisualizer.plot_sunburst_chart(new_lab)

    # 扩展仪表板与超级仪表板（集成多个图表在一个页面）
    # extra_dashboard = AKIVisualizer.generate_extra_dashboard(predictor, new_micro, new_lab)
    # super_dashboard = AKIVisualizer.generate_super_dashboard(predictor, new_micro, new_lab)

    # 保存各图表为 HTML 文件
    risk_dist.write_html(os.path.join(output_dir, "risk_dist.html"))
    prob_chart.write_html(os.path.join(output_dir, "prob_chart.html"))
    gauge.write_html(os.path.join(output_dir, "gauge.html"))
    prob_hist.write_html(os.path.join(output_dir, "prob_histogram.html"))
    lab_box.write_html(os.path.join(output_dir, "lab_boxplot.html"))
    lab_corr.write_html(os.path.join(output_dir, "lab_correlation.html"))
    lab_pair.write_html(os.path.join(output_dir, "lab_pairplot.html"))
    lab_violin.write_html(os.path.join(output_dir, "lab_violin.html"))
    lab_parallel.write_html(os.path.join(output_dir, "lab_parallel.html"))
    radar.write_html(os.path.join(output_dir, "radar.html"))
    sunburst.write_html(os.path.join(output_dir, "sunburst.html"))
    # extra_dashboard.write_html(os.path.join(output_dir, "extra_dashboard.html"))
    # super_dashboard.write_html(os.path.join(output_dir, "super_dashboard.html"))

    print("所有图表已生成并保存在指定目录中。")


if __name__ == "__main__":
    generate_charts()