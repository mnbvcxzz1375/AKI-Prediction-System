import psycopg2
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# # ================== 数据库配置部分 ==================
# # 数据库连接配置
# con = psycopg2.connect(
#     database="mimiciii",
#     user="postgres",
#     password="123456",
#     host="localhost",
#     port="5433"
# )
#
# # ================== 修改后的SQL查询 ==================
# # 新的阳性样本查询
# query_positive = '''
# SELECT DISTINCT lfd.*,
#                 p.gender,
#                 1 AS match_flag
# FROM labs_first_day lfd
# LEFT JOIN diagnoses_icd d ON d.hadm_id = lfd.hadm_id
# LEFT JOIN d_icd_diagnoses di ON di.icd9_code = d.icd9_code
# LEFT JOIN patients p ON p.subject_id = d.subject_id
# WHERE (di.long_title ILIKE '% renal fail%'
#     OR di.long_title ILIKE '%kidney fail%'
#     OR di.long_title ILIKE '%liver fail%'
#     OR di.long_title ILIKE '%spleen fail%')
# '''
#
# # 新的阴性样本查询（动态匹配阳性样本数量）
# query_negative = '''
# WITH positive_hadm AS (
#     SELECT DISTINCT d.hadm_id
#     FROM diagnoses_icd d
#     JOIN d_icd_diagnoses di ON di.icd9_code = d.icd9_code
#     WHERE di.long_title ILIKE '% renal fail%'
#         OR di.long_title ILIKE '%kidney fail%'
#         OR di.long_title ILIKE '%liver fail%'
#         OR di.long_title ILIKE '%spleen fail%'
# )
# SELECT
#     lfd.*,
#     p.gender,
#     0 AS match_flag
# FROM labs_first_day lfd
# LEFT JOIN patients p ON p.subject_id = lfd.subject_id
# WHERE NOT EXISTS (
#     SELECT 1
#     FROM positive_hadm ph
#     WHERE ph.hadm_id = lfd.hadm_id
# )
# ORDER BY RANDOM()
# LIMIT (SELECT COUNT(*) FROM positive_hadm);
# '''
#
#
# ================== 数据获取部分 ==================
# def get_dynamic_keep_columns(cursor_description):
#     """
#     从数据库查询结果中动态提取列名，并确保保留关键列。
#     :param cursor_description: 数据库查询结果的description属性
#     :return: 动态生成的keep_columns列表
#     """
#     # 提取所有列名
#     all_columns = [desc[0] for desc in cursor_description]
#
#     # 确保关键列存在
#     required_columns = ['hadm_id', 'match_flag']
#     for col in required_columns:
#         if col not in all_columns:
#             raise ValueError(f"关键列 {col} 不在查询结果中！")
#
#     # 动态构建keep_columns
#     # 排除不需要的列（可根据需要调整）
#     exclude_columns = ['row_id', 'subject_id', 'icustay_id']  # 示例：排除这些列
#     keep_columns = [col for col in all_columns if col not in exclude_columns]
#
#     return keep_columns
#
#
# try:
#     cur = con.cursor()
#
#     # 获取阳性样本
#     print("正在获取阳性样本...")
#     cur.execute(query_positive)
#     positive_columns = [desc[0] for desc in cur.description]  # 提取列名
#     positive_df = pd.DataFrame(cur.fetchall(), columns=positive_columns)
#     print(f"阳性样本获取完成，共 {len(positive_df)} 条")
#
#     # 动态更新阴性样本查询
#     query_negative = query_negative.replace('SELECT COUNT(*) FROM positive_hadm', str(len(positive_df)))
#
#     # 获取阴性样本
#     print("正在获取阴性样本...")
#     cur.execute(query_negative)
#     negative_columns = [desc[0] for desc in cur.description]  # 提取列名
#     negative_df = pd.DataFrame(cur.fetchall(), columns=negative_columns)
#     print(f"阴性样本获取完成，共 {len(negative_df)} 条")
#
#     # 合并数据
#     combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
#
#     # 动态生成keep_columns
#     keep_columns = get_dynamic_keep_columns(cur.description)
#     print(f"动态保留的列：{keep_columns}")
#
#     # 仅保留需要的列
#     combined_df = combined_df[keep_columns]
#
#     # 保存原始数据
#     raw_output = "./data/labs_first_day_raw.csv"
#     combined_df.to_csv(raw_output, index=False)
#     print(f"原始数据已保存到 {raw_output}")
#
# except Exception as e:
#     print(f"数据库操作失败：{str(e)}")
# finally:
#     if con:
#         cur.close()
#         con.close()
#
# ================== 数据预处理部分 ==================
# 读取数据
df = pd.read_csv('./data/labs_first_day_raw.csv')

# 1. 动态保留列（基于查询结果）
keep_columns = get_dynamic_keep_columns([(col,) for col in df.columns])  # 模拟cursor.description
df = df[keep_columns]

# 2. 去除重复记录（基于hadm_id）
df = df.drop_duplicates(subset=['hadm_id'], keep='first')

# 3. 性别编码
df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1).astype(int)

# 4. 缺失值处理（保留hadm_id）
impute_cols = [col for col in df.columns if col not in ['hadm_id', 'match_flag']]  # 动态选择需要填补的列
imputer = IterativeImputer(
    estimator=LGBMRegressor(n_estimators=50, random_state=42),
    max_iter=10,
    random_state=42
)
df[impute_cols] = imputer.fit_transform(df[impute_cols])

# 5. 标准化处理（排除hadm_id和标签列）
features = df.drop(columns=['hadm_id', 'match_flag'])
scaler = StandardScaler().fit(features)
features_scaled = MinMaxScaler().fit_transform(scaler.transform(features))

# 重组最终DataFrame
df_final = pd.DataFrame(features_scaled, columns=features.columns)
df_final = pd.concat([
    df[['hadm_id']].reset_index(drop=True),
    df_final,
    df['match_flag'].reset_index(drop=True)
], axis=1)

# 保存最终数据
final_output = './data/cleaned_labs_first_day_lgbm.csv'
df_final.to_csv(final_output, index=False)
print(f"处理后的数据已保存到 {final_output}")
print("数据特征分布：\n", df_final.describe())

# ================== 新增验证部分 ==================
# 验证hadm_id保留情况
assert 'hadm_id' in df_final.columns, "hadm_id列丢失！"
print("\n验证结果：")
print(f"总样本数：{len(df_final)}")
print(f"阳性样本数：{df_final.match_flag.sum()}")
print(f"阴性样本数：{len(df_final) - df_final.match_flag.sum()}")
print(f"唯一hadm_id数量：{df_final.hadm_id.nunique()}")