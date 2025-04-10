import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from utils import Logger
from lightgbm import log_evaluation, early_stopping
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

seed = 42
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 预测')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'lightgbm 预测')

def evaluate_model(df_train_features, models_with_user, models_without_user, feature_cols_with_user, feature_cols_without_user, classify_model_with_user, classify_model_without_user, test_size=0.2, random_state=42, threshold=0.8):
    """
    从训练集中抽取一部分数据作为验证集，评估模型性能。

    Args:
        df_train_features (pd.DataFrame): 训练数据
        models_with_user (dict): 模型 1（包含用户特征），键为目标变量名，值为 LGBMRegressor 模型
        models_without_user (dict): 模型 2（不包含用户特征），键为目标变量名，值为 LGBMRegressor 模型
        feature_cols_with_user (list): 模型 1 的特征列
        feature_cols_without_user (list): 模型 2 的特征列
        target_cols (list): 目标列
        test_size (float): 验证集比例，默认为 0.2
        random_state (int): 随机种子，默认为 42

    Returns:
        dict: 包含验证集得分和详细结果的字典
    """
    df_train_features['mask'] = 1

    # 随机选择 10% 的行
    num_rows = df_train_features.shape[0]  # 获取行数
    num_mask_zero_rows = int(num_rows * 0.1)  # 计算需要设置为 0 的行数

    # 随机选择 10% 的行索引
    random_rows = np.random.choice(df_train_features.index, size=num_mask_zero_rows, replace=False)

    # 将随机选择的行的 'mask' 列设置为 0
    df_train_features.loc[random_rows, 'mask'] = 0


    df_valid = df_train_features
    # 抽取验证集
    # df_train, df_valid = train_test_split(df_train_features, test_size=test_size, random_state=random_state)
    # print(f"验证集样本数：{len(df_valid)}，训练集剩余样本数：{len(df_train)}")

    # 预测验证集
    df_valid.loc[df_valid['mask'] == 1, 'forward_pre'] = df_valid['user_median_forward_count'].round().astype(int)
    df_valid.loc[df_valid['mask'] == 1, 'comment_pre'] = df_valid['user_median_comment_count'].round().astype(int)
    df_valid.loc[df_valid['mask'] == 1, 'like_pre'] = df_valid['user_median_like_count'].round().astype(int)
    df_valid.loc[df_valid['mask'] == 0, 'forward_pre'] = int(0)
    df_valid.loc[df_valid['mask'] == 0, 'comment_pre'] = int(0)
    df_valid.loc[df_valid['mask'] == 0, 'like_pre'] = int(0)

    df_valid['forward_pre'] = df_valid['forward_pre'].astype(int)
    df_valid['comment_pre'] = df_valid['comment_pre'].astype(int)
    df_valid['like_pre'] = df_valid['like_pre'].astype(int)

    df_valid['is_high_interaction'] = 0

    df_valid.loc[df_valid['mask'] == 1, 'is_high_interaction'] = classify_model_with_user.predict(df_valid.loc[df_valid['mask'] == 1, feature_cols_with_user])
    df_valid.loc[df_valid['mask'] == 0, 'is_high_interaction'] = classify_model_without_user.predict(df_valid.loc[df_valid['mask'] == 0, feature_cols_without_user])

    df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 1), 'forward_pre'] = models_with_user['forward_count'].predict(df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 1), feature_cols_with_user]).round().astype(int)
    df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 1), 'comment_pre'] = models_with_user['comment_count'].predict(df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 1), feature_cols_with_user]).round().astype(int)
    df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 1), 'like_pre'] = models_with_user['like_count'].predict(df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 1), feature_cols_with_user]).round().astype(int)

    df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 0), 'forward_pre'] = models_without_user['forward_count'].predict(df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 0), feature_cols_without_user]).round().astype(int)
    df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 0), 'comment_pre'] = models_without_user['comment_count'].predict(df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 0), feature_cols_without_user]).round().astype(int)
    df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 0), 'like_pre'] = models_without_user['like_count'].predict(df_valid.loc[(df_valid['is_high_interaction'] >= threshold) & (df_valid['mask'] == 0), feature_cols_without_user]).round().astype(int)
    
    # 计算 deviation_f
    df_valid['deviation_f'] = abs(df_valid['forward_pre'] - df_valid['forward_count']) / (df_valid['forward_count'] + 5)

    # 计算 deviation_c
    df_valid['deviation_c'] = abs(df_valid['comment_pre'] - df_valid['comment_count']) / (df_valid['forward_count'] + 3)

    # 计算 deviation_l
    df_valid['deviation_l'] = abs(df_valid['like_pre'] - df_valid['like_count']) / (df_valid['like_count'] + 3)

    # 计算综合 deviation_i
    df_valid['deviation_i'] = 1 - 0.5 * df_valid['deviation_f'] - 0.25 * df_valid['deviation_c'] - 0.25 * df_valid['deviation_l']

    # 计算 count_i（转发 + 评论 + 赞，总数不超过 100）
    df_valid['count_i'] = df_valid[['forward_count', 'comment_count', 'like_count']].sum(axis=1).clip(upper=100)

    # 计算改进的符号函数 sgn(x)
    df_valid['sgn'] = (df_valid['deviation_i'] > 0.8).astype(int)

    # 计算 precision
    numerator = np.sum((df_valid['count_i'] + 1) * df_valid['sgn'])
    denominator = np.sum(df_valid['count_i'] + 1)
    precision = numerator / denominator * 100
    log.info(f"预测精度为{precision}%")


def define_feature_columns():
    """
    定义两组特征列：
    - feature_cols_with_user: 包含用户特征的所有特征（模型 1）
    - feature_cols_without_user: 不包含用户特征（模型 2）
    
    Returns:
        tuple: (feature_cols_with_user, feature_cols_without_user, target_cols)
    """
    # 用户特征
    user_features = [
        'user_mean_forward_count', 'user_max_forward_count', 'user_median_forward_count', 'user_std_forward_count',
        'user_mean_comment_count', 'user_max_comment_count', 'user_median_comment_count', 'user_std_comment_count',
        'user_mean_like_count', 'user_max_like_count', 'user_median_like_count', 'user_std_like_count',
        'post_count',
        'user_forward_quantile_95', 'user_comment_quantile_95', 'user_like_quantile_95',
        'user_nonzero_forward_ratio', 'user_nonzero_comment_ratio', 'user_nonzero_like_ratio',
        'user_interaction_score'
    ]

    # 博文特征
    post_features = [
        'day_of_week', 'hour', 'is_weekend', 'time_since_last_post',
        'content_length', 'has_topic', 'has_mention', 'has_lottery', 'has_link',
        'bert_0', 'bert_1', 'bert_2', 'bert_3', 'bert_4', 'bert_5', 'bert_6', 'bert_7', 'bert_8', 'bert_9',
        'bert_10', 'bert_11', 'bert_12', 'bert_13', 'bert_14', 'bert_15', 'bert_16', 'bert_17', 'bert_18', 'bert_19'
    ]

    # 模型 1：包含用户特征的所有特征
    feature_cols_with_user = user_features + post_features
    # 模型 2：不包含用户特征
    feature_cols_without_user = post_features
    # 目标列
    target_cols = ['forward_count', 'comment_count', 'like_count']

    return feature_cols_with_user, feature_cols_without_user, target_cols


def mark_high_interaction_posts(df, forward_threshold=15, comment_threshold=12, like_threshold=15):
    """
    标记高互动博文。
    如果 forward_count, comment_count, like_count 都低于 threshold，则为低互动博文。
    
    Args:
        df (pd.DataFrame): 数据集
        threshold (int): 互动阈值，默认为 5(百分之94的位置，百分之95分位的三个值分别是7，5，6)
    
    Returns:
        pd.DataFrame: 包含 is_high_interaction 列的数据集
    """
    df['is_high_interaction'] = ((df['forward_count'] >= forward_threshold) | 
                                  (df['comment_count'] >= comment_threshold) | 
                                  (df['like_count'] >= like_threshold)).astype(int)
    
    # high_interaction_uids = df.loc[
    #     (df['forward_count'] >= 50) & 
    #     (df['comment_count'] >= 25) & 
    #     (df['like_count'] >= 50),
    #     'uid'
    # ].unique()
    # df['is_high_interaction'] = df['uid'].isin(high_interaction_uids).astype(int)
    
    # 计算 is_high_interaction 为 1 的行数
    high_interaction_count = df['is_high_interaction'].sum()
    # 计算总行数
    total_rows = len(df)
    # 计算占比
    proportion = high_interaction_count / total_rows
    log.info(f"高互动用户数为: {high_interaction_count}")
    log.info(f'高互动用户数占比为: {proportion * 100:.2f}%')
    return df


def encode_uid(df):
    # 只对 "uid" 列进行编码
    label_encoder = LabelEncoder()
    df['uid'] = label_encoder.fit_transform(df['uid'])
    uid_map = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    return df, uid_map


def train_lgb_models(df_train_features, feature_cols_with_user, feature_cols_without_user, target_cols, classify_model_with_user, classify_model_without_user, n_splits=5):
    """
    训练两个 LightGBM 模型，使用提前停止和多折交叉验证，仅使用高互动博文。
    
    Args:
        df_train (pd.DataFrame): 训练数据
        feature_cols_with_user (list): 模型 1 的特征列（包含用户特征）
        feature_cols_without_user (list): 模型 2 的特征列（不包含用户特征）
        target_cols (list): 目标列
        n_splits (int): 交叉验证的折数，默认为 5
    
    Returns:
        tuple: (model_with_user, model_without_user)
    """
    # 筛选高互动博文
    df_train_, df_train_val = train_test_split(
        df_train_features, test_size=0.2, random_state=seed
    )
    df_train_high = df_train_features[df_train_features['is_high_interaction'] == 1]

    log.info(f"使用 {len(df_train_high)} 条高互动博文进行训练.")

    # 准备训练数据
    X_with_user = df_train_high[feature_cols_with_user]
    # X_without_user = df_train_features[feature_cols_without_user]
    X_without_user = df_train_high[feature_cols_without_user]
    y_with_user = df_train_high[target_cols]
    # y_without_user = df_train_features[target_cols]
    y_without_user = y_with_user

    # 初始化KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 训练模型 1（包含用户特征）...
    log.info('训练模型 1（包含用户特征）...')  
    lgb_params = {
        'num_leaves': 256,  
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'random_state': seed,
        'subsample': 0.8,
        'reg_alpha':0.5,  # L1 正则化参数
        'reg_lambda':0.5,  # L2 正则化参数
        'metric': None
    }

    models_with_user = {}
    for target in target_cols:
        log.info(f"训练目标变量：{target}")
        model = lgb.LGBMRegressor(**lgb_params)
        oof_predictions = np.zeros(len(X_with_user))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_with_user)):
            log.info(f"Fold {fold + 1}/{n_splits}")
            X_train_fold, X_val_fold = X_with_user.iloc[train_idx], X_with_user.iloc[val_idx]
            y_train_fold, y_val_fold = y_with_user[target].iloc[train_idx], y_with_user[target].iloc[val_idx]
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],  # 直接使用 eval_set
                eval_metric='rmse',
                callbacks=[
                    log_evaluation(100),  # 每 100 轮打印一次日志
                    early_stopping(100)   # 早停策略，连续 100 轮未提升则停止训练
                ])
            oof_predictions[val_idx] = model.predict(X_val_fold)

        models_with_user[target] = model

    # 模型 2：不包含用户特征
    log.info("训练模型 2（不包含用户特征）...") 
    models_without_user = {}
    for target in target_cols:
        log.info(f"训练目标变量：{target}")
        model = lgb.LGBMRegressor(**lgb_params)
        oof_predictions = np.zeros(len(X_without_user))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_without_user)):
            log.info(f"Fold {fold + 1}/{n_splits}")
            X_train_fold, X_val_fold = X_without_user.iloc[train_idx], X_without_user.iloc[val_idx]
            y_train_fold, y_val_fold = y_without_user[target].iloc[train_idx], y_without_user[target].iloc[val_idx]

            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],  # 直接使用 eval_set
                eval_metric='rmse',
                callbacks=[
                    log_evaluation(100),  # 每 100 轮打印一次日志
                    early_stopping(10)   # 早停策略，连续 100 轮未提升则停止训练
                ])
            oof_predictions[val_idx] = model.predict(X_val_fold)
        models_without_user[target] = model

    # 查看特征重要性（以第一个目标变量为例）
    feature_importances_with_user = pd.DataFrame({
        'feature': feature_cols_with_user,
        'importance': models_with_user[target_cols[0]].feature_importances_
    })
    log.debug("模型 1 特征重要性（forward_count）：")
    log.debug(feature_importances_with_user.sort_values(by='importance', ascending=False))

    feature_importances_without_user = pd.DataFrame({
        'feature': feature_cols_without_user,
        'importance': models_without_user[target_cols[0]].feature_importances_
    })
    log.debug("模型 2 特征重要性（forward_count）：")
    log.debug(feature_importances_without_user.sort_values(by='importance', ascending=False))

    # 评估模型效果
    evaluate_model(df_train_val, models_with_user, models_without_user, feature_cols_with_user, feature_cols_without_user, classify_model_with_user, classify_model_without_user, threshold=0.5)

    return models_with_user, models_without_user


def predict_result(df_test, feature_cols_with_user, feature_cols_without_user, models_with_user, models_without_user, classify_model_with_user, classify_model_without_user, threshold=0.8):
    log.info('开始预测测试集...')
    df_test['forward_pre'] = 0
    df_test['comment_pre'] = 0
    df_test['like_pre'] = 0

    # 见过的用户就用中位数
    df_test.loc[df_test['user_mean_comment_count'] != -1, 'forward_pre'] = df_test['user_median_forward_count'].round().astype(int)
    df_test.loc[df_test['user_mean_comment_count'] != -1, 'comment_pre'] = df_test['user_median_forward_count'].round().astype(int)
    df_test.loc[df_test['user_mean_comment_count'] != -1, 'like_pre'] = df_test['user_median_like_count'].round().astype(int)
    # 未见过的用户初始化为 0
    df_test.loc[df_test['user_mean_comment_count'] == -1, 'forward_pre'] = 0
    df_test.loc[df_test['user_mean_comment_count'] == -1, 'comment_pre'] = 0
    df_test.loc[df_test['user_mean_comment_count'] == -1, 'like_pre'] = 0

    df_test['forward_pre'] = df_test['forward_pre'].astype(int)
    df_test['comment_pre'] = df_test['comment_pre'].astype(int)
    df_test['like_pre'] = df_test['like_pre'].astype(int)

    df_test['is_high_interaction'] = 0

    # 见过的用户
    df_test.loc[df_test['user_mean_comment_count'] != -1, 'is_high_interaction'] = classify_model_with_user.predict(df_test.loc[df_test['user_mean_comment_count'] != -1, feature_cols_with_user])
    # 没见过的用户
    df_test.loc[df_test['user_mean_comment_count'] == -1, 'is_high_interaction'] = classify_model_without_user.predict(df_test.loc[df_test['user_mean_comment_count'] == -1, feature_cols_without_user])


    # 见过的用户
    df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] != -1), 'forward_pre'] = models_with_user['forward_count'].predict(df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] != -1), feature_cols_with_user]).round().astype(int)
    df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] != -1), 'comment_pre'] = models_with_user['comment_count'].predict(df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] != -1), feature_cols_with_user]).round().astype(int)
    df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] != -1), 'like_pre'] = models_with_user['like_count'].predict(df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] != -1), feature_cols_with_user]).round().astype(int)

    # 没见过的用户
    filtered_data = df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), feature_cols_with_user]
    if filtered_data.empty:
        log.warning("No data to predict for the filtered conditions")
    else:
        df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), 'forward_pre'] = models_without_user['forward_count'].predict(df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), feature_cols_without_user]).round().astype(int)
        df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), 'comment_pre'] = models_without_user['comment_count'].predict(df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), feature_cols_without_user]).round().astype(int)
        df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), 'like_pre'] = models_without_user['like_count'].predict(df_test.loc[(df_test['is_high_interaction'] >= threshold) & (df_test['user_mean_comment_count'] == -1), feature_cols_without_user]).round().astype(int)
        
    result_cols = ['uid', 'mid', 'forward_pre', 'comment_pre', 'like_pre']
    df_result = df_test[result_cols]
    df_result.to_pickle('../prediction_result/df_result.pkl')

    # 转换为指定格式
    log.info('预测完成，输出为指定格式...')
    # 确保转、评、赞的值为整数
    df_result['forward_pre'] = df_result['forward_pre'].astype(int)
    df_result['comment_pre'] = df_result['comment_pre'].astype(int)
    df_result['like_pre'] = df_result['like_pre'].astype(int)

    # 创建一个新的列，合并数据并按指定格式存储
    df_result['output'] = df_result['uid'].astype(str) + '\t' + df_result['mid'].astype(str) + '\t' + df_result['forward_pre'].astype(str) + ',' + df_result['comment_pre'].astype(str) + ',' + df_result['like_pre'].astype(str)

    # 保存为 .txt 文件，每行一个记录
    df_result['output'].to_csv('../prediction_result/submit.txt', index=False, header=False, sep='\n')
    log.info(f'结果已保存到../prediction_result/submit.txt中')


def train_classify_models(df_train_features, feature_cols):
    """
    训练逻辑回归模型，用于分类未见用户的博文是否为高互动博文。

    Args:
        df_train (pd.DataFrame): 训练数据
        feature_cols (list): 特征列

    Returns:
        LogisticRegression: 训练好的逻辑回归模型
    """
    # 分离正负样本
    df_positive = df_train_features[df_train_features['is_high_interaction'] == 1]  # 正样本
    df_negative = df_train_features[df_train_features['is_high_interaction'] == 0]  # 负样本
    print(f'正样本占总比例: {len(df_positive) / len(df_train_features)}')

    df_positive = pd.concat([df_positive]*4, ignore_index=True)
    df_balanced = pd.concat([df_positive, df_negative])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_balanced[feature_cols]
    y = df_balanced['is_high_interaction']

    # X = df_train_features[feature_cols]
    # y = df_train_features['is_high_interaction']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"正负样本权重比例（scale_pos_weight）: {pos_weight:.2f}")

    # LightGBM模型参数
    model = lgb.LGBMClassifier(
        num_leaves=128,  # 叶子节点数
        max_depth=10,  # 树的最大深度
        learning_rate=0.05,  # 学习率
        n_estimators=1500,  # 最大迭代次数
        subsample=0.8,  # 子样本比例
        feature_fraction=0.8,  # 特征子采样比例
        reg_alpha=0.5,  # L1 正则化参数
        reg_lambda=0.5,  # L2 正则化参数
        random_state=seed,  # 随机种子
        importance_type='gain',  # 特征重要性计算方式
        metric=None  # 关闭默认评估指标（在 fit 时指定）
    )

    # 训练模型
    lgb_model = model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],  # 使用训练集和验证集
        eval_names=['train', 'valid'],  # 监测训练集和验证集的效果
        eval_metric='auc',  # 评估指标使用 auc
        callbacks=[
            log_evaluation(100),  # 每 100 轮打印一次日志
            early_stopping(100)   # 早停策略，连续 100 轮未提升则停止训练
        ]
    )
    # 在验证集上预测
    y_pred = lgb_model.predict(X_val)
    # 打印分类报告（评估模型性能）
    print("LightGBM训练完成，分类报告：")
    print(classification_report(y_val, y_pred, target_names=['低互动', '高互动']))

    return lgb_model


if __name__ == '__main__':
    # 特征数据载入
    log.info('特征数据载入...')
    train_feature_path = '../user_data/features/weibo_train_feature.pkl'
    test_feature_path = '../user_data/features/weibo_test_feature.pkl'
    df_train_features = pd.read_pickle(train_feature_path)
    df_test_features = pd.read_pickle(test_feature_path)

    # 定义两组特征列
    log.info('定义特征列...')
    feature_cols_with_user, feature_cols_without_user, target_cols = define_feature_columns()

    # 标记高互动博文
    log.info('标记高互动博文')
    df_train_features = mark_high_interaction_posts(df_train_features)
    
    # 训练分类模型，用于预测未见过的用户是否是一个高互动用户
    log.info('分类模型训练(使用用户特征)...')
    classify_model_with_user = train_classify_models(df_train_features, feature_cols_with_user)
    log.info('分类模型训练(不使用用户特征)...')
    classify_model_without_user = train_classify_models(df_train_features, feature_cols_without_user)

    # 训练模型（仅使用高互动博文）
    log.info('训练预测模型（仅使用高互动博文）')
    models_with_user, models_without_user = train_lgb_models(
        df_train_features, feature_cols_with_user, feature_cols_without_user, target_cols, classify_model_with_user=classify_model_with_user, classify_model_without_user = classify_model_without_user, n_splits=5 
    )

    predict_result(df_test_features, feature_cols_with_user, feature_cols_without_user, models_with_user, models_without_user, classify_model_with_user, classify_model_without_user, threshold=0.5)




