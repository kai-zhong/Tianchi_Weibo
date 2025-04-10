import numpy as np
import pandas as pd
import os
import argparse
import math
import pickle
import re
import random
import signal
import multitasking
import warnings
import torch

from random import shuffle
from tqdm import tqdm
from utils import Logger
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 42
random.seed(seed)

parser = argparse.ArgumentParser(description='特征提取')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--datafrom', default='txt')

args = parser.parse_args()
logfile = args.logfile
datafrom = args.datafrom

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger

# 加载 BERT 分词器和模型
model_path = "../utils/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # 设置为评估模式

def data_preprocess(data_path, train_filename, test_filename):
    # 读取训练集合
    df_train = pd.read_csv(data_path+train_filename+'.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])
    train_dtype_dict = {
        'uid': 'string',
        'mid': 'string',
        'time': 'datetime64[s]',
        'forward_count': 'Int32',
        'comment_count': 'Int32',
        'like_count': 'Int32',
        'content': 'string'
    }
    # 将字段类型转换为相应类型
    df_train = df_train.astype(train_dtype_dict)
    # 按博文发布时间从老到新排序
    df_train = df_train.sort_values(['time']).reset_index(drop=True)
    # 存储为pkl文件
    df_train.to_pickle(data_path+train_filename+'.pkl')
    
    # 读取测试集合
    df_test = pd.read_csv(data_path+test_filename+'.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'content'])
    test_dtype_dict = {
        'uid': 'string',
        'mid': 'string',
        'time': 'datetime64[s]',
        'content': 'string'
    }
    # 将字段类型转换为相应类型
    df_test = df_test.astype(test_dtype_dict).reset_index(drop=True)
    # 转换为plk格式存储
    df_test.to_pickle(data_path+test_filename+'.pkl')
    return df_train, df_test


def get_bert_embedding(text, batch_size=32, max_length=512):
    # 分词并转换为张量
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # 使用 BERT 模型生成嵌入
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取 [CLS] token 的嵌入（表示整个句子）
    # 或者对所有 token 的嵌入取平均（last_hidden_state.mean(dim=1)）
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 平均池化
    # 或者使用 pooler_output（[CLS] token 的表示）
    # embeddings = outputs.pooler_output.cpu().numpy()
    return embeddings


def get_bert_embedding_batch(texts, batch_size=32, max_length=512):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = get_bert_embedding(batch_texts)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)
    

def user_feature_extract(df_train, df_test):
    # 特征1-5：互动数（转发、评论、点赞）平均数、最大值、中位数、标准差、【历史列表】（不一定使用），和发微博数
    log.info("正在构建用户特征1-5...")
    user_stats = df_train.groupby('uid').agg({
    'forward_count': ['mean', lambda x : np.max(x), lambda x : np.median(x), lambda x: np.std(x, ddof=1) if len(x) > 1 else 0, list],  # 平均转发数，转发数最大值，转发数中位数，转发数标准差，转发数列表
    'comment_count': ['mean', lambda x : np.max(x), lambda x : np.median(x), lambda x: np.std(x, ddof=1) if len(x) > 1 else 0, list],  # 平均评论数，评论数最大值，评论数中位数，评论数标准差，评论数列表
    'like_count': ['mean', lambda x : np.max(x), lambda x : np.median(x), lambda x: np.std(x, ddof=1) if len(x) > 1 else 0, list],     # 平均点赞数，点赞数最大值，点赞数中位数，点赞数标准差，点赞数列表
    'mid': 'count'                    # 发微博数
    }).reset_index()
    # 重命名列名
    user_stats.columns = [
        'uid', 
        'user_mean_forward_count', 'user_max_forward_count', 'user_median_forward_count', 'user_std_forward_count', 'forward_count_list',
        'user_mean_comment_count', 'user_max_comment_count', 'user_median_comment_count', 'user_std_comment_count', 'comment_count_list',
        'user_mean_like_count', 'user_max_like_count', 'user_median_like_count', 'user_std_like_count', 'like_count_list',
        'post_count'
    ]

    # 特征6：95%分位数
    log.info('正在构建用户特征6:95%分位数...')
    user_stats['user_forward_quantile_95'] = user_stats['forward_count_list'].apply(lambda x: np.percentile(x, 95))
    user_stats['user_comment_quantile_95'] = user_stats['comment_count_list'].apply(lambda x: np.percentile(x, 95))
    user_stats['user_like_quantile_95'] = user_stats['like_count_list'].apply(lambda x: np.percentile(x, 95))

    # 特征7：非零互动比例
    log.info('正在构建用户特征7：非零互动比例...')
    user_stats['user_nonzero_forward_ratio'] = user_stats['forward_count_list'].apply(lambda x: np.mean(np.array(x) > 0))
    user_stats['user_nonzero_comment_ratio'] = user_stats['comment_count_list'].apply(lambda x: np.mean(np.array(x) > 0))
    user_stats['user_nonzero_like_ratio'] = user_stats['like_count_list'].apply(lambda x: np.mean(np.array(x) > 0))

    # 特征8：综合分数（文章的综合分数由三类分数加权组合获得，用户的综合分数是所有发布的博文综合分数的平均值,也就是发博的互动数的平均数）
    log.info('正在构建用户特征8：综合分数...')
    user_stats['user_interaction_score'] = user_stats['user_mean_forward_count'] * 0.5 + user_stats['user_mean_comment_count'] * 0.25 + user_stats['user_mean_like_count'] * 0.25

    # 用户特征构建完成
    log.info('整合用户特征')
    df_train = pd.merge(df_train, user_stats, on='uid', how='left').drop(columns=['forward_count_list','comment_count_list','like_count_list'])
    df_test = pd.merge(df_test, user_stats, on='uid', how='left').drop(columns=['forward_count_list','comment_count_list','like_count_list'])
    
    # 用空字符串填充文本数据
    df_train['content'] = df_train['content'].fillna('')
    df_test['content'] = df_test['content'].fillna('')

    # df_test中存在df_train中未出现的用户，用-1来填充未知用户的用户特征
    df_test.fillna(-1, inplace=True)
    df_test = df_test.infer_objects()
    
    return df_train, df_test


def post_feature_extract(df_train, df_test):

    # 特征1：星期几
    log.info('正在构建博文特征1：星期信息...')
    df_train['day_of_week'] = df_train['time'].dt.dayofweek
    df_test['day_of_week'] = df_test['time'].dt.dayofweek

    # 特征2：发博时间是哪个一天中哪个小时
    log.info('正在构建博文特征2：时刻信息...')
    df_train['hour'] = df_train['time'].dt.hour
    df_test['hour'] = df_test['time'].dt.hour

    # 特征3：周末信息，是否是周末
    log.info('正在构建博文特征3：周末信息...')
    df_train['is_weekend'] = df_train['day_of_week'].isin([6, 7]).astype(int)
    df_test['is_weekend'] = df_test['day_of_week'].isin([6, 7]).astype(int)

    # 特征4：发博时间间隔：当前博文与用户上一条博文的时间间隔（单位：小时）
    log.info('正在构建博文特征4：发博时间间隔...')
    df_train = df_train.sort_values(['uid', 'time']).reset_index(drop=True)
    df_train['time_since_last_post'] = df_train.groupby('uid')['time'].diff().dt.total_seconds() / 3600
    df_train['time_since_last_post'] = df_train['time_since_last_post'].fillna(0)

    df_test = df_test.sort_values(['uid', 'time']).reset_index(drop=True)
    df_test['time_since_last_post'] = df_test.groupby('uid')['time'].diff().dt.total_seconds() / 3600
    df_test['time_since_last_post'] = df_test['time_since_last_post'].fillna(0)

    # 特征5：content长度
    log.info('正在构建博文特征5：content长度...')
    df_train['content_length'] = df_train['content'].apply(len)
    df_test['content_length'] = df_test['content'].apply(len)

    # 特征6：是否包含话题标签
    log.info('正在构建博文特征6：是否包含话题标签...')
    df_train['has_topic'] = df_train['content'].apply(lambda x: 1 if re.search(r'#', x) else 0)
    df_test['has_topic'] = df_test['content'].apply(lambda x: 1 if re.search(r'#', x) else 0)

    # 特征7：是否包含@用户
    log.info('正在构建博文特征7：是否包含@用户...')
    df_train['has_mention'] = df_train['content'].apply(lambda x: 1 if re.search(r'@', x) else 0)
    df_test['has_mention'] = df_test['content'].apply(lambda x: 1 if re.search(r'@', x) else 0)

    # 特征8：是否包含抽奖关键词
    log.info('正在构建博文特征8：是否包含抽奖关键词...')
    df_train['has_lottery'] = df_train['content'].apply(lambda x: 1 if re.search(r'抽奖|转发有礼', x) else 0)
    df_test['has_lottery'] = df_test['content'].apply(lambda x: 1 if re.search(r'抽奖|转发有礼', x) else 0)

    # 特征9：是否包含链接
    log.info('正在构建博文特征9：是否包含链接...')
    df_train['has_link'] = df_train['content'].apply(lambda x: 1 if re.search(r'http[s]?://', x) else 0)
    df_test['has_link'] = df_test['content'].apply(lambda x: 1 if re.search(r'http[s]?://', x) else 0)

    # 特征10：Bert生成嵌入特征
    train_texts = df_train['content'].tolist()
    test_texts = df_test['content'].tolist()

    log.info('Bert生成训练集嵌入特征...')
    train_embeddings = get_bert_embedding_batch(train_texts, batch_size=32, max_length=512)
    log.info('Bert生成测试集嵌入特征...')
    test_embeddings = get_bert_embedding_batch(test_texts, batch_size=32, max_length=512)
    
    log.info('对Bert生成的特征降维')
    n_components = 20
    pca = PCA(n_components=n_components)
    # 对训练数据拟合 PCA 并转换
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    # 对测试数据应用相同的 PCA 转换
    test_embeddings_pca = pca.transform(test_embeddings)
    
    log.info('并入特征dataframe')
    train_embeddings_df = pd.DataFrame(
        train_embeddings_pca,
        columns=[f'bert_{i}' for i in range(n_components)],
        index=df_train.index  # 确保索引与 df_train 对齐
    )
    test_embeddings_df = pd.DataFrame(
        test_embeddings_pca,
        columns=[f'bert_{i}' for i in range(n_components)],
        index=df_test.index  # 确保索引与 df_test 对齐
    )
    # 将嵌入添加到原始 DataFrame
    df_train = pd.concat([df_train, train_embeddings_df], axis=1)
    df_test = pd.concat([df_test, test_embeddings_df], axis=1)
    # 删除 content 列 和 mid 列
    df_train_feature = df_train.drop(columns=['mid', 'content'])
    df_test_feature = df_test.drop(columns=['content'])
    df_train_feature = df_train_feature.reset_index(drop=True)
    df_test_feature = df_test_feature.reset_index(drop=True)
    log.info('博文特征提取完毕.')

    return df_train_feature, df_test_feature

if __name__ == '__main__':
    data_path = '../data/'
    train_filename = 'weibo_train_data'
    test_filename = 'weibo_predict_data'
    save_train_feature_path = '../user_data/features/weibo_train_feature.pkl'
    save_test_feature_path = '../user_data/features/weibo_test_feature.pkl'
    os.makedirs('../user_data/features/', exist_ok=True)

    log.info("数据载入...")
    if os.path.exists(data_path+train_filename+'.pkl') and (datafrom == 'pickle' or datafrom == 'pkl'):
        df_train = pd.read_pickle(data_path+train_filename+'.pkl')
        df_test = pd.read_pickle(data_path+test_filename+'.pkl')
    elif datafrom == 'txt':
        df_train, df_test = data_preprocess(data_path, train_filename, test_filename)
    else:
        raise ValueError("格式不符合要求.请选择'txt'或'pkl'")
    
    log.info('用户特征提取...')
    df_train, df_test = user_feature_extract(df_train, df_test)

    log.info('博文特征提取...')
    df_train_feature, df_test_feature = post_feature_extract(df_train, df_test)

    df_train_feature.to_pickle(save_train_feature_path)
    df_test_feature.to_pickle(save_test_feature_path)