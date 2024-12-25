import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path='data/ml-1m/'):
        self.data_path = data_path
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        self.merged_df = None
        
    def load_data(self):
        """加载MovieLens数据集"""
        print("Loading data...")
        
        # 读取电影数据
        self.movies_df = pd.read_csv(f'{self.data_path}movies.dat', 
                                   sep='::', 
                                   engine='python',
                                   names=['movieId', 'title', 'genres'],
                                   encoding='latin-1')
        
        # 读取评分数据
        self.ratings_df = pd.read_csv(f'{self.data_path}ratings.dat',
                                    sep='::',
                                    engine='python', 
                                    names=['userId', 'movieId', 'rating', 'timestamp'])
        
        # 读取用户数据
        self.users_df = pd.read_csv(f'{self.data_path}users.dat',
                                  sep='::',
                                  engine='python',
                                  names=['userId', 'gender', 'age', 'occupation', 'zipcode'])
        
        print("Data loaded successfully!")
        
    def preprocess_data(self):
        """数据预处理"""
        print("Preprocessing data...")
        
        # 合并数据
        self.merged_df = pd.merge(self.ratings_df, self.movies_df, on='movieId')
        self.merged_df = pd.merge(self.merged_df, self.users_df, on='userId')
        
        # 处理时间戳
        self.merged_df['timestamp'] = pd.to_datetime(self.merged_df['timestamp'], unit='s')
        
        # 性别编码
        self.merged_df['gender'] = self.merged_df['gender'].map({'M': 1, 'F': 0})
        
        # 提取年份
        self.merged_df['year'] = self.merged_df['title'].str.extract('(\d{4})', expand=False)
        
        # 处理类型
        self.merged_df['genres'] = self.merged_df['genres'].fillna('')
        
        print("Data preprocessing completed!")
        return self.merged_df
    
    def split_data(self, test_size=0.2):
        """分割训练集和测试集"""
        train_data, test_data = train_test_split(
            self.merged_df, 
            test_size=test_size, 
            random_state=42
        )
        return train_data, test_data
    
    def create_user_movie_matrix(self):
        """创建用户-电影评分矩阵"""
        return pd.pivot_table(
            self.merged_df,
            values='rating',
            index='userId',
            columns='movieId'
        )