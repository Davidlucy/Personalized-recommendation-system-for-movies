import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import svds

class RecommenderSystem:
    def __init__(self, data):
        self.data = data
        self.user_movie_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_matrices(self):
        """创建用户-电影矩阵"""
        self.user_movie_matrix = pd.pivot_table(
            self.data,
            values='rating',
            index='userId',
            columns='movieId'
        ).fillna(0)
        
    def compute_similarity(self):
        """使用GPU计算相似度矩阵"""
        print("正在计算相似度矩阵...")
        print(f"使用设备: {self.device}")
        
        # 将数据转移到GPU
        matrix_gpu = torch.tensor(self.user_movie_matrix.values, dtype=torch.float32).to(self.device)
        
        # 计算用户相似度
        norm = torch.norm(matrix_gpu, dim=1).reshape(-1, 1)
        normalized = matrix_gpu / norm
        self.user_similarity = torch.mm(normalized, normalized.t())
        
        # 计算物品相似度
        matrix_gpu_T = matrix_gpu.t()
        norm_items = torch.norm(matrix_gpu_T, dim=1).reshape(-1, 1)
        normalized_items = matrix_gpu_T / norm_items
        self.item_similarity = torch.mm(normalized_items, normalized_items.t())
        
        # 将结果转回CPU
        self.user_similarity = self.user_similarity.cpu().numpy()
        self.item_similarity = self.item_similarity.cpu().numpy()
        
        print("相似度矩阵计算完成！")
        
    def user_based_cf(self, user_id, n_recommendations=5):
        """基于用户的协同过滤"""
        if self.user_similarity is None:
            self.compute_similarity()
            
        # 获取用户索引
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        
        # 转移数据到GPU
        user_sim = torch.tensor(self.user_similarity[user_idx], dtype=torch.float32).to(self.device)
        matrix = torch.tensor(self.user_movie_matrix.values, dtype=torch.float32).to(self.device)
        
        # 获取用户未评分的电影
        user_ratings = self.user_movie_matrix.iloc[user_idx]
        unwatched = user_ratings[user_ratings == 0].index
        
        # 预测评分
        predictions = []
        for movie_id in unwatched:
            movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
            
            # 获取相似用户的评分
            other_ratings = matrix[:, movie_idx]
            
            # 计算预测评分
            pred_rating = (torch.sum(other_ratings * user_sim) / 
                         (torch.sum(torch.abs(user_sim)) + 1e-8))
            
            predictions.append((movie_id, float(pred_rating.cpu().numpy())))
            
        # 排序并返回推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def item_based_cf(self, user_id, n_recommendations=5):
        """基于物品的协同过滤"""
        if self.item_similarity is None:
            self.compute_similarity()
            
        # 获取用户评分
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        unwatched = user_ratings[user_ratings == 0].index
        
        # 转移数据到GPU
        ratings = torch.tensor(user_ratings.values, dtype=torch.float32).to(self.device)
        item_sim = torch.tensor(self.item_similarity, dtype=torch.float32).to(self.device)
        
        # 预测评分
        predictions = []
        for movie_id in unwatched:
            movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
            similar_items = item_sim[movie_idx]
            
            # 计算预测评分
            pred_rating = (torch.sum(similar_items * ratings) / 
                         (torch.sum(torch.abs(similar_items)) + 1e-8))
            
            predictions.append((movie_id, float(pred_rating.cpu().numpy())))
        
        # 排序并返回推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def matrix_factorization(self, n_factors=50):
        """矩阵分解"""
        # 转移数据到GPU
        matrix = torch.tensor(self.user_movie_matrix.values, dtype=torch.float32).to(self.device)
        
        # 使用 PyTorch 的 SVD
        U, S, V = torch.svd(matrix)
        
        # 取前n_factors个奇异值
        U = U[:, :n_factors]
        S = S[:n_factors]
        V = V[:, :n_factors]
        
        # 重建评分矩阵
        predicted_ratings = torch.mm(torch.mm(U, torch.diag(S)), V.t())
        
        # 转回CPU
        predicted_ratings = predicted_ratings.cpu().numpy()
        
        return pd.DataFrame(
            predicted_ratings,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.columns
        )