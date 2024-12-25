import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Evaluator:
    def __init__(self):
        pass
    
    def calculate_mae(self, true_ratings, pred_ratings):
        """计算平均绝对误差(MAE)"""
        return mean_absolute_error(true_ratings, pred_ratings)
    
    def calculate_rmse(self, true_ratings, pred_ratings):
        """计算均方根误差(RMSE)"""
        return np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    
    def calculate_coverage(self, recommendations, all_items):
        """计算推荐系统的覆盖率"""
        recommended_items = set()
        for user_recs in recommendations.values():
            recommended_items.update([item[0] for item in user_recs])
        return len(recommended_items) / len(all_items)
    
    def calculate_diversity(self, recommendations, item_similarity):
        """计算推荐列表的多样性"""
        diversity_scores = []
        
        for user_recs in recommendations.values():
            items = [item[0] for item in user_recs]
            n_items = len(items)
            
            if n_items < 2:
                continue
                
            diversity = 0
            count = 0
            for i in range(n_items):
                for j in range(i+1, n_items):
                    diversity += (1 - item_similarity[items[i]][items[j]])
                    count += 1
                    
            if count > 0:
                diversity_scores.append(diversity / count)
                
        return np.mean(diversity_scores)