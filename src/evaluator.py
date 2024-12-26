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
            # 确保每个用户至少推荐10个电影
            items = [item[0] for item in user_recs[:10]]  
            recommended_items.update(items)
        coverage = len(recommended_items) / len(all_items)
        print(f"推荐电影数: {len(recommended_items)}, 总电影数: {len(all_items)}")
        return coverage
    
    def calculate_diversity(self, recommendations, item_similarity):
        """计算推荐列表的多样性"""
        diversity_scores = []
        
        for user_recs in recommendations.values():
            items = [item[0] for item in user_recs[:10]]  # 限制推荐数量
            n_items = len(items)
            
            if n_items < 2:
                continue
                
            user_diversity = 0
            count = 0
            for i in range(n_items):
                for j in range(i+1, n_items):
                    if items[i] in item_similarity and items[j] in item_similarity[items[i]]:
                        user_diversity += (1 - item_similarity[items[i]][items[j]])
                        count += 1
                    
            if count > 0:
                diversity_scores.append(user_diversity / count)
        
        if not diversity_scores:
            print("警告：没有计算出有效的多样性分数")
            return 0
                
        return np.mean(diversity_scores)