import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

class RecommenderGUI:
    def __init__(self, root, recommender, visualizer):
        self.root = root
        self.recommender = recommender
        self.visualizer = visualizer
        
        # 设置窗口
        self.root.title("电影推荐系统")
        self.root.geometry("1000x700")  # 加大窗口尺寸
        self.root.configure(bg='#f0f0f0')  # 设置背景色
        
        # 创建样式
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 使用clam主题
        
        # 配置样式
        self.style.configure('TNotebook', background='#f0f0f0')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('微软雅黑', 10))
        self.style.configure('TButton', 
                           padding=10, 
                           font=('微软雅黑', 10),
                           background='#2196F3',  # 蓝色按钮
                           foreground='white')
        self.style.configure('TRadiobutton', 
                           font=('微软雅黑', 10),
                           background='#f0f0f0')
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(expand=True, fill='both')
        
        self.create_notebook()
        
    def create_notebook(self):
        """创建标签页"""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(pady=10, expand=True, fill='both')
        
        # 创建各个页面
        self.create_recommendation_page()
        self.create_visualization_page()
        self.create_evaluation_page()
        
    def create_recommendation_page(self):
        """创建推荐页面"""
        rec_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(rec_frame, text="电影推荐")
        
        # 创建左侧输入区域
        input_frame = ttk.Frame(rec_frame)
        input_frame.pack(side='left', padx=20, fill='y')
        
        # 用户ID输入
        ttk.Label(input_frame, text="用户ID:", font=('微软雅黑', 12, 'bold')).pack(pady=5)
        self.user_id_entry = ttk.Entry(input_frame, width=20, font=('微软雅黑', 10))
        self.user_id_entry.pack(pady=5)
        
        # 推荐方法选择
        method_frame = ttk.LabelFrame(input_frame, text="推荐方法", padding=10)
        method_frame.pack(pady=20, fill='x')
        
        self.method_var = tk.StringVar(value="user")
        ttk.Radiobutton(method_frame, text="基于用户的协同过滤", 
                       variable=self.method_var, value="user").pack(pady=5, anchor='w')
        ttk.Radiobutton(method_frame, text="基于物品的协同过滤", 
                       variable=self.method_var, value="item").pack(pady=5, anchor='w')
        ttk.Radiobutton(method_frame, text="矩阵分解(SVD)", 
                       variable=self.method_var, value="svd").pack(pady=5, anchor='w')
        
        # 获取推荐按钮
        ttk.Button(input_frame, text="获取推荐", 
                  command=self.show_recommendations,
                  style='Accent.TButton').pack(pady=20)
        
        # 创建右侧推荐结果显示区域
        result_frame = ttk.LabelFrame(rec_frame, text="推荐结果", padding=10)
        result_frame.pack(side='left', expand=True, fill='both', padx=20)
        
        # 推荐结果显示
        self.rec_text = tk.Text(result_frame, 
                              height=20, 
                              width=50, 
                              font=('微软雅黑', 10),
                              wrap=tk.WORD,
                              bg='white')
        self.rec_text.pack(expand=True, fill='both')
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', command=self.rec_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.rec_text.configure(yscrollcommand=scrollbar.set)
        
    def create_visualization_page(self):
        """创建可视化页面"""
        vis_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(vis_frame, text="数据可视化")
        
        # 创建按钮框架
        button_frame = ttk.Frame(vis_frame)
        button_frame.pack(pady=(0, 20))  # 只在底部添加间距
        
        # 可视化按钮
        ttk.Button(button_frame, text="评分分布图", 
                  command=lambda: self.update_visualization('rating')).pack(side='left', padx=10)
        ttk.Button(button_frame, text="电影类型分布图", 
                  command=lambda: self.update_visualization('genre')).pack(side='left', padx=10)
        ttk.Button(button_frame, text="性别评分分布图", 
                  command=lambda: self.update_visualization('gender')).pack(side='left', padx=10)
        
        # 创建图表显示区域
        self.fig = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_visualization(self, plot_type):
        """更新可视化图表"""
        self.fig.clear()  # 清除当前图表
        ax = self.fig.add_subplot(111)
        
        if plot_type == 'rating':
            # 评分分布图
            sns.histplot(data=self.visualizer.data, x='rating', bins=10, ax=ax)
            ax.set_title('评分分布')
            ax.set_xlabel('评分')
            ax.set_ylabel('数量')
        
        elif plot_type == 'genre':
            # 电影类型分布图
            genres = self.visualizer.data['genres'].str.split('|', expand=True).stack()
            genre_counts = genres.value_counts()
            genre_counts.plot(kind='bar', ax=ax)
            ax.set_title('电影类型分布')
            ax.set_xlabel('类型')
            ax.set_ylabel('数量')
            plt.xticks(rotation=45)
            self.fig.tight_layout()  # 调整布局以显示完整的标签
        
        elif plot_type == 'gender':
            # 性别评分分布图
            sns.boxplot(data=self.visualizer.data, x='gender', y='rating', ax=ax)
            ax.set_title('不同性别的评分分布')
            ax.set_xlabel('性别')
            ax.set_ylabel('评分')
        
        self.canvas.draw()  # 重绘图表
        
    def create_evaluation_page(self):
        """创建评估页面"""
        eval_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(eval_frame, text="系统评估")
        
        # 创建评估按钮
        ttk.Button(eval_frame, 
                  text="开始系统评估", 
                  command=self.show_evaluation_results,
                  style='Accent.TButton').pack(pady=20)
        
        # 评估结果显示框架
        result_frame = ttk.LabelFrame(eval_frame, text="评估结果", padding=10)
        result_frame.pack(expand=True, fill='both', padx=20)
        
        # 评估结果显示
        self.eval_text = tk.Text(result_frame, 
                               height=15, 
                               width=60,
                               font=('微软雅黑', 10),
                               wrap=tk.WORD,
                               bg='white')
        self.eval_text.pack(expand=True, fill='both')
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', command=self.eval_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.eval_text.configure(yscrollcommand=scrollbar.set)
        
    # 创建自定义按钮样式
    def create_custom_style(self):
        self.style.configure('Accent.TButton',
                           background='#2196F3',
                           foreground='white',
                           padding=(20, 10),
                           font=('微软雅黑', 10, 'bold'))
        
        self.style.map('Accent.TButton',
                      background=[('active', '#1976D2')],
                      foreground=[('active', 'white')])
        
    def show_recommendations(self):
        """显示推荐结果"""
        try:
            user_id = int(self.user_id_entry.get())
            method = self.method_var.get()
            
            if method == "user":
                recommendations = self.recommender.user_based_cf(user_id)
            elif method == "item":
                recommendations = self.recommender.item_based_cf(user_id)
            else:  # svd
                # 使用矩阵分解方法
                predicted_matrix = self.recommender.matrix_factorization(n_factors=50)
                user_predictions = predicted_matrix.loc[user_id]
                # 获取用户未评分的电影
                watched_movies = self.recommender.data[
                    self.recommender.data['userId'] == user_id
                ]['movieId'].values
                unwatched = user_predictions.index.difference(watched_movies)
                # 获取预测评分
                predictions = [(movie_id, user_predictions[movie_id]) 
                             for movie_id in unwatched]
                # 排序并获取前N个推荐
                recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
                
            # 清空文本框
            self.rec_text.delete(1.0, tk.END)
            
            # 显示推荐结果
            self.rec_text.insert(tk.END, "推荐电影列表:\n\n")
            for movie_id, score in recommendations:
                movie_title = self.recommender.data[
                    self.recommender.data['movieId'] == movie_id
                ]['title'].iloc[0]
                self.rec_text.insert(tk.END, f"{movie_title}: {score:.2f}分\n")
                
        except ValueError:
            self.rec_text.delete(1.0, tk.END)
            self.rec_text.insert(tk.END, "请输入有效的用户ID")
        except Exception as e:
            self.rec_text.delete(1.0, tk.END)
            self.rec_text.insert(tk.END, f"推荐过程中出现错误：\n{str(e)}")
        
    def show_evaluation_results(self):
        """显示评估结果"""
        try:
            # 创建进度条窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("评估进度")
            progress_window.geometry("300x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="正在进行系统评估...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=200, mode='determinate')
            progress_bar.pack(pady=10)
            
            detail_label = ttk.Label(progress_window, text="准备数据...")
            detail_label.pack(pady=5)
            
            def update_progress(progress, text):
                progress_bar['value'] = progress
                detail_label['text'] = text
                progress_window.update()
            
            # 优化1: 减少测试数据量
            update_progress(10, "分割训练集和测试集...")
            # 随机选择1000个样本进行评估
            sample_data = self.recommender.data.sample(n=1000, random_state=42)
            train_data, test_data = train_test_split(
                sample_data, 
                test_size=0.2, 
                random_state=42
            )
            
            # 优化2: 预先计算相似度矩阵
            update_progress(20, "计算相似度矩阵...")
            if self.method_var.get() == "user" and self.recommender.user_similarity is None:
                self.recommender.compute_similarity()
            elif self.method_var.get() == "item" and self.recommender.item_similarity is None:
                self.recommender.compute_similarity()
            
            # 优化3: 批量处理评分预测
            update_progress(30, "预测评分...")
            true_ratings = test_data['rating'].values
            pred_ratings = []
            
            # 每次处理50个样本
            batch_size = 50
            total_batches = len(test_data) // batch_size + 1
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(test_data))
                batch_data = test_data.iloc[start_idx:end_idx]
                
                progress = 30 + (batch_idx / total_batches) * 30
                update_progress(progress, f"预测评分进度: {batch_idx+1}/{total_batches}")
                
                for _, row in batch_data.iterrows():
                    user_id = row['userId']
                    movie_id = row['movieId']
                    if self.method_var.get() == "user":
                        recommendations = self.recommender.user_based_cf(user_id, n_recommendations=10)
                    else:
                        recommendations = self.recommender.item_based_cf(user_id, n_recommendations=10)
                    pred = next((score for mid, score in recommendations if mid == movie_id), 
                              test_data['rating'].mean())
                    pred_ratings.append(pred)
            
            # 计算评估指标
            update_progress(60, "计算评估指标...")
            mae = np.mean(np.abs(np.array(true_ratings) - np.array(pred_ratings)))
            rmse = np.sqrt(np.mean((np.array(true_ratings) - np.array(pred_ratings)) ** 2))
            
            # 优化4: 减少覆盖率计算的样本量
            update_progress(70, "计算推荐覆盖率...")
            all_movies = set(self.recommender.data['movieId'].unique())
            recommendations = {}
            # 只选择50个用户进行评估
            sample_users = np.random.choice(self.recommender.data['userId'].unique(), 50)
            
            for i, user_id in enumerate(sample_users):
                progress = 70 + (i / len(sample_users)) * 20
                update_progress(progress, f"计算用户推荐: {i+1}/{len(sample_users)}")
                
                if self.method_var.get() == "user":
                    recs = self.recommender.user_based_cf(user_id, n_recommendations=10)
                else:
                    recs = self.recommender.item_based_cf(user_id, n_recommendations=10)
                recommendations[user_id] = recs
            
            recommended_items = set()
            for user_recs in recommendations.values():
                recommended_items.update([item[0] for item in user_recs])
            coverage = len(recommended_items) / len(all_movies)
            
            # 优化5: 简化多样性计算
            update_progress(90, "计算推荐多样性...")
            diversity_scores = []
            for user_recs in recommendations.values():
                items = [item[0] for item in user_recs]
                if len(items) < 2:
                    continue
                
                # 只计算相邻推荐项之间的多样性
                diversity = 0
                count = 0
                for i in range(len(items)-1):
                    if items[i] in self.recommender.item_similarity and \
                       items[i+1] in self.recommender.item_similarity[items[i]]:
                        diversity += (1 - self.recommender.item_similarity[items[i]][items[i+1]])
                        count += 1
                
                if count > 0:
                    diversity_scores.append(diversity / count)
            
            diversity = np.mean(diversity_scores) if diversity_scores else 0
            
            update_progress(100, "评估完成!")
            progress_window.destroy()
            
            # 显示评估结果
            self.eval_text.delete(1.0, tk.END)
            self.eval_text.insert(tk.END, "系统评估结果:\n\n")
            self.eval_text.insert(tk.END, f"平均绝对误差 (MAE): {mae:.4f}\n")
            self.eval_text.insert(tk.END, f"均方根误差 (RMSE): {rmse:.4f}\n")
            self.eval_text.insert(tk.END, f"推荐覆盖率: {coverage:.2%}\n")
            self.eval_text.insert(tk.END, f"推荐多样性: {diversity:.4f}\n\n")
            
            self.eval_text.insert(tk.END, "指标说明:\n")
            self.eval_text.insert(tk.END, "1. MAE：预测评分与实际评分的平均绝对误差，越小越好\n")
            self.eval_text.insert(tk.END, "2. RMSE：预测评分与实际评分的均方根误差，越小越好\n")
            self.eval_text.insert(tk.END, "3. 覆盖率：推荐系统能够推荐的电影占总电影的比例\n")
            self.eval_text.insert(tk.END, "4. 多样性：推荐结果的多样化程度，值越大表示推荐的电影越多样化\n")
            
            self.eval_text.insert(tk.END, "\n注：为提高性能，评估使用了随机抽样的数据子集。")
        
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            self.eval_text.delete(1.0, tk.END)
            self.eval_text.insert(tk.END, f"评估过程中出现错误：\n{str(e)}")