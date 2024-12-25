import tkinter as tk
from src.data_processor import DataProcessor
from src.recommender import RecommenderSystem
from src.evaluator import Evaluator
from src.visualizer import Visualizer
from src.gui import RecommenderGUI

def main():
    # 初始化数据处理器
    processor = DataProcessor()
    processor.load_data()
    data = processor.preprocess_data()
    
    # 初始化推荐系统
    recommender = RecommenderSystem(data)
    recommender.create_matrices()
    
    # 初始化可视化器
    visualizer = Visualizer(data)
    
    # 创建GUI
    root = tk.Tk()
    app = RecommenderGUI(root, recommender, visualizer)
    
    # 运行应用
    root.mainloop()

if __name__ == "__main__":
    main()