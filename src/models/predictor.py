"""
模型预测器
Model predictor for loan data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from config.config_loader import config

class ModelPredictor:
    """模型预测器类"""
    
    def __init__(self, model, scaler):
        """初始化模型预测器
        
        Args:
            model: 训练好的模型
            scaler: 特征标准化器
        """
        self.model = model
        self.scaler = scaler
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测结果
        """
        # 1. 特征标准化
        X_scaled = self.scaler.transform(X)
        
        # 2. 模型预测
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率
        """
        # 1. 特征标准化
        X_scaled = self.scaler.transform(X)
        
        # 2. 预测概率
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
