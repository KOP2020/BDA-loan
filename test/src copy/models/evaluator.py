"""
模型评估器
Model evaluator for loan data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from config.config_loader import config

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self):
        """初始化模型评估器"""
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            评估指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('docs/figures/confusion_matrix.png')
        plt.close()
