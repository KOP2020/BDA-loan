"""
模型训练运行脚本
Run model training pipeline
"""
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import config
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.predictor import ModelPredictor

def main():
    """运行模型训练流程"""
    # 1. 加载特征工程后的数据
    print("\nLoading feature engineered data...")
    df = pd.read_csv('data/processed/features_selected.csv.gz', compression='gzip')
    print(f"Data loaded: {len(df):,} records")
    
    # 2. 准备特征矩阵和目标变量
    target = 'loan_condition'
    X = df.drop(columns=[target])
    y = df[target]
    
    # 3. 训练模型
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    training_results = trainer.train(X_train, y_train)
    
    # 4. 评估模型
    evaluator = ModelEvaluator()
    predictor = ModelPredictor(trainer.model, trainer.scaler)
    
    # 在测试集上评估
    y_pred = predictor.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred)
    
    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix(y_test, y_pred)
    
    print("\nModel training and evaluation completed")

if __name__ == "__main__":
    main()
