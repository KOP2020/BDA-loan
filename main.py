"""
回归分析运行脚本 - 集成特征选择和模型训练
Run regression analysis pipeline
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import joblib
import os

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入特征选择和模型训练模块
from src.features.feature_selector import select_features_for_regression
from src.models.trainer import RegressionTrainer

def main():
    """运行回归分析BDA流程"""
    start_time = time.time()
    
    # 创建输出目录
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载预处理后的数据
    print("\nLoading processed data...")
    df = pd.read_csv('data/processed/loan_processed.csv')
    print(f"Data loaded: {len(df):,} records with {df.shape[1]} columns")
    
    # 2. 定义目标变量
    target = 'total_rec_prncp'
    
    # 3. 特征选择
    print("\nPerforming feature selection...")
    df_selected, selected_features = select_features_for_regression(
        df, target, n_features=7, output_dir=output_dir
    )
    
    # 保存特征选择后的数据
    df_selected.to_csv(output_dir / 'features_selected.csv', index=False)
    print(f"Selected features saved to 'data/processed/features_selected.csv'")
    
    # 4. 准备特征矩阵和目标变量
    X = df_selected.drop(columns=[target])
    y = df_selected[target]
    
    # 5. 训练和评估回归模型
    print("\nTraining regression models...")
    trainer = RegressionTrainer(output_dir=output_dir)
    results = trainer.run_workflow(X, y)
    
    # 6. 汇总分析结果
    print("\nSummary of regression analysis:")
    print(f"Best model: {trainer.best_model}")
    print(f"Best RMSE: {trainer.best_score:.4f}")
    
    # 计算总运行时间
    elapsed = time.time() - start_time
    print(f"\nRegression analysis completed in {elapsed:.1f} seconds")
    
    # 将结果保存为CSV
    results_df = pd.DataFrame([
        {
            'model': model,
            'rmse': results[model]['rmse'],
            'r2': results[model]['r2']
        } for model in results
    ]).sort_values('rmse')
    
    results_df.to_csv(output_dir / 'regression_results.csv', index=False)
    print("Results saved to 'data/processed/regression_results.csv'")
    
    return results

if __name__ == "__main__":
    main()