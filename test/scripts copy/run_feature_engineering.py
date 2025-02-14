"""
特征工程运行脚本
Run feature engineering pipeline
"""
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import config
from src.features.feature_generator import FeatureGenerator
from src.features.feature_selector import FeatureSelector

def main():
    """运行特征工程流程"""
    # 1. 加载预处理后的数据
    print("\nLoading preprocessed data...")
    df = pd.read_csv(config.get_path('data.processed'), compression='gzip')
    print(f"Data loaded: {len(df):,} records")
    
    # 2. 生成特征
    generator = FeatureGenerator()
    df = generator.generate_features(df)
    
    # 3. 准备特征矩阵和目标变量
    target = 'loan_condition'
    X = df.drop(columns=[target])
    y = df[target]
    
    # 4. 特征选择
    selector = FeatureSelector()
    X_selected, selected_features = selector.select_features(X, y)
    
    # 5. 保存结果
    output_path = 'data/processed/features_selected.csv.gz'
    X_selected.to_csv(output_path, compression='gzip', index=False)
    print(f"\nSelected features saved to: {output_path}")

if __name__ == "__main__":
    main()
