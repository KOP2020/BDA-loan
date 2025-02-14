"""
运行数据预处理
Run data preprocessing
"""
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import config
from src.data.preprocessor import LoanDataPreprocessor

def main():
    """运行数据预处理流程"""
    # 加载原始数据
    print("\nLoading raw data...")
    df = pd.read_csv(config.get_path('data.raw'), compression='gzip')
    print(f"Raw data loaded: {len(df):,} records")
    
    # 初始化预处理器
    preprocessor = LoanDataPreprocessor()
    
    # 执行预处理
    print("\n=== Processing Data ===")
    df_processed = preprocessor.preprocess(df.copy())
    print(f"Processed data shape: {df_processed.shape}")
    
    print("\nPreprocessing completed!")

if __name__ == "__main__":
    main()
