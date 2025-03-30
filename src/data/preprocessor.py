"""
简化的贷款数据预处理流程 - 一体化版本
Simplified loan data preprocessing pipeline - All-in-one version
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 硬编码配置信息
CONFIG = {
    'paths': {
        'data': {
            'raw': 'data/raw/loan_final313.csv',
            'processed': 'data/processed/loan_processed.csv'
        }
    },
    'preprocessing': {
        # 要删除的列
        'columns_to_drop': [
            'id',  # ID列不参与建模
            'issue_d',  # 已有year列，不需要具体日期
            'final_d',  # 已有year列，不需要具体日期
            'year',  # 不分析时序关系
            'recoveries',  # 与贷款收回过程相关，不直接影响还款能力
            'total_pymnt',  # 与目标变量高度相关
            'loan_condition',  # 与目标变量高度相关
            'loan_condition_cat',  # 与目标变量高度相关
            'installment'  # 和loanamount高度相关，多变量共线
        ],
        
        # 原始categorical变量及其对应的数值编码列
        'categorical_mappings': {
            'grade': 'grade_cat',
            'purpose': 'purpose_cat',
            'home_ownership': 'home_ownership_cat',
            'term': 'term_cat',
            'interest_payments': 'interest_payment_cat',
            'application_type': 'application_type_cat',
            'income_category': 'income_cat'
        }
    }
}

def preprocess_loan_data(input_path=None, output_path=None):
    """预处理贷款数据的完整流程
    
    参数:
        input_path: 输入数据路径，默认使用硬编码配置
        output_path: 输出数据路径，默认使用硬编码配置
    
    返回:
        处理后的数据框
    """
    # 设置路径
    if input_path is None:
        input_path = CONFIG['paths']['data']['raw']
    if output_path is None:
        output_path = CONFIG['paths']['data']['processed']
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\nLoading raw data...")
    df = pd.read_csv(input_path)
    print(f"Raw data loaded: {len(df):,} records")
    
    print("\n=== Processing Data ===")
    
    # 1. 筛选2007-2010年的数据
    if 'year' in df.columns:
        df = df[df['year'].between(2007, 2010)]
        print(f"After filtering 2007-2010 data: {df.shape}")
    
    # 2. 检查缺失值
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values found:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found")
        
    # 3. 删除不需要的列
    columns_to_drop = CONFIG['preprocessing']['columns_to_drop']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    print(f"After dropping {len(columns_to_drop)} columns: {df.shape}")
    
    # 4. 将原始分类变量转换为独热编码，删除标签编码变量
    print("\n=== Converting Categorical Variables to One-Hot Encoding ===")
    
    cat_mappings = CONFIG['preprocessing']['categorical_mappings']
    # 获取所有要进行独热编码的原始分类变量
    cols_to_encode = [col for col in cat_mappings.keys() if col in df.columns]
    # 获取对应的标签编码变量（要删除的）
    encoded_cols_to_drop = [cat_mappings[col] for col in cols_to_encode if cat_mappings[col] in df.columns]
    
    print(f"Processing {len(cols_to_encode)} categorical columns")
    
    if cols_to_encode:
        # 一次性对所有分类变量进行独热编码，更高效
        # 确保结果是0/1整数而不是TRUE/FALSE布尔值
        df_encoded = pd.get_dummies(df[cols_to_encode], prefix_sep='_').astype(int)
        
        # 删除标签编码变量
        if encoded_cols_to_drop:
            df = df.drop(columns=encoded_cols_to_drop)
        
        # 合并独热编码结果与原始数据
        # 首先删除已经独热编码的原始分类列
        df = df.drop(columns=cols_to_encode)
        # 然后将独热编码列加到数据框
        df = pd.concat([df, df_encoded], axis=1)
        
        # 输出结果信息
        print(f"- Added {df_encoded.shape[1]} one-hot encoded columns")
        print(f"- Dropped {len(encoded_cols_to_drop)} label-encoded columns")
        print(f"- Final data shape: {df.shape}")
    
    # 5. 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    print(f"Final data shape: {df.shape}")
    
    print("\nPreprocessing completed!")
    return df

if __name__ == "__main__":
    # 添加项目根目录到Python路径
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 运行预处理
    preprocess_loan_data()