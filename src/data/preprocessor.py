"""
贷款数据预处理
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config.config_loader import config
from pathlib import Path

class LoanDataPreprocessor:
    def __init__(self):
        """初始化预处理器"""
        print("预处理器初始化完成")
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据
        
        Args:
            df: 输入数据框
            
        Returns:
            预处理后的数据框
        """
        print("\n开始数据预处理...")
        
        # 1. 筛选2007-2010年的数据
        if 'year' in df.columns:
            df = df[df['year'].between(2007, 2010)]
            print(f"筛选2007-2010年数据后的形状：{df.shape}")
        
        # 2. 检查缺失值
        missing = df.isnull().sum()
        if missing.any():
            print("\n发现缺失值：")
            print(missing[missing > 0])
        else:
            print("\n没有发现缺失值")
            
        # 3. 删除不需要的列
        columns_to_drop = config.get('preprocessing.columns_to_drop', [])
        print("\n使用通用列删除配置")
            
        # 确保要删除的列在数据中存在
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        print(f"删除{len(columns_to_drop)}列后的形状：{df.shape}")
        print("删除的列：", columns_to_drop)
        
        # 4. 检查categorical variable的编码映射是否完整
        cat_cols = df.select_dtypes(include=['object']).columns
        cat_mappings = config.get('preprocessing.categorical_mappings', {})
        for orig_col, cat_col in cat_mappings.items():
            if orig_col in df.columns and cat_col in df.columns:
                print(f"\n检查 {orig_col} 的编码映射:")
                unique_cats = sorted(df[orig_col].unique())
                unique_codes = sorted(df[cat_col].unique())
                print(f"类别数量: {len(unique_cats)}, 编码数量: {len(unique_codes)}")
                print("编码映射:")
                mapping_dict = dict(zip(df[orig_col], df[cat_col]))
                for cat in unique_cats:
                    print(f"  {cat}: {mapping_dict[cat]}")
        
        # 5. 保存处理后的数据
        output_path = config.get_path('data.processed')
            
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        df.to_csv(output_path, compression='gzip', index=False)
        print(f"\n数据已保存到: {output_path}")
        print(f"最终数据形状: {df.shape}")
        
        print("\n预处理完成！")
        return df
