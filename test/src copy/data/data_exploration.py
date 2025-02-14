"""
Loan Data Exploration Analysis Module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import config
from typing import List, Dict, Any, Tuple

class LoanDataExplorer:
    """Loan data exploration analysis class"""
    
    def __init__(self):
        """Initialize data explorer"""
        self.cat_mappings = config.get('preprocessing.categorical_mappings', {})
    
    def load_data(self) -> pd.DataFrame:
        """Load preprocessed loan data"""
        try:
            print("\nLoading preprocessed data...")
            df = pd.read_csv(config.get_path('data.processed'), compression='gzip')
            print(f"Data loaded successfully: {len(df):,} records")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        stats = df.describe(include='all')
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        return {
            'total_records': len(df),
            'total_variables': len(df.columns),
            'memory_usage': memory_usage,
            'variable_types': df.dtypes.value_counts().to_dict(),
            'missing_stats': {
                col: {
                    'count': missing_count,
                    'percentage': round(missing_count / len(df) * 100, 2)
                }
                for col, missing_count in df.isnull().sum().items()
                if missing_count > 0
            }
        }
    
    def get_numeric_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get numerical variables statistics"""
        numeric_stats = {}
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            desc = df[col].describe()
            numeric_stats[col] = {
                'mean': desc['mean'],          # 均值
                'std': desc['std'],            # 标准差
                'min': desc['min'],            # 最小值
                'q1': desc['25%'],             # 第一四分位数
                'median': desc['50%'],         # 中位数
                'q3': desc['75%'],             # 第三四分位数
                'max': desc['max'],            # 最大值
                'skewness': df[col].skew(),    # 偏度
                'kurtosis': df[col].kurtosis() # 峰度
            }
        return numeric_stats
    
    def get_categorical_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get categorical variables statistics"""
        categorical_stats = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            value_counts = df[col].value_counts()
            total = len(df)
            
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_categories': value_counts.head().to_dict(),
                'category_percentages': (value_counts.head() / total * 100).to_dict()
            }
        return categorical_stats
    
    def get_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get correlation matrix for numerical variables"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        return df[numeric_cols].corr()
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and return results"""
        try:
            results = {
                'basic_stats': self.get_basic_stats(df),
                'numeric_stats': self.get_numeric_stats(df),
                'categorical_stats': self.get_categorical_stats(df),
                'correlation_matrix': self.get_correlation_matrix(df)
            }
            return results
        except Exception as e:
            print(f"Error during data analysis: {str(e)}")
            raise

def print_analysis_results(results: Dict[str, Any]):
    """Print analysis results"""
    # 1. Basic Information
    print("\n" + "="*50)
    print("Dataset Overview:")
    print("-"*30)
    basic = results['basic_stats']
    print(f"Total Records: {basic['total_records']:,}")
    print(f"Total Features: {basic['total_variables']}")
    print(f"Memory Usage: {basic['memory_usage']:.2f} MB")
    
    print("\nFeature Types Distribution:")
    for dtype, count in basic['variable_types'].items():
        print(f"- {dtype}: {count}")
    
    # 2. Missing Values
    if basic['missing_stats']:
        print("\nMissing Values Analysis:")
        print("-"*30)
        for col, stats in basic['missing_stats'].items():
            print(f"{col}: {stats['count']:,} ({stats['percentage']}%)")
    
    # 3. Numerical Features
    print("\nNumerical Features Statistics:")
    print("-"*30)
    for col, stats in results['numeric_stats'].items():
        print(f"\n{col}:")
        print(f"- mean: {stats['mean']:.2f}")
        print(f"- std: {stats['std']:.2f}")
        print(f"- min: {stats['min']:.2f}")
        print(f"- q1: {stats['q1']:.2f}")
        print(f"- median: {stats['median']:.2f}")
        print(f"- q3: {stats['q3']:.2f}")
        print(f"- max: {stats['max']:.2f}")
        print(f"- skewness: {stats['skewness']:.2f}")
        print(f"- kurtosis: {stats['kurtosis']:.2f}")
    
    # 4. Categorical Features
    print("\nCategorical Features Analysis:")
    print("-"*30)
    for col, stats in results['categorical_stats'].items():
        print(f"\n{col}:")
        print(f"- Unique Values: {stats['unique_values']}")
        print("- Top Categories (%):")
        for cat, count in stats['top_categories'].items():
            pct = stats['category_percentages'][cat]
            print(f"  {cat}: {count:,} ({pct:.1f}%)")

def main():
    """Main function"""
    explorer = LoanDataExplorer()
    df = explorer.load_data()
    results = explorer.analyze_data(df)
    print_analysis_results(results)
    return df

if __name__ == "__main__":
    main()
