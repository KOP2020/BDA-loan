"""
运行数据探索分析
Run data exploration analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import config
from src.data.data_exploration import LoanDataExplorer, print_analysis_results
from src.visualization.visualizer import LoanVisualizer

def main():
    """运行数据探索流程"""
    print("\n" + "="*50)
    print("Data Exploration Begins")
    print("="*50 + "\n")
    
    try:
        # 1. 初始化数据探索器和可视化器
        explorer = LoanDataExplorer()
        visualizer = LoanVisualizer(use_brown_theme=False)
        
        # 2. 加载预处理后的数据
        print("\nLoading preprocessed data...")
        df = pd.read_csv(config.get_path('data.processed'), compression='gzip')
        print(f"Data loaded: {len(df):,} records")
        
        # 3. 数据分析
        print("\n执行数据分析...")
        results = explorer.analyze_data(df)
        
        # 4. 打印分析结果
        print("\n数据分析结果：")
        print_analysis_results(results)  # 使用独立的print_analysis_results函数
        
        # 5. 可视化分析
        print("\n开始生成可视化...")
        
        # 5.1 贷款状态分布
        cat_mappings = config.get('preprocessing.categorical_mappings', {})
        if all(col in df.columns for col in ['loan_condition', 'loan_condition_cat']):
            visualizer.plot_loan_condition(df)
        
        
        
        # 5.2 重要数值变量分布
        important_numeric_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 
                                'total_pymnt', 'total_rec_prncp', 'total_rec_int']
        numeric_cols = [col for col in important_numeric_cols if col in df.columns]
        if numeric_cols:
            visualizer.plot_numeric_vars(df, numeric_cols)
        
        # 5.3 相关性矩阵
        visualizer.plot_correlation_matrix(df)
        
        
        # 5.4 重要变量的3D散点图
        scatter_vars = ['interest_rate', 'loan_amount', 'annual_inc']
        if all(var in df.columns for var in scatter_vars) and \
           all(col in df.columns for col in ['loan_condition', 'loan_condition_cat']):
            visualizer.plot_3d_scatter(df)
        
        # 5.5 重要变量的配对关系
        pairwise_vars = [
            # 数值变量
            'loan_amnt', 'int_rate', 'annual_inc', 'dti', 
            'total_pymnt', 'total_rec_prncp',
            # 分类变量的数值映射
            'grade_cat', 'home_ownership_cat', 'purpose_cat', 
            'term_cat', 'loan_condition_cat'
        ]
        # 确保所有变量都在数据集中
        vars_to_plot = [col for col in pairwise_vars if col in df.columns]
        if len(vars_to_plot) >= 2:
            visualizer.plot_pairwise_relationships(df, vars_to_plot)
        
        # 5.6 重要分类变量分析
        for orig_col, cat_col in cat_mappings.items():
            if orig_col in df.columns and cat_col in df.columns:
                # 分类变量分布
                visualizer.plot_categorical_dist(df, orig_col)
                
                # 与数值目标变量的关系
                target_cols = ['int_rate', 'total_pymnt']
                for target_col in target_cols:
                    if target_col in df.columns:
                        visualizer.plot_categorical_target(df, orig_col, target_col)
        
        print("\n可视化完成！所有图表已保存到figures目录。")
        print("\n" + "="*50)
        print("数据探索分析完成！")
        print("="*50)
        
    except Exception as e:
        print(f"\n错误：{str(e)}")
        raise

if __name__ == "__main__":
    main()
