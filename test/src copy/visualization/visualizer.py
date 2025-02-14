"""
Loan data visualization module
可视化模块：用于生成贷款数据的各类图表
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from config.config_loader import config
from mpl_toolkits.mplot3d import Axes3D


class LoanVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        self.figures_dir = config.get_path('visualization.figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 加载类别映射配置
        self.cat_mappings = config.get('preprocessing.categorical_mappings', {})
        
    def save_plot(self, name: str):
        """Save the plot"""
        plt.savefig(self.figures_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_loan_condition(self, df: pd.DataFrame):
        """Plot loan condition distribution"""
        plt.figure(figsize=(15, 6))
        
        # Left: bar chart
        plt.subplot(121)
        condition_counts = df['loan_condition'].value_counts()
            
        ax1 = sns.barplot(x=range(len(condition_counts)), y=condition_counts.values)
        plt.xticks(range(len(condition_counts)), condition_counts.index, rotation=45)
        
        # Add count and percentage labels
        total = len(df)
        for i, v in enumerate(condition_counts.values):
            ax1.text(i, v, f'{v:,}\n({v/total*100:.1f}%)', ha='center', va='bottom')
        
        plt.title('Loan Condition Distribution')  # 贷款状态分布
        plt.xlabel('Condition')        # 状态
        plt.ylabel('Count')      # 数量
        
        # Right: pie chart
        plt.subplot(122)
        plt.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%')
        plt.title('Loan Condition (%)')  # 贷款状态占比
        
        self.save_plot('loan_condition_distribution')
        print("✓ Loan condition plots done")  # 贷款状态图表已完成
        
    def plot_numeric_vars(self, df: pd.DataFrame, columns: list):
        """Plot numeric variable distribution"""
        for col in columns:
            fig = plt.figure(figsize=(15, 5))
            
            # Left: histogram
            plt.subplot(121)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'{col} Distribution')  # 变量分布
            
            # Add statistical information
            stats = df[col].describe()
            stats_text = (f'mean: {stats["mean"]:.2f}\n'
                         f'std: {stats["std"]:.2f}\n'
                         f'median: {stats["50%"]:.2f}\n'
                         f'min: {stats["min"]:.2f}\n'
                         f'max: {stats["max"]:.2f}')
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Right: box plot
            plt.subplot(122)
            sns.boxplot(data=df, y=col)
            plt.title('Box Plot')  # 箱型图
            
            self.save_plot(f'{col}_distribution')
            print(f"✓ {col} plots done")  # 变量图表已完成
            
    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Plot correlation matrix for numeric variables
        
        Args:
            df: DataFrame containing the variables
        """
        # 获取所有数值列，但排除_cat结尾的
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        vars_to_plot = [col for col in numeric_cols if not col.endswith('_cat')]
        
        if len(vars_to_plot) >= 2:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[vars_to_plot].corr(), 
                       annot=True,      # 显示具体数值
                       cmap='RdBu_r',   # 使用红蓝配色
                       center=0,        # 相关系数0值对应白色
                       fmt='.2f')       # 保留两位小数
            plt.title('Correlation Matrix (Numeric Variables)')
            self.save_plot('correlation_matrix')
            print("✓ Correlation plot done")
        
    def plot_3d_scatter(self, df: pd.DataFrame):
        """Plot 3D scatter plot"""
            
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点
        scatter = ax.scatter(df['interest_rate'],
                            df['loan_amount'],
                            df['annual_inc'],
                            c=df['loan_condition_cat'],
                            cmap='coolwarm',
                            alpha=0.6)
            
        # 设置轴标签
        ax.set_xlabel('Interest Rate (%)')    # 利率
        ax.set_ylabel('Loan Amount ($)')      # 贷款金额
        ax.set_zlabel('Annual Income ($)')    # 年收入
        plt.title('Loan Features in 3D Space')    # 贷款特征的3D空间分布
        
        # 添加颜色条
        unique_conditions = sorted(df['loan_condition'].unique())
        plt.colorbar(scatter, label='Loan Condition', 
                    ticks=range(len(unique_conditions)),
                    boundaries=range(len(unique_conditions)))
        ax.figure.axes[-1].set_yticklabels(unique_conditions)
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        # 保存图形
        self.save_plot('3d_relationship')
        print("✓ 3D散点图生成完成")
            
        
        
    def plot_pairwise_relationships(self, df: pd.DataFrame, vars_to_plot: list):
        """Plot pairwise relationships using seaborn's pairplot
        
        Args:
            df: DataFrame containing the variables
            vars_to_plot: List of variable names to include in the plot
        """
        # 使用seaborn的pairplot直接生成配对图
        g = sns.pairplot(df[vars_to_plot], 
                        diag_kind='kde',  # 对角线显示核密度估计
                        plot_kws={'alpha': 0.6})  # 设置散点图透明度
        
        # 设置标题
        g.fig.suptitle('Feature Pairwise Relationships', y=1.02, fontsize=16)
        
        # 保存图形
        self.save_plot('pairwise_relationships')
        print("✓ Feature pairs plot done")  # 特征配对图表已完成
        
    def plot_categorical_dist(self, df: pd.DataFrame, cat_col: str):
        """Plot categorical variable distribution"""
        plt.figure(figsize=(15, 6))
        
        # Left: bar chart
        plt.subplot(121)
        value_counts = df[cat_col].value_counts()
            
        ax1 = sns.barplot(x=range(len(value_counts)), y=value_counts.values)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        # Add labels
        total = len(df)
        for i, v in enumerate(value_counts.values):
            ax1.text(i, v, f'{v:,}\n({v/total*100:.1f}%)',
                    ha='center', va='bottom')
        
        plt.title(f'{cat_col} Distribution')  # 分类变量分布
        plt.xlabel(cat_col)
        plt.ylabel('Count')
        
        # Right: pie chart
        plt.subplot(122)
        plt.pie(value_counts.values, labels=value_counts.index,
                autopct='%1.1f%%')
        plt.title(f'{cat_col} (%)')  # 分类变量占比
        
        self.save_plot(f'{cat_col}_distribution')
        print(f"✓ {cat_col} plots done")  # 分类变量图表已完成
        
    def plot_categorical_target(self, df: pd.DataFrame, cat_col: str, target_col: str):
        """Plot categorical variable vs target variable"""
        plt.figure(figsize=(12, 6))
        
        if df[target_col].dtype in ['int64', 'float64']:
            # If target variable is numeric, use box plot
            sns.boxplot(x=cat_col, y=target_col, data=df)
            plt.xticks(rotation=45)
            plt.title(f'{cat_col} vs {target_col}')  # 变量间的关系
        else:
            # If target variable is categorical, use stacked bar chart
            crosstab = pd.crosstab(df[cat_col], df[target_col], normalize='index')
            crosstab.plot(kind='bar', stacked=True)
            plt.xticks(rotation=45)
            plt.title(f'{cat_col} vs {target_col}')  # 变量间的关系
        
        plt.xticks(rotation=45)
        
        self.save_plot(f'{cat_col}_vs_{target_col}')
        print(f"✓ {cat_col} and {target_col} plot done")  # 关系图表已完成