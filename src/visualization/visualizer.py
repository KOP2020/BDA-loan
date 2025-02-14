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
    def __init__(self, use_brown_theme=True):
        """Initialize the visualizer
        
        Args:
            use_brown_theme (bool): 是否使用棕色主题，默认为True
        """
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
        
        # 设置主题
        self.use_brown_theme = use_brown_theme
        if self.use_brown_theme:
            # 定义棕色系配色方案
            self.colors = ["#2B1B10", "#4E342E", "#795548", "#A1887F", "#D7CCC8"]
            self.cmap = plt.cm.colors.LinearSegmentedColormap.from_list("brown", self.colors)
            self.bg_color = '#FAF6F3'  # 浅米色背景
            # 设置seaborn的配色方案
            sns.set_palette(self.colors)
        else:
            # 使用蓝白色系
            blue_colors = ["#1f77b4", "#3498db", "#5dade2", "#85c1e9", "#aed6f1"]
            self.colors = blue_colors
            self.cmap = plt.cm.colors.LinearSegmentedColormap.from_list("blues", blue_colors)
            self.bg_color = '#f7fbff'  # 非常浅的蓝色背景
            sns.set_palette(blue_colors)
    
    def _set_fig_style(self, fig):
        """设置图形样式
        
        Args:
            fig: matplotlib figure对象
        """
        if self.use_brown_theme:
            fig.patch.set_facecolor(self.bg_color)
    
    def save_plot(self, name: str):
        """Save the plot"""
        plt.savefig(self.figures_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_loan_condition(self, df: pd.DataFrame):
        """Plot loan condition distribution"""
        fig = plt.figure(figsize=(15, 6))
        self._set_fig_style(fig)
        
        # Left: bar chart
        plt.subplot(121)
        condition_counts = df['loan_condition'].value_counts()
            
        ax1 = sns.barplot(x=range(len(condition_counts)), y=condition_counts.values, 
                         palette=self.colors[:len(condition_counts)])
        plt.xticks(range(len(condition_counts)), condition_counts.index, rotation=45)
        
        # Add count and percentage labels
        total = len(df)
        for i, v in enumerate(condition_counts.values):
            ax1.text(i, v, f'{v:,}\n({v/total*100:.1f}%)', ha='center', va='bottom')
        
        plt.title('Loan Condition Distribution')
        plt.xlabel('Condition')
        plt.ylabel('Count')
        
        # Right: pie chart
        plt.subplot(122)
        plt.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%',
                colors=self.colors[:len(condition_counts)])
        plt.title('Loan Condition (%)')
        
        self.save_plot('loan_condition_distribution')
        print("✓ Loan condition plots done")
        
    def plot_numeric_vars(self, df: pd.DataFrame, columns: list):
        """Plot numeric variable distribution"""
        for col in columns:
            fig = plt.figure(figsize=(15, 5))
            self._set_fig_style(fig)
            
            # Left: histogram
            plt.subplot(121)
            sns.histplot(data=df, x=col, kde=True, color=self.colors[2])
            plt.title(f'{col} Distribution')
            
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
            sns.boxplot(data=df, y=col, color=self.colors[2])
            plt.title('Box Plot')
            
            self.save_plot(f'{col}_distribution')
            print(f"✓ {col} plots done")
            
    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Plot correlation matrix for numeric variables"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        vars_to_plot = [col for col in numeric_cols if not col.endswith('_cat')]
        
        if len(vars_to_plot) >= 2:
            fig = plt.figure(figsize=(12, 10))
            self._set_fig_style(fig)
            
            # 创建相关性矩阵colormap
            if self.use_brown_theme:
                corr_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                    "brown_corr", ["#D7CCC8", "#FFFFFF", "#2B1B10"])
            else:
                corr_cmap = 'RdBu_r'
            
            sns.heatmap(df[vars_to_plot].corr(), 
                       annot=True,
                       cmap=corr_cmap,
                       center=0,
                       fmt='.2f')
            plt.title('Correlation Matrix (Numeric Variables)')
            self.save_plot('correlation_matrix')
            print("✓ Correlation plot done")
        
    def plot_3d_scatter(self, df: pd.DataFrame):
        """Plot 3D scatter plot"""
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        self._set_fig_style(fig)
        ax = fig.add_subplot(111, projection='3d')
        if self.use_brown_theme:
            ax.set_facecolor(self.bg_color)
        
        # 绘制散点
        scatter = ax.scatter(df['interest_rate'],
                           df['loan_amount'],
                           df['annual_inc'],
                           c=df['loan_condition_cat'],
                           cmap=self.cmap,
                           alpha=0.8)
            
        # 设置轴标签
        ax.set_xlabel('Interest Rate (%)')
        ax.set_ylabel('Loan Amount ($)')
        ax.set_zlabel('Annual Income ($)')
        plt.title('Loan Features in 3D Space')
        
        # 添加颜色条
        unique_conditions = sorted(df['loan_condition'].unique())
        plt.colorbar(scatter, label='Loan Condition', 
                    ticks=range(len(unique_conditions)),
                    boundaries=range(len(unique_conditions)))
        ax.figure.axes[-1].set_yticklabels(unique_conditions)
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        self.save_plot('3d_relationship')
        print("✓ 3D scatter plot done")
            
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
        fig = plt.figure(figsize=(15, 6))
        self._set_fig_style(fig)
        
        # Left: bar chart
        plt.subplot(121)
        value_counts = df[cat_col].value_counts()
            
        # 新的barplot调色方式
        ax1 = sns.barplot(x=range(len(value_counts)), y=value_counts.values,
                         hue=range(len(value_counts)),
                         palette=self.colors[:len(value_counts)],
                         legend=False)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        # Add labels
        total = len(df)
        for i, v in enumerate(value_counts.values):
            ax1.text(i, v, f'{v:,}\n({v/total*100:.1f}%)',
                    ha='center', va='bottom')
        
        plt.title(f'{cat_col} Distribution')
        plt.xlabel(cat_col)
        plt.ylabel('Count')
        
        # Right: pie chart
        plt.subplot(122)
        plt.pie(value_counts.values, labels=value_counts.index,
                autopct='%1.1f%%', colors=self.colors[:len(value_counts)])
        plt.title(f'{cat_col} (%)')
        
        self.save_plot(f'{cat_col}_distribution')
        print(f"✓ {cat_col} plots done")
        
    def plot_categorical_target(self, df: pd.DataFrame, cat_col: str, target_col: str):
        """Plot categorical variable vs target variable"""
        fig = plt.figure(figsize=(12, 6))
        self._set_fig_style(fig)
        
        if df[target_col].dtype in ['int64', 'float64']:
            # If target variable is numeric, use box plot
            sns.boxplot(x=cat_col, y=target_col, data=df, 
                       hue=cat_col,
                       palette=self.colors[:len(df[cat_col].unique())],
                       legend=False)
            plt.xticks(rotation=45)
            plt.title(f'{cat_col} vs {target_col}')
        else:
            # If target variable is categorical, use stacked bar chart
            crosstab = pd.crosstab(df[cat_col], df[target_col], normalize='index')
            crosstab.plot(kind='bar', stacked=True, 
                         color=self.colors[:len(df[target_col].unique())])
            plt.xticks(rotation=45)
            plt.title(f'{cat_col} vs {target_col}')
        
        plt.xticks(rotation=45)
        
        self.save_plot(f'{cat_col}_vs_{target_col}')
        print(f"✓ {cat_col} and {target_col} plot done")