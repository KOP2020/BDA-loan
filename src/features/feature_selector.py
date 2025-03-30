"""
特征选择模块 - 使用包装器方法进行特征选择
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

N_FEATURES = 7

def rfe_selection(X, y, n_features=N_FEATURES, estimator_type='rf', output_dir='data/processed'):
    """使用递归特征消除法(RFE)选择特征
    
    参数:
        X: 特征矩阵（必须是数值型）
        y: 目标变量
        n_features: 要选择的特征数量
        estimator_type: 'rf'为随机森林, 'linear'为线性回归
        output_dir: 输出目录
    
    返回:
        选中特征列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据指定类型选择基础估计器
    if estimator_type == 'rf':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        estimator = LinearRegression()
    
    # 使用RFE选择特征
    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    
    # 获取选中的特征名称
    selected_features = X.columns[selector.support_].tolist()
    
    # 如果是随机森林，绘制特征重要性
    if estimator_type == 'rf':
        # 使用选中的特征训练随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[selected_features], y)
        
        # 绘制特征重要性
        importances = pd.DataFrame({
            'feature': selected_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importances['feature'], importances['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png')
        plt.close()
    
    # 保存选中的特征
    joblib.dump(selected_features, output_dir / 'selected_features.pkl')
    
    print(f"选择了{len(selected_features)}个特征: {selected_features}")
    
    return selected_features

def filter_numeric_features(df, target_col):
    """过滤出数值特征，包括自然数值特征和独热编码特征
    
    参数:
        df: 包含特征和目标变量的数据框
        target_col: 目标变量列名
    
    返回:
        只包含数值特征的数据框
    """
    # 创建特征矩阵的副本
    X = df.drop(columns=[target_col]).copy()
    
    # 识别原始分类特征（将被排除）
    categorical_cols = []
    
    # 独热编码特征的前缀列表 - 基于我们的数据预处理流程
    one_hot_prefixes = [
        'grade_', 'purpose_', 'home_ownership_', 'term_',
        'interest_payments_', 'application_type_', 'income_category_'
    ]
    
    for col in X.columns:
        # 如果列是object或category类型，则标记为分类特征
        if X[col].dtype in ['object', 'category']:
            categorical_cols.append(col)
        # 保留特殊的原始列，如region（非数值但可能被单独处理）
        elif col == 'region':
            categorical_cols.append(col)
    
    print(f"将跳过的非数值特征: {categorical_cols}")
    
    # 只保留数值特征
    X_numeric = X.drop(columns=categorical_cols)
    
    # 确认所有剩余特征都是数值型
    for col in X_numeric.columns:
        if X_numeric[col].dtype not in ['int64', 'float64']:
            print(f"警告: '{col}'不是数值类型，将被转换为数值")
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    
    print(f"选择的数值特征: {X_numeric.columns.tolist()}")
    
    return X_numeric

def select_features_for_regression(df, target_col, n_features=10, output_dir='data/processed'):
    """为回归任务选择特征
    
    参数:
        df: 包含特征和目标变量的数据框
        target_col: 目标变量列名
        n_features: 要选择的特征数量
        output_dir: 输出目录
    
    返回:
        包含选中特征和目标变量的数据框，以及选中的特征列表
    """
    output_dir = Path(output_dir)
    
    # 过滤出数值特征
    X = filter_numeric_features(df, target_col)
    y = df[target_col]
    
    # 使用RFE选择特征
    selected_features = rfe_selection(X, y, n_features, 'rf', output_dir)
    
    # 返回包含选中特征和目标变量的数据框
    return df[selected_features + [target_col]], selected_features

if __name__ == "__main__":
    # 示例用法
    print("特征选择模块。导入并在工作流中使用。")
