"""
回归模型训练器
Regression model trainer for loan data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import time

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.base import clone

class RegressionTrainer:
    """回归模型训练器"""
    
    def __init__(self, output_dir='data/processed', outer_cv=5, inner_cv=5):
        """初始化回归模型训练器
        
        参数:
            output_dir: 输出目录
            outer_cv: 外部交叉验证折数
            inner_cv: 内部交叉验证折数
        """
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')  # 越小越好（MSE）
        self.scaler = StandardScaler()
        self.output_dir = Path(output_dir)
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备训练和测试数据
        
        参数:
            X: 特征矩阵
            y: 目标变量
            
        返回:
            训练集和测试集的特征矩阵和目标变量
        """
        # 1. 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 2. 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_linear_regression_nested_cv(self, X, y):
        """使用嵌套交叉验证训练并评估线性回归模型"""
        model = LinearRegression()
        
        # 线性回归没有超参数，所以只需执行外部交叉验证
        outer_cv = KFold(n_splits=self.outer_cv, shuffle=True, random_state=42)
        outer_scores = []
        
        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练模型
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            # 评估性能
            y_pred = model_clone.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            outer_scores.append(rmse)
        
        avg_rmse = np.mean(outer_scores)
        print(f"线性回归嵌套CV评估: RMSE={avg_rmse:.4f} (std={np.std(outer_scores):.4f})")
        
        # 在全部数据上训练最终模型
        final_model = clone(model)
        final_model.fit(X, y)
        
        # 计算平均 R² 值
        avg_r2 = np.mean([r2_score(y[test_idx], clone(final_model).fit(X[train_idx], y[train_idx]).predict(X[test_idx])) 
                         for train_idx, test_idx in outer_cv.split(X)])
        
        return final_model, {'rmse': avg_rmse, 'r2': avg_r2}
    
    def train_model_with_nested_cv(self, model_name, base_model, param_grid, X, y):
        """使用嵌套交叉验证训练并评估模型
        
        参数:
            model_name: 模型名称
            base_model: 基础模型
            param_grid: 超参数网格
            X: 特征矩阵
            y: 目标变量
            
        返回:
            训练好的最终模型和评估指标字典
        """
        start_time = time.time()
        outer_cv = KFold(n_splits=self.outer_cv, shuffle=True, random_state=42)
        
        outer_scores = []
        best_params_list = []
        
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            print(f"\n{model_name}: 外部CV折 {i+1}/{self.outer_cv}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 内部交叉验证找最佳超参数
            inner_cv = KFold(n_splits=self.inner_cv, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                clone(base_model), 
                param_grid, 
                cv=inner_cv, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            # 记录当前外部折的最佳超参数
            best_params = grid_search.best_params_
            best_params_list.append(best_params)
            print(f"折 {i+1} 最佳超参数: {best_params}")
            
            # 使用最佳超参数在当前外部折训练模型
            best_model = grid_search.best_estimator_
            
            # 在测试集上评估
            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)  # 计算R²
            outer_scores.append(rmse)
            print(f"折 {i+1} RMSE: {rmse:.4f}, R²: {r2:.4f}")  # 添加R²输出
        
        avg_rmse = np.mean(outer_scores)
        print(f"\n{model_name} 嵌套CV评估: RMSE={avg_rmse:.4f} (std={np.std(outer_scores):.4f})")
        
        # 超参数投票: 统计哪些超参数在各折中出现最频繁
        param_votes = {}
        for params in best_params_list:
            param_str = str(params)
            if param_str not in param_votes:
                param_votes[param_str] = 1
            else:
                param_votes[param_str] += 1
        
        # 获得票数最高的超参数组合
        best_param_str = max(param_votes, key=param_votes.get)
        # 将字符串转回字典形式
        import ast
        final_params = ast.literal_eval(best_param_str.replace("'", '"'))
        print(f"最终选择的超参数: {final_params} (出现频率: {param_votes[best_param_str]}/{self.outer_cv})")
        
        # 在全部数据上训练最终模型
        final_model = clone(base_model)
        # 设置最佳超参数
        final_model.set_params(**final_params)
        final_model.fit(X, y)
        
        elapsed = time.time() - start_time
        print(f"{model_name} 训练完成，耗时: {elapsed:.2f}秒")
        
        # 计算平均 R² 值
        avg_r2 = np.mean([r2_score(y[test_idx], clone(final_model).fit(X[train_idx], y[train_idx]).predict(X[test_idx])) 
                         for train_idx, test_idx in outer_cv.split(X)])
        
        # 返回模型和评估指标字典
        return final_model, {'rmse': avg_rmse, 'r2': avg_r2}
    
    def train_ridge_nested_cv(self, X, y):
        """使用嵌套交叉验证训练岭回归模型"""
        model = Ridge(random_state=42)
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        return self.train_model_with_nested_cv('Ridge', model, param_grid, X, y)
    
    def train_lasso_nested_cv(self, X, y):
        """使用嵌套交叉验证训练Lasso回归模型"""
        model = Lasso(random_state=42)
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
        return self.train_model_with_nested_cv('Lasso', model, param_grid, X, y)
    
    def train_elastic_net_nested_cv(self, X, y):
        """使用嵌套交叉验证训练弹性网络模型"""
        model = ElasticNet(random_state=42)
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        return self.train_model_with_nested_cv('ElasticNet', model, param_grid, X, y)
    
    def train_random_forest_nested_cv(self, X, y):
        """使用嵌套交叉验证训练随机森林回归模型"""
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 8, 12],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', None]
        }
        return self.train_model_with_nested_cv('RandomForest', model, param_grid, X, y)
    
    def train_gradient_boosting_nested_cv(self, X, y):
        """使用嵌套交叉验证训练梯度提升回归模型"""
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0]
        }
        return self.train_model_with_nested_cv('GradientBoosting', model, param_grid, X, y)
    
    def train_all_models_nested_cv(self, X, y):
        """使用嵌套交叉验证训练所有回归模型"""
        print("开始嵌套交叉验证训练所有模型...")
        models = {}
        results = {}
        
        # 线性回归
        print("\n======= 训练线性回归 =======")
        models['LinearRegression'], results['LinearRegression'] = self.train_linear_regression_nested_cv(X, y)
        
        # 岭回归
        print("\n======= 训练岭回归 =======")
        models['Ridge'], results['Ridge'] = self.train_ridge_nested_cv(X, y)
        
        # Lasso回归
        print("\n======= 训练Lasso回归 =======")
        models['Lasso'], results['Lasso'] = self.train_lasso_nested_cv(X, y)
        
        # 弹性网络
        print("\n======= 训练弹性网络 =======")
        models['ElasticNet'], results['ElasticNet'] = self.train_elastic_net_nested_cv(X, y)
        
        # 随机森林
        print("\n======= 训练随机森林 =======")
        models['RandomForest'], results['RandomForest'] = self.train_random_forest_nested_cv(X, y)
        
        # 梯度提升
        print("\n======= 训练梯度提升 =======")
        models['GradientBoosting'], results['GradientBoosting'] = self.train_gradient_boosting_nested_cv(X, y)
        
        self.models = models
        
        # 找出最佳模型
        self.best_model = min(results, key=lambda x: results[x]['rmse'])
        self.best_score = results[self.best_model]['rmse']
        
        # 可视化结果
        self.visualize_nested_cv_results(results)
        
        # 保存最佳模型
        joblib.dump(self.models[self.best_model], self.output_dir / 'best_model.pkl')
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        
        print(f"\n最佳模型: {self.best_model}，嵌套CV RMSE={self.best_score:.4f}")
        
        return models, results
    
    def visualize_nested_cv_results(self, results):
        """可视化嵌套交叉验证结果"""
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[model]['rmse'] for model in results],
            'R²': [results[model]['r2'] for model in results]
        }).sort_values('RMSE')
        
        # 保存到CSV
        results_df.to_csv(self.output_dir / 'nested_cv_results.csv', index=False)
        
        # 绘制评估结果
        plt.figure(figsize=(10, 6))
        bars = plt.barh(results_df['Model'], results_df['RMSE'])
        
        # 为每个条形添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    ha='left', va='center')
        
        plt.title('nested_cv RMSE Comparison of Regression Models (Lower is Better)')
        plt.xlabel('RMSE')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'nested_cv_model_comparison.png')
        plt.close()
    
    def run_workflow(self, X, y):
        """运行完整的训练和评估流程
        
        参数:
            X: 特征矩阵
            y: 目标变量
            
        返回:
            评估结果
        """
        print("\n开始执行嵌套交叉验证训练流程...")
        
        # 1. 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. 使用嵌套CV训练和评估所有模型
        _, results = self.train_all_models_nested_cv(X_scaled, y)
        
        return results
    
    # 保留原始方法用于向后兼容
    def train_linear_regression(self, X_train, y_train):
        """训练线性回归模型"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def train_ridge(self, X_train, y_train, params=None):
        """训练岭回归模型"""
        if params is None:
            params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        
        model = Ridge(random_state=42)
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        
        print(f"最佳岭回归超参数: {grid.best_params_}")
        return grid.best_estimator_
    
    def train_lasso(self, X_train, y_train, params=None):
        """训练Lasso回归模型"""
        if params is None:
            params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
        
        model = Lasso(random_state=42)
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        
        print(f"最佳Lasso超参数: {grid.best_params_}")
        return grid.best_estimator_
    
    def train_elastic_net(self, X_train, y_train, params=None):
        """训练弹性网络模型"""
        if params is None:
            params = {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        
        model = ElasticNet(random_state=42)
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        
        print(f"最佳弹性网络超参数: {grid.best_params_}")
        return grid.best_estimator_
    
    def train_random_forest(self, X_train, y_train, params=None):
        """训练随机森林回归模型"""
        if params is None:
            params = {
                'n_estimators': [100, 200],  
                'max_depth': [None, 8, 12],  
                'min_samples_split': [2, 5],  # 分裂所需最小样本数
                'min_samples_leaf': [1, 2],  # 叶节点最小样本数
                'max_features': ['sqrt', None]  # 特征选择策略
            }
        
        model = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        
        print(f"最佳随机森林超参数: {grid.best_params_}")
        return grid.best_estimator_
    
    def train_gradient_boosting(self, X_train, y_train, params=None):
        """训练梯度提升回归模型"""
        if params is None:
            params = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0]  # 样本抽样比例（降低过拟合）
            }
        
        model = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        
        print(f"最佳梯度提升超参数: {grid.best_params_}")
        return grid.best_estimator_
    
    def train_svr(self, X_train, y_train, params=None):
        """训练支持向量回归模型"""
        if params is None:
            params = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto']
            }
        
        model = SVR()
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        
        print(f"最佳SVR超参数: {grid.best_params_}")
        return grid.best_estimator_
    
    def train_all_models(self, X_train, y_train):
        """训练所有回归模型 (传统方法，非嵌套CV)"""
        models = {
            'LinearRegression': self.train_linear_regression(X_train, y_train),
            'Ridge': self.train_ridge(X_train, y_train),
            'Lasso': self.train_lasso(X_train, y_train),
            'ElasticNet': self.train_elastic_net(X_train, y_train),
            'RandomForest': self.train_random_forest(X_train, y_train),
            'GradientBoosting': self.train_gradient_boosting(X_train, y_train)
        }
        
        self.models = models
        return models
    
    def evaluate_models(self, X_test, y_test):
        """评估所有模型并选择最佳模型 (传统方法，非嵌套CV)"""
        results = {}
        
        for name, model in self.models.items():
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"{name} 模型性能: RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # 更新最佳模型
            if rmse < self.best_score:
                self.best_score = rmse
                self.best_model = name
        
        # 保存评估结果
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[model]['rmse'] for model in results],
            'R²': [results[model]['r2'] for model in results]
        }).sort_values('RMSE')
        
        # 绘制评估结果
        plt.figure(figsize=(10, 6))
        bars = plt.barh(results_df['Model'], results_df['RMSE'])
        
        # 为每个条形添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    ha='left', va='center')
        
        plt.title('RMSE Comparison of Regression Models (Lower is Better)')
        plt.xlabel('RMSE')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png')
        plt.close()
        
        # 保存最佳模型
        best_model = self.models[self.best_model]
        joblib.dump(best_model, self.output_dir / 'best_model.pkl')
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        
        print(f"\n最佳模型: {self.best_model}，RMSE={self.best_score:.4f}")
        return results

if __name__ == "__main__":
    print("回归模型训练器模块。在工作流中导入并使用。")
