Loading processed data...
Data loaded: 20,814 records with 41 columns

Performing feature selection...
将跳过的非数值特征: ['region']
选择的数值特征: ['emp_length_int', 'annual_inc', 'loan_amount', 'interest_rate', 'dti', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding', 'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'term_ 36 months', 'term_ 60 months', 'interest_payments_High', 'interest_payments_Low', 'application_type_INDIVIDUAL', 'income_category_High', 'income_category_Low', 'income_category_Medium']
选择了7个特征: ['emp_length_int', 'annual_inc', 'loan_amount', 'interest_rate', 'dti', 'purpose_debt_consolidation', 'home_ownership_MORTGAGE']
Selected features saved to 'data/processed/features_selected.csv'

Training regression models...

开始执行嵌套交叉验证训练流程...
开始嵌套交叉验证训练所有模型...

======= 训练线性回归 =======
线性回归嵌套CV评估: RMSE=3248.1886 (std=72.0123)

======= 训练岭回归 =======

Ridge: 外部CV折 1/5
折 1 最佳超参数: {'alpha': 0.01}
折 1 RMSE: 3319.5895, R²: 0.7122

Ridge: 外部CV折 2/5
折 2 最佳超参数: {'alpha': 0.01}
折 2 RMSE: 3188.4738, R²: 0.7254

Ridge: 外部CV折 3/5
折 3 最佳超参数: {'alpha': 0.01}
折 3 RMSE: 3137.3521, R²: 0.7382

Ridge: 外部CV折 4/5
折 4 最佳超参数: {'alpha': 0.01}
折 4 RMSE: 3292.0731, R²: 0.7110

Ridge: 外部CV折 5/5
折 5 最佳超参数: {'alpha': 0.01}
折 5 RMSE: 3303.4544, R²: 0.7087

Ridge 嵌套CV评估: RMSE=3248.1886 (std=72.0123)
最终选择的超参数: {'alpha': 0.01} (出现频率: 5/5)
Ridge 训练完成，耗时: 4.44秒

======= 训练Lasso回归 =======

Lasso: 外部CV折 1/5
折 1 最佳超参数: {'alpha': 1.0}
折 1 RMSE: 3319.6040, R²: 0.7122

Lasso: 外部CV折 2/5
折 2 最佳超参数: {'alpha': 1.0}
折 2 RMSE: 3188.4800, R²: 0.7254

Lasso: 外部CV折 3/5
折 3 最佳超参数: {'alpha': 1.0}
折 3 RMSE: 3137.3758, R²: 0.7382

Lasso: 外部CV折 4/5
折 4 最佳超参数: {'alpha': 1.0}
折 4 RMSE: 3292.0260, R²: 0.7110

Lasso: 外部CV折 5/5
折 5 最佳超参数: {'alpha': 1.0}
折 5 RMSE: 3303.4351, R²: 0.7087

Lasso 嵌套CV评估: RMSE=3248.1842 (std=71.9981)
最终选择的超参数: {'alpha': 1.0} (出现频率: 5/5)
Lasso 训练完成，耗时: 0.37秒

======= 训练弹性网络 =======

ElasticNet: 外部CV折 1/5
折 1 最佳超参数: {'alpha': 0.001, 'l1_ratio': 0.9}
折 1 RMSE: 3319.5771, R²: 0.7122

ElasticNet: 外部CV折 2/5
折 2 最佳超参数: {'alpha': 0.001, 'l1_ratio': 0.9}
折 2 RMSE: 3188.4807, R²: 0.7254

ElasticNet: 外部CV折 3/5
折 3 最佳超参数: {'alpha': 0.001, 'l1_ratio': 0.9}
折 3 RMSE: 3137.3732, R²: 0.7382

ElasticNet: 外部CV折 4/5
折 4 最佳超参数: {'alpha': 0.001, 'l1_ratio': 0.9}
折 4 RMSE: 3292.0697, R²: 0.7110

ElasticNet: 外部CV折 5/5
折 5 最佳超参数: {'alpha': 0.001, 'l1_ratio': 0.9}
折 5 RMSE: 3303.4447, R²: 0.7087

ElasticNet 嵌套CV评估: RMSE=3248.1891 (std=72.0003)
最终选择的超参数: {'alpha': 0.001, 'l1_ratio': 0.9} (出现频率: 5/5)
ElasticNet 训练完成，耗时: 1.10秒

======= 训练随机森林 =======

RandomForest: 外部CV折 1/5
折 1 最佳超参数: {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
折 1 RMSE: 3261.8840, R²: 0.7221

RandomForest: 外部CV折 2/5
折 2 最佳超参数: {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
折 2 RMSE: 3175.9719, R²: 0.7276

RandomForest: 外部CV折 3/5
折 3 最佳超参数: {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
折 3 RMSE: 3119.0738, R²: 0.7412

RandomForest: 外部CV折 4/5
折 4 最佳超参数: {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
折 4 RMSE: 3268.5738, R²: 0.7151

RandomForest: 外部CV折 5/5
折 5 最佳超参数: {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
折 5 RMSE: 3261.8964, R²: 0.7159

RandomForest 嵌套CV评估: RMSE=3217.4800 (std=59.9363)
最终选择的超参数: {'max_depth': 8, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200} (出现频率: 3/5)
RandomForest 训练完成，耗时: 171.20秒

======= 训练梯度提升 =======

GradientBoosting: 外部CV折 1/5
折 1 最佳超参数: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 1.0}
折 1 RMSE: 3258.7743, R²: 0.7227

GradientBoosting: 外部CV折 2/5
折 2 最佳超参数: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50, 'subsample': 1.0}
折 2 RMSE: 3173.9622, R²: 0.7279

GradientBoosting: 外部CV折 3/5
折 3 最佳超参数: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8}
折 3 RMSE: 3102.0967, R²: 0.7440

GradientBoosting: 外部CV折 4/5
折 4 最佳超参数: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8}
折 4 RMSE: 3267.1509, R²: 0.7153

GradientBoosting: 外部CV折 5/5
折 5 最佳超参数: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 1.0}
折 5 RMSE: 3244.4841, R²: 0.7190

GradientBoosting 嵌套CV评估: RMSE=3209.2936 (std=62.8867)
最终选择的超参数: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 1.0} (出现频率: 2/5)
GradientBoosting 训练完成，耗时: 61.49秒

最佳模型: GradientBoosting，嵌套CV RMSE=3209.2936

Summary of regression analysis:
Best model: GradientBoosting
Best RMSE: 3209.2936

Regression analysis completed in 464.8 seconds
Results saved to 'data/processed/regression_results.csv'