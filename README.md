# loan数据分析项目

这是一个ml的贷款数据分析项目，关注BDA全流程。

## 项目结构

```
├── config/             # 配置文件目录
├── data/              # 数据目录
│   ├── raw/          # 原始数据
│   ├── processed/    # 处理后的数据
├── docs/              # 文档目录
├── src/               # 源代码目录
│   ├── data/         # 数据处理相关代码
│   ├── features/     # 特征工程相关代码
│   ├── models/       # 模型相关代码
│   ├── visualization/# 可视化相关代码
└── tests/             # 测试代码目录
```

## 项目设置

1. conda创建虚拟环境：
```bash
conda create -n loan-analysis python=3.8
conda activate loan-analysis
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 核心组件使用指南

项目使用一个完整的分析流程，所有核心组件都在主脚本 `main.py` 中集成运行。下面是主要组件及其功能：

### 1. 数据预处理 (preprocessor.py)

`src/data/preprocessor.py` 负责数据清洗和预处理工作，包括：
- 筛选2007-2010年的数据
- 删除不需要的列
- 处理分类变量

可以单独运行预处理：
```bash
# 单独运行数据预处理
python src/data/preprocessor.py
```

### 2. 特征选择 (feature_selector.py)

`src/features/feature_selector.py` 使用递归特征消除(RFE)方法选择最重要的特征：
- 自动过滤出数值特征
- 根据重要性选择指定数量的特征
- 生成特征重要性图表

**注意**：此组件通常不单独运行，而是在main.py工作流中被调用。当然，你可以自己写单独运行它的指令

### 3. 模型训练与评估 (trainer.py)

`src/models/trainer.py` 包含 `RegressionTrainer` 类，用于训练和评估多种回归模型：
- 线性回归、岭回归、Lasso回归、弹性网络
- 随机森林、梯度提升回归
- 自动进行超参数调优
- 评估模型性能并选择最佳模型

**注意**：在main.py工作流中被调用。

### 4. 一键运行全流程 (main.py)

`main.py` 是项目的主入口点，它集成了上述所有组件为一个完整的BDA分析流程：

1. 首先加载预处理后的数据
2. 然后调用feature_selector进行特征选择（默认选择7个最重要的特征）
3. 最后调用RegressionTrainer训练和评估所有模型，并选出最佳模型

使用方法非常简单：
```bash
# 运行整个分析流程
python main.py
```

运行后将在 `data/processed` 目录生成以下文件：
- `features_selected.csv`: 特征选择后的数据
- `feature_importance.png`: 特征重要性图表
- `regression_results.csv`: 各模型性能指标
- `model_comparison.png`: 模型性能比较图
- `best_model.pkl`: 保存的最佳模型

## 项目输出

完整运行后，项目将输出以下分析结果：
1. 选择的最重要特征列表
2. 各回归模型的性能指标（RMSE和R²值）
3. 最佳模型及其超参数
4. 模型性能比较可视化图表
