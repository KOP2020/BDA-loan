# loan数据分析项目

这是一个基于传统ml的贷款数据分析项目，关注数据可视化和分析过程。

## 项目结构

```
├── config/             # 配置文件目录
├── data/              # 数据目录
│   ├── raw/          # 原始数据
│   ├── processed/    # 处理后的数据
├── docs/              # 文档目录
├── notebooks/         # Jupyter notebooks目录
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

## 特征工程Pipeline

我们的特征工程pipeline遵循以下顺序：

### 1. 特征创建/提取 (Feature Creation/Extraction)
- 基础特征创建
  - 时间特征（如贷款期限的年月特征）
  - 数值特征的统计特征（均值、方差等）
  - 类别特征的编码特征
- 高级特征创建
  - 特征组合（如收入负债比）
  - 基于领域知识的特征
  - 基于聚类的特征（计算样本到各类群体中心的距离）

### 2. 特征变换 (Feature Transformation)
- 缺失值处理
  - 数值型：均值/中位数填充
  - 类别型：众数/特殊类别填充
- 异常值处理
  - 基于IQR的异常值检测
  - 基于领域知识的异常值处理
- 编码转换
  - 类别特征：One-hot/Label encoding
  - 时间特征：周期性编码
- 标准化/归一化
  - StandardScaler：均值为0，方差为1
  - MinMaxScaler：缩放到特定区间

### 3. 特征选择 (Feature Selection)
- 单变量特征选择（Univariate Selection）
  - 基于统计检验选择最相关特征
- 递归特征消除（RFE）
  - 使用模型反复训练，逐步消除不重要特征
- 基于模型的特征选择
  - 使用树模型的特征重要性
  - L1正则化（Lasso）的特征筛选

### 4. 降维 (Dimensionality Reduction)
- PCA（Principal Component Analysis）
  - 保留解释95%方差所需的主成分
  - 用于处理高度相关的特征
- 其他降维方法（根据需要）
  - LDA（Linear Discriminant Analysis）
    - LDA是有监督的降维方法：它会考虑类别信息，找到最能区分不同类别的方向
  - t-SNE（用于可视化）

### Pipeline实现
- 使用sklearn的Pipeline确保特征工程步骤的一致性
- 所有参数都在config文件中配置
- 提供特征重要性分析和可视化
- 支持特征工程步骤的灵活组合

## 开发路线

- 探索性数据分析（EDA）
- 特征工程和选择
- 模型训练与评估
- 模型解释
