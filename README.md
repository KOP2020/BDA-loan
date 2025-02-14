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

## 开发路线

- 探索性数据分析（EDA）
- 特征工程和选择
- 模型训练与评估
- 模型解释
