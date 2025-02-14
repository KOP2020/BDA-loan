import pandas as pd

# 读取原始 CSV
print("正在读取原始数据集...")
df = pd.read_csv('data/raw/loan_final313.csv')

# 保存为压缩格式
print("正在压缩数据集...")
compressed_file = 'data/raw/loan_final313.csv.gz'
df.to_csv(compressed_file, compression='gzip', index=False)

# 验证压缩后的文件大小
import os
original_size = os.path.getsize('data/raw/loan_final313.csv.gz') / (1024 * 1024)  # MB
compressed_size = os.path.getsize(compressed_file) / (1024 * 1024)  # MB
print(f"原始文件大小: {original_size:.2f} MB")
print(f"压缩后文件大小: {compressed_size:.2f} MB")