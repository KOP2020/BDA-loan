"""
贷款数据分析入口脚本
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入数据探索模块
from src.data.data_exploration import LoanDataExplorer
from src.visualization.visualizer import LoanVisualizer

def main():
    """主函数"""
    # 初始化数据探索器和可视化器
    explorer = LoanDataExplorer()
    visualizer = LoanVisualizer()
    
    # 执行数据探索
    explorer.run_analysis()

if __name__ == "__main__":
    main()
