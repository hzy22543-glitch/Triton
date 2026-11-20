"""Run the local correctness suite without requiring a package installation."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 把源码目录加入当前进程的导入搜索路径，才能 import w10_triton。
# 只影响这次 Python 进程，不修改系统环境，也不安装包。
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    # 以主程序身份执行测试文件，触发其末尾 main；命令行选项继续由它解析。
    runpy.run_path(str(ROOT / 'tests' / 'correctness.py'), run_name='__main__')
