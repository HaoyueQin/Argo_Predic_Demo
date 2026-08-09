"""pytest 配置：确保仓库根目录在 sys.path，使测试可以 import models/ 与 scripts/。"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
