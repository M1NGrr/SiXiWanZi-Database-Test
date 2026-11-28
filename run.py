"""顶层快捷启动脚本: python run.py <command> ..."""

from __future__ import annotations

import sys

from sixiwanzi_corpus.cli import main

if __name__ == "__main__":
    sys.exit(main())
