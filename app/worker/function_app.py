import sys
from pathlib import Path

worker_src = Path(__file__).resolve().parent / "src"
if str(worker_src) not in sys.path:
    sys.path.insert(0, str(worker_src))

from rag_worker.function_app import app

__all__ = ["app"]
