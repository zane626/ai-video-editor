"""
进度显示和日志工具
"""

import sys
import time
from typing import Optional


class ProgressTracker:
    """简单的进度追踪器"""

    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self._last_print_len = 0

    def update(self, n: int = 1):
        self.current += n
        self._print()

    def _print(self):
        elapsed = time.time() - self.start_time
        pct = self.current / self.total if self.total > 0 else 0

        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = _format_time(eta)
        else:
            eta_str = "--:--"

        bar_len = 30
        filled = int(bar_len * pct)
        bar = "=" * filled + ">" + " " * (bar_len - filled - 1)

        msg = f"\r{self.desc}: [{bar}] {pct*100:5.1f}% ({self.current}/{self.total}) ETA: {eta_str}"
        # 用空格覆盖多余字符
        padding = max(0, self._last_print_len - len(msg))
        sys.stdout.write(msg + " " * padding)
        sys.stdout.flush()
        self._last_print_len = len(msg)

        if self.current >= self.total:
            total_str = _format_time(elapsed)
            sys.stdout.write(f"\n{self.desc}: 完成，耗时 {total_str}\n")
            sys.stdout.flush()

    def finish(self):
        self.current = self.total
        self._print()


def log_info(msg: str):
    """普通日志"""
    print(f"[INFO] {msg}")


def log_warn(msg: str):
    """警告日志"""
    print(f"[WARN] {msg}")


def log_error(msg: str):
    """错误日志"""
    print(f"[ERROR] {msg}", file=sys.stderr)


def log_stage(stage_num: int, msg: str):
    """Stage 分隔日志"""
    print(f"\n{'='*60}")
    print(f"  Stage {stage_num}: {msg}")
    print(f"{'='*60}")


def _format_time(seconds: float) -> str:
    """格式化秒数为 MM:SS 或 HH:MM:SS"""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
