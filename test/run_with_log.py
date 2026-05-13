"""
run_with_log.py
================
包装器：把考试目录里的 run.py 当子进程跑，stdout / stderr 实时打到控制台
同时落盘到 ../../output/<时间戳>/run.log，方便回看与对比改进。

特点
----
- 不改任何考试文件（run.py / solution.py / harness_base.py / llm_client.py 全部原样调用）
- 时间戳目录格式 YYYYMMDD_HHMMSS（精确到秒，足够区分多次运行）
- 同目录下还会落：
    run.log         —— 完整 stdout+stderr
    meta.json       —— 命令行参数、起止时间、退出码、host 等
    solution.py     —— 当前 solution.py 的副本（便于 diff 历次实验）
- 透传命令行参数：本脚本接收的所有参数都原样转给 run.py
    例：python test/run_with_log.py --runs 1 --workers 50

用法
----
    python test/run_with_log.py                         # 默认（4 轮，workers=20）
    python test/run_with_log.py --runs 1 --workers 50   # 快速试一次
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent                    # .../chuangzhi-summer-camp/test
EXAM_DIR = HERE.parent                                    # .../chuangzhi-summer-camp
OUTPUT_ROOT = EXAM_DIR.parent / "output"                  # .../student_package/output
RUN_PY = EXAM_DIR / "run.py"
SOLUTION_PY = EXAM_DIR / "solution.py"


def main() -> int:
    if not RUN_PY.exists():
        print(f"[run_with_log] FATAL: {RUN_PY} 不存在", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    meta_path = out_dir / "meta.json"

    if SOLUTION_PY.exists():
        shutil.copy2(SOLUTION_PY, out_dir / "solution.py")

    forwarded = sys.argv[1:]
    cmd = [sys.executable, "-u", str(RUN_PY), *forwarded]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    started = datetime.now()
    print(f"[run_with_log] 输出目录: {out_dir}")
    print(f"[run_with_log] 命令: {' '.join(cmd)}")
    print(f"[run_with_log] 开始: {started.isoformat(timespec='seconds')}")
    print("-" * 60, flush=True)

    exit_code = 0
    try:
        with open(log_path, "w", encoding="utf-8", newline="") as log_f:
            header = (
                f"# cmd: {' '.join(cmd)}\n"
                f"# cwd: {EXAM_DIR}\n"
                f"# started_at: {started.isoformat(timespec='seconds')}\n"
                f"{'-' * 60}\n"
            )
            log_f.write(header)
            log_f.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=str(EXAM_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
            exit_code = proc.wait()
    except KeyboardInterrupt:
        print("\n[run_with_log] 收到中断，已停止子进程", file=sys.stderr)
        exit_code = 130

    finished = datetime.now()
    elapsed_s = (finished - started).total_seconds()

    meta = {
        "timestamp": timestamp,
        "cmd": cmd,
        "forwarded_args": forwarded,
        "cwd": str(EXAM_DIR),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_s, 2),
        "exit_code": exit_code,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 60)
    print(f"[run_with_log] 结束: {finished.isoformat(timespec='seconds')}  耗时 {elapsed_s:.1f}s  exit={exit_code}")
    print(f"[run_with_log] 日志: {log_path}")
    print(f"[run_with_log] 元信息: {meta_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
