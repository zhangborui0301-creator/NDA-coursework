from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "part1_small_task_a.py",
    "part1_small_task_b.py",
    "part1_small_task_c.py",
    "part1_medium_task_a.py",
    "part1_medium_task_b.py",
    "part1_medium_task_c.py",
    "part1_large_task_a.py",
    "part1_large_task_b.py",
    "part1_large_task_c.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for script_name in SCRIPTS:
        print(f"\nRunning {script_name} ...")
        completed = subprocess.run([sys.executable, str(script_dir / script_name)], check=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    print("\nAll Part 1 scripts finished successfully.")


if __name__ == "__main__":
    main()
