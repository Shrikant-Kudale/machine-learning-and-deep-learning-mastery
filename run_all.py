"""Run every day's model_pipeline.py and report pass/fail + runtime.  Usage: python run_all.py [--days 1 5 20]"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="*", default=list(range(1, 31)))
    args = ap.parse_args()
    failed = []
    for d in args.days:
        script = ROOT / f"day-{d:02d}" / "model_pipeline.py"
        t0 = time.time()
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"day-{d:02d}  {'PASS' if ok else 'FAIL'}  {time.time() - t0:6.1f}s")
        if not ok:
            failed.append(d)
            print(r.stdout[-800:], r.stderr[-1500:], sep="\n")
    print(f"\n{len(args.days) - len(failed)}/{len(args.days)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
