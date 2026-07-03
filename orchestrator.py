#!/usr/bin/env python3
"""
orchestrator.py - build and publish `machine-learning-and-deep-learning-mastery`.

What it does
  1. Scaffold: creates any missing day-XX folders and verifies that every deliverable exists
     (day-XX/notes.md, day-XX/model_pipeline.py for 01..30 + README.md, requirements.txt, ...).
  2. Git: `git init` on branch `main` (refuses to touch a repo that already has commits).
  3. Commits: one bootstrap commit, then one commit per day dated 2026-07-<day> at 14:00,
        GIT_AUTHOR_DATE / GIT_COMMITTER_DATE="2026-07-XXT14:00:00"
        git commit -m "feat(day-XX): <Topic>"
  4. Remote: `gh repo create machine-learning-and-deep-learning-mastery --public --source=. --remote=origin`
     (or `--remote-url <url>` for a pre-created repo / plain `git remote add`).
  5. Push everything to `origin main`.

Usage
  python orchestrator.py --dry-run                 # print the plan, change nothing
  python orchestrator.py                           # init + commit + create remote with gh + push
  python orchestrator.py --remote-url git@github.com:<you>/machine-learning-and-deep-learning-mastery.git
  python orchestrator.py --no-push                 # commit locally only
  python orchestrator.py --tz +05:30               # append a UTC offset to the commit timestamps
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_NAME = "machine-learning-and-deep-learning-mastery"
YEAR, MONTH = 2026, 7
ROOT = Path(__file__).resolve().parent

TOPICS = {
    1: "Linear regression: normal equation and gradient descent from scratch",
    2: "Logistic and softmax regression with cross-entropy and Newton/IRLS",
    3: "L1/L2/ElasticNet regularization with coordinate descent",
    4: "Decision trees (CART) and random forests from scratch",
    5: "Gradient boosting from scratch and XGBoost/LightGBM internals",
    6: "SVMs with kernels: hinge loss, Pegasos, RBF and polynomial kernels",
    7: "Hyperparameter tuning: grid, random, successive halving, Bayesian optimization",
    8: "Cross-validation for leaky and time-dependent data (purged, grouped)",
    9: "ROC-AUC vs PR-AUC for imbalanced classification",
    10: "Probability calibration: reliability curves, Platt scaling, isotonic regression",
    11: "Bias-variance decomposition with bootstrap experiments",
    12: "Model explainability: Shapley values, LIME and permutation importance",
    13: "Scikit-Learn custom transformers and leak-safe pipelines",
    14: "Neural network backpropagation derivation with gradient checking",
    15: "Custom PyTorch autograd Functions and nn.Module design",
    16: "Optimizers from scratch: AdamW, Lion and warmup-cosine schedules",
    17: "Activation functions and dead-ReLU analysis",
    18: "Weight initialization: Xavier and He",
    19: "Exploding and vanishing gradients: diagnosis and fixes",
    20: "Capstone: production-grade PyTorch training harness",
    21: "Convolutions from scratch (im2col) and CNNs",
    22: "ResNet skip connections and bottleneck blocks",
    23: "Object detection: IoU, NMS and a YOLO-style grid head",
    24: "LSTM from scratch and sequence-to-sequence modeling",
    25: "Self-attention from scratch",
    26: "Multi-head attention and the Transformer block",
    27: "INT8 quantization and pruning",
    28: "ONNX export and ONNX Runtime benchmarking",
    29: "Edge inference: dynamic batching and TorchScript",
    30: "ML system design: two-tower retrieval and visual anomaly detection",
}
ROOT_FILES = ["README.md", "requirements.txt", ".gitignore", "run_all.py", "orchestrator.py"]
DAY_FILES = ["notes.md", "model_pipeline.py"]


def run(cmd, env=None, check=True, dry=False, capture=False):
    print("  $", " ".join(cmd) if isinstance(cmd, list) else cmd)
    if dry:
        return None
    return subprocess.run(cmd, cwd=ROOT, env=env, check=check, text=True, capture_output=capture)


def scaffold(dry):
    """Step 1: make sure the structure exists and every file is present."""
    print("[1/5] Scaffold / verify")
    missing = []
    for d in range(1, 31):
        folder = ROOT / f"day-{d:02d}"
        if not folder.exists() and not dry:
            folder.mkdir(parents=True)
        missing += [str(folder / f) for f in DAY_FILES if not (folder / f).exists()]
    missing += [f for f in ROOT_FILES if not (ROOT / f).exists()]
    if missing:
        sys.exit("Missing deliverables (restore them before committing):\n  " + "\n  ".join(missing))
    print(f"  OK: {len(ROOT_FILES)} root files + {30 * len(DAY_FILES)} day files present")


def commit_env(day, hour, tz):
    stamp = f"{YEAR}-{MONTH:02d}-{day:02d}T{hour:02d}:00:00{tz}"
    return {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}


def has_commits():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True).returncode == 0


def ensure_identity(dry):
    for key in ("user.name", "user.email"):
        r = subprocess.run(["git", "config", key], cwd=ROOT, capture_output=True, text=True)
        if not r.stdout.strip() and not dry:
            sys.exit(f"git {key} is not set. Run:  git config --global {key} \"<value>\"\n"
                     "(use the e-mail attached to your GitHub account or the commits will not count on your graph)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print commands without executing")
    ap.add_argument("--no-push", action="store_true", help="skip remote creation and push")
    ap.add_argument("--remote-url", help="use this existing remote instead of `gh repo create`")
    ap.add_argument("--private", action="store_true", help="create the GitHub repo as private")
    ap.add_argument("--tz", default="", help="UTC offset appended to commit timestamps, e.g. +05:30 (default: local)")
    ap.add_argument("--hour", type=int, default=14, help="commit hour (default 14)")
    args = ap.parse_args()
    dry = args.dry_run

    scaffold(dry)
    ensure_identity(dry)

    print("[2/5] git init (branch main)")
    if (ROOT / ".git").exists() and has_commits():
        sys.exit("This directory already has commits. Use a fresh clone/folder (or delete .git) to avoid rewriting history.")
    if not (ROOT / ".git").exists():
        r = run(["git", "init", "-b", "main"], dry=dry, check=False)
        if r is not None and r.returncode != 0:                      # git < 2.28
            run(["git", "init"], dry=dry)
            run(["git", "checkout", "-b", "main"], dry=dry)
    else:
        run(["git", "checkout", "-B", "main"], dry=dry)

    print("[3/5] Commits")
    run(["git", "add"] + ROOT_FILES, dry=dry)
    run(["git", "commit", "-m", "chore: bootstrap repository (README, requirements, orchestrator, runner)"],
        env=commit_env(1, 9, args.tz), dry=dry)
    for d in range(1, 31):
        run(["git", "add", f"day-{d:02d}"], dry=dry)
        run(["git", "commit", "-m", f"feat(day-{d:02d}): {TOPICS[d]}"], env=commit_env(d, args.hour, args.tz), dry=dry)

    if args.no_push:
        print("[4/5] Remote + [5/5] Push skipped (--no-push)")
        return
    print("[4/5] Remote")
    if args.remote_url:
        run(["git", "remote", "add", "origin", args.remote_url], dry=dry, check=False)
    elif shutil.which("gh"):
        visibility = "--private" if args.private else "--public"
        r = run(["gh", "repo", "create", REPO_NAME, visibility, "--source=.", "--remote=origin"], dry=dry, check=False)
        if r is not None and r.returncode != 0:
            sys.exit("gh repo create failed (not logged in? name taken?). Run `gh auth login`, or create the repo "
                     "on github.com and re-run with --remote-url <url> in a fresh folder.")
    else:
        sys.exit("GitHub CLI (gh) not found. Install it, or pass --remote-url <url>.")

    print("[5/5] Push")
    run(["git", "push", "-u", "origin", "main"], dry=dry)
    print("\nDone. Commits are dated July 2026 (author/committer date) and pushed to origin/main.")


if __name__ == "__main__":
    main()
