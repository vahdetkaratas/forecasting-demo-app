from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .smoke import run_demo_smoke
except ImportError:
    from smoke import run_demo_smoke


def _run_tests(base_dir: Path) -> int:
    result = subprocess.run([sys.executable, "-m", "pytest"], cwd=base_dir, check=False)
    return int(result.returncode)


def _run_streamlit(base_dir: Path) -> int:
    app_path = base_dir / "src" / "app.py"
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], cwd=base_dir, check=False)
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecasting app task runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Run pytest suite")
    sub.add_parser("run", help="Run streamlit app")

    smoke_p = sub.add_parser("smoke", help="Run end-to-end smoke checks")
    smoke_p.add_argument("--horizon", type=int, default=14)
    smoke_p.add_argument("--holdout", type=int, default=14)

    gen_p = sub.add_parser("generate", help="Generate demo sample outputs")
    gen_p.add_argument("--horizon", type=int, default=14)
    gen_p.add_argument("--holdout", type=int, default=14)

    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parents[1]

    if args.command == "test":
        return _run_tests(base_dir)

    if args.command == "run":
        return _run_streamlit(base_dir)

    if args.command in {"smoke", "generate"}:
        report = run_demo_smoke(base_dir=base_dir, horizon=args.horizon, holdout=args.holdout)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

