"""Run fixture generation + all five exporters in dependency order.

Order is fixed: make_fixtures -> acoustic -> vocoder -> istft -> stress
-> ngram. Stops at the first failing step (non-zero exit propagates).
Each step writes its own *_result.txt log next to the models.

Usage:
  python scripts/export_all.py [--out-dir onnx]
"""
import argparse
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS = ["make_fixtures.py", "export_acoustic.py", "export_vocoder.py",
         "export_istft.py", "export_stress.py", "export_ngram.py"]


def main():
    ap = argparse.ArgumentParser(description="export all five ONNX models")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(SCRIPTS_DIR), "onnx"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for step in STEPS:
        print(f"=== {step} ===", flush=True)
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, step),
                            "--out-dir", a.out_dir])
        if r.returncode != 0:
            print(f"export_all: {step} FAILED (exit {r.returncode})",
                  flush=True)
            raise SystemExit(r.returncode)
    print(f"export_all: all 5 models written to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
