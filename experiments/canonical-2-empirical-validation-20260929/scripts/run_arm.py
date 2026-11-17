"""Run one ablation arm: per-frame perception in isolated processes (#688 item 5, #689 workaround).

Each frame is a fresh `run_perception_frame.py` process, which is what makes the run reliable on
the 8 GB device (see issue #689). The arm's only scientific variable is the config layer:
`config/pooling-on.toml` (canonical/2, dense_region_pooling ON) vs `config/pooling-off.toml`
(canonical/1, OFF), both over `config/base.template.toml`.

Usage:
  run_arm.py --config base.toml --config arm.toml --workspace DIR --sequence-id ID
      --out-prefix pool_on [--frames 0 6 12 18] [--stride 6]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 6, 12, 18])
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()

    for index in args.frames:
        argv = [sys.executable, str(HERE / "run_perception_frame.py")]
        for layer in args.config:
            argv += ["--config", layer]
        argv += [
            "--workspace",
            str(args.workspace),
            "--sequence-id",
            args.sequence_id,
            "--frame-index",
            str(index),
            "--stride",
            str(args.stride),
            "--count",
            "1",
            "--out",
            f"{args.out_prefix}_f{index}",
        ]
        print(f"=== {args.out_prefix} frame {index} ===", flush=True)
        result = subprocess.run(argv, check=False)
        if result.returncode != 0:
            print(f"frame {index} failed (exit {result.returncode})", file=sys.stderr)
            return result.returncode
    print(f"{args.out_prefix} ARM DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
