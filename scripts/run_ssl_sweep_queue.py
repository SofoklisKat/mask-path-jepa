#!/usr/bin/env python3
"""Run a static experiment manifest sequentially on one GPU, skipping completed runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--wait-for")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / args.manifest
    jobs = json.loads(manifest_path.read_text())
    status_path = manifest_path.with_suffix(".status.jsonl")

    if args.wait_for:
        prerequisite = root / args.wait_for
        print(f"[{stamp()}] waiting for {prerequisite}", flush=True)
        while not prerequisite.is_file():
            time.sleep(args.poll_seconds)

    for job in jobs:
        config = root / job["config"]
        run_name = job["run_name"]
        output_dir = root / job["output_dir"]
        result = output_dir / run_name / "results.json"
        log = root / job["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        if result.is_file():
            print(f"[{stamp()}] SKIP complete {run_name}", flush=True)
            continue

        command = [
            sys.executable,
            "-u",
            str(root / "train.py"),
            "--config",
            str(config),
            "--run-name",
            run_name,
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
        ]
        checkpoint = output_dir / run_name / "checkpoint_last.pt"
        if checkpoint.is_file():
            command.extend(["--resume", str(checkpoint)])
            print(f"[{stamp()}] RESUME {run_name} from {checkpoint}", flush=True)
        event = {"time": stamp(), "event": "start", "run_name": run_name, "device": args.device}
        with status_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        print(f"[{event['time']}] START {run_name} on {args.device}", flush=True)
        with log.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(line, end="", flush=True)
            code = process.wait()
        event = {"time": stamp(), "event": "finish", "run_name": run_name, "exit_code": code}
        with status_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        if code != 0:
            raise SystemExit(f"{run_name} failed with exit code {code}; see {log}")

    print(f"[{stamp()}] queue complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
