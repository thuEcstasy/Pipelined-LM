"""Portable folded PLT-4 acceptance check. Run from the repository root."""

import argparse
import json
import math
import platform
import subprocess
import sys
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch

from tests.folded_backend import check_backend, configure_cuda
from tests.test_folded_plt import FoldedPLTTest
from tests.test_folded_visibility import check_visibility
from tests.test_pipeline import PipelineTest


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def json_values(value):
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True, choices=("cpu", "cuda"))
    p.add_argument("--report", required=True, type=Path, help="new JSON report path")
    a = p.parse_args()
    report = {
        "status": "running",
        "requested_device": a.device,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "cuda_build": torch.version.cuda,
        "utc": datetime.now(timezone.utc).isoformat(),
        "model_seed": 20261017,
        "oracle_rtol": 2e-4,
        "oracle_atol": 3e-5,
        "checks": {},
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open("x") as f:
        json.dump(report, f)

    def save():
        temp = a.report.with_suffix(a.report.suffix + ".tmp")
        temp.write_text(
            json.dumps(json_values(report), indent=2, allow_nan=False) + "\n"
        )
        temp.replace(a.report)

    def check(name, fn, *args):
        record = report["checks"][name] = {"status": "running", "metrics": {}}
        save()
        print(name, flush=True)
        try:
            fn(*args, record["metrics"])
        except Exception:
            record.update(status="failed", traceback=traceback.format_exc())
            raise
        else:
            record["status"] = "passed"
        finally:
            save()

    def unit(case, metrics):
        result = unittest.TestResult()
        case.run(result)
        metrics.update(getattr(case, "metrics", {}))
        if not result.wasSuccessful() or result.skipped:
            raise AssertionError(
                str(result.skipped)
                if result.skipped
                else "\n".join(msg for _, msg in result.errors + result.failures)
            )

    try:
        torch.set_num_threads(min(2, torch.get_num_threads()))
        if a.device == "cuda":
            configure_cuda()
            report["gpu"] = torch.cuda.get_device_name()
            report["compute_capability"] = torch.cuda.get_device_capability()
        for name in unittest.defaultTestLoader.getTestCaseNames(FoldedPLTTest):
            check("cpu/" + name, unit, FoldedPLTTest(name))
        check(
            "cpu/untied_cache_policies",
            unit,
            PipelineTest("test_cache_policies"),
        )
        for device in dict.fromkeys(("cpu", a.device)):
            check(
                device + "/visible_source_weights",
                check_visibility,
                device,
            )
        for enabled in (False, True):
            label = "on" if enabled else "off"
            check(
                f"{a.device}/trained_parity/{label}",
                check_backend,
                a.device,
                enabled,
            )
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    else:
        report["status"] = "passed"
    finally:
        save()
        print(f"{report['status'].upper()} ({a.device}) — {a.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
