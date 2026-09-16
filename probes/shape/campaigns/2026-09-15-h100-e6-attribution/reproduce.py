#!/usr/bin/env python3
"""Reanalyse C1 on CPU, without changing its raw evidence or extracting tar members."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = "Qwen_Qwen2.5-7B-Instruct_tp1_none_compile_v2_graphs1_prefix0"
RECORDS = {
    REF: "A1_record",
    REF + "_autotune_off": "C1_autotune_off_record",
    REF + "_rmsnorm_custom": "E1_rmsnorm_record",
    REF + "_combo_off": "G1_combo_off_record",
    "Qwen_Qwen2.5-7B-Instruct_tp1_fp8_compile_v2_graphs1_prefix0": "B1_fp8_record",
    "Qwen_Qwen2.5-7B-Instruct_tp1_none_nocompile_v2_graphs1_prefix0_nocompile": "F1_nocompile_record",
}


def choices(label):
    paths = list(ROOT.glob(f"artefacts_m*/{label}.light.tgz"))
    assert paths, label
    result = {}
    with tarfile.open(paths[0]) as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".best_config"):
                cfg = json.load(archive.extractfile(member))
                cfg.pop("time_taken_ms", None)
                key = Path(member.name).name.removesuffix(".best_config")
                assert key not in result
                result[key] = cfg
    return result


def rows(path):
    result = []
    for line in (path / "hook/rank0.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "forward":
            continue
        for row in record["requests"]:
            if not str(row.get("req", "")).startswith("_warmup"):
                result.append([row.get("h"), row.get("logits_h")])
    return result


def main():
    manifest = json.loads((ROOT / "SHA256.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    comparisons = []
    for log in sorted(ROOT.glob("logs_m*/queue_*.log")):
        for line in log.read_text().splitlines():
            match = re.match(r"QUEUE ARM_START (\S+) \S+ :: kind=(\S+) cache=(\S+) env='[^']*' caches_before=\S+ :: (.*)$", line)
            if match and match[2] in ("replay", "boundary"):
                arm, output, *_ = match[4].split()
                comparisons.append((match[1], match[2], match[3], arm, output))
    assert len(comparisons) == 32
    agreement = Counter()
    classes = defaultdict(list)
    record_rows = rows(ROOT / "results_pod/e6" / REF / "record")
    reference_runs = [("A1_record", "record")]
    with tempfile.TemporaryDirectory(prefix="c1-reanalysis-") as temporary:
        target = Path(temporary) / "e6"
        shutil.copytree(ROOT / "results_pod/e6", target)
        for pattern in ("summary*.json", "boundary_*.json"):
            for path in target.rglob(pattern):
                path.unlink()
        for label, kind, cache, arm, output in comparisons:
            run = target / arm
            subprocess.run([sys.executable, str(ROOT / "src/probes/shape/run_e6.py"),
                            "--compare", str(run / "record"), str(run / output)],
                           check=True, capture_output=True, text=True)
            name = f"{output}.json" if kind == "boundary" else f"summary_{output}.json"
            saved = ROOT / "results_pod/e6" / arm / name
            assert (run / name).read_bytes() == saved.read_bytes(), name
            summary = json.loads(saved.read_text())
            equal = choices(label) == choices(RECORDS[arm])
            differs = summary["rows_differing"] > 0
            assert equal != differs, label
            agreement[f"{'equal' if equal else 'unequal'}_{'DIFFERS' if differs else 'IDENTICAL'}"] += 1
            if arm == REF and kind == "replay":
                reference_runs.append((label, output))
    for label, output in reference_runs:
        cfg = choices(label)
        signature = tuple(next(value["R0_BLOCK"] for key, value in cfg.items() if key.startswith(prefix))
                          for prefix in ("d899c7b6", "4e10c47a", "e0f56c5c"))
        content = rows(ROOT / "results_pod/e6" / REF / output)
        assert len(content) == len(record_rows) == 512
        digest = hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()
        classes[signature].append((label, digest, sum(a != b for a, b in zip(record_rows, content))))
    for runs in classes.values():
        assert len({row[1] for row in runs}) == 1
    assert len(classes) == len({runs[0][1] for runs in classes.values()}) == 4
    print(json.dumps({"files_verified": len(manifest), "summaries_byte_identical": len(comparisons),
                      "agreement": dict(agreement), "reference_classes": [
                          {"R0_BLOCK": signature, "runs": [row[0] for row in runs],
                           "row_sha256": runs[0][1], "rows_differing": runs[0][2]}
                          for signature, runs in sorted(classes.items())]}, indent=2))


if __name__ == "__main__":
    main()
