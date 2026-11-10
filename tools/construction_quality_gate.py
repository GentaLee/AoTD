"""Fail closed before training on an incomplete or yield-regressed reconstruction.

This operational gate is not a semantic quality certificate. Fixed-sample trace
review is still required before approving a new construction recipe.
"""
import argparse
import json
from pathlib import Path
from construction_io import records, atomic_json

BASELINE = {"NExT-QA": (34132, 2205), "STAR": (45731, 1814), "AGQA": (25000, 380)}


def audit_source(directory, expected, minimum):
    result = {}; previous = None
    for stage in ["construct", "transfer", "filter"]:
        rows = {}; accepted = 0
        for row in records(directory / (stage + ".jsonl")):
            rows[row["sample_id"]] = row  # latest retry is authoritative
        expected_ids = set(rows) if previous is None else previous
        if previous is None and len(rows) != expected:
            raise ValueError(f"{directory.name}: incomplete construction")
        if set(rows) != expected_ids:
            raise ValueError(f"{directory.name}: {stage} coverage mismatch")
        if any(row.get("status") not in {"ok", "rejected"} for row in rows.values()):
            raise ValueError(f"{directory.name}: unresolved operational errors")
        if stage != "construct" and any(row.get("status") != "ok" for row in rows.values()):
            raise ValueError(f"{directory.name}: text stage rejected records")
        previous = {sid for sid, row in rows.items() if row.get("status") == "ok" and row.get("correct")}
        accepted_ids = {sid for sid, row in rows.items() if row.get("status") == "ok" and row.get("correct") and row.get("accepted")}
        result[stage] = dict(records=len(rows), correct=len(previous), accepted=len(accepted_ids))
    exported = list(records(directory / "cot_filtered.jsonl"))
    if len(exported) != len(accepted_ids) or {r["sample_id"] for r in exported} != accepted_ids:
        raise ValueError(f"{directory.name}: export mismatch or duplicate IDs")
    if len(exported) < minimum:
        raise ValueError(f"{directory.name}: CoT yield {len(exported)} below baseline floor {minimum}; inspect before training")
    result["minimum_accepted"] = minimum
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--construction-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = {"status": "checking", "semantic_quality_certified": False, "sources": {}}
    try:
        for name, (expected, minimum) in BASELINE.items():
            report["sources"][name] = audit_source(a.construction_dir / name, expected, minimum)
        report["status"] = "PASS"
    except (ValueError, OSError, KeyError) as error:
        report.update(status="FAIL", error=str(error))
        atomic_json(a.output, report)
        raise
    atomic_json(a.output, report)
    print("[construction quality gate] complete coverage and baseline yield floors: PASS", flush=True)


if __name__ == "__main__":
    main()
