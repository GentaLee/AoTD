"""Reuse a completed QA-only control only after content, base and recipe checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distillation"))
from aotd_checkpoints import training_identity, validate_completion, atomic_json
from construction_io import records


def semantic_data(yaml_path):
    config = yaml.safe_load(yaml_path.read_text())
    for entry in config["datasets"]:
        path = Path(entry["json_path"])
        if not path.is_absolute(): path = yaml_path.parent / path
        digest = hashlib.sha256(); count = 0
        for row in records(path):
            row = dict(row)
            row["video_path"] = str(Path(row["video_path"]).resolve(strict=True))
            digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
            count += 1
        entry["json_path"] = dict(records=count, sha256=digest.hexdigest())
    return config


def reuse(source, run, root=ROOT):
    source = source.resolve(strict=True); run = run.resolve(strict=True)
    old_root = source.parents[2]
    # Training implementation must be identical; construction-only changes are allowed.
    current_files = {p.relative_to(root/"distillation") for p in (root/"distillation").rglob("*.py")}
    old_files = {p.relative_to(old_root/"distillation") for p in (old_root/"distillation").rglob("*.py")}
    if current_files != old_files or any((root/"distillation"/p).read_bytes() != (old_root/"distillation"/p).read_bytes() for p in current_files):
        raise ValueError("Instruct reuse requires identical training implementation")
    identity = json.loads((source/"training_provenance.json").read_text())
    validate_completion(source, identity)
    if semantic_data(run/"instruct.yaml") != semantic_data(source.parent/"instruct.yaml"):
        raise ValueError("QA content/order or dataset settings changed; do not reuse Instruct")
    # Reconstruct exactly the full entrypoint recipe; only relocated QA paths may differ.
    enum = lambda value: SimpleNamespace(value=value)
    args = SimpleNamespace(world_size=4, per_device_train_batch_size=1, gradient_accumulation_steps=12,
        max_steps=-1, num_train_epochs=1, seed=42, optim=enum("adamw_torch"), learning_rate=4e-5,
        warmup_ratio=.03, lr_scheduler_type=enum("cosine"))
    cli = SimpleNamespace(data_yaml=run/"instruct.yaml", model_path=root/"models/LLaVA-7B-base",
        deepspeed=root/"distillation/configs/ds_config_zero2.json", attn_implementation="sdpa")
    current = training_identity(cli, args, identity.get("training_view_padding"))
    for key in current:
        if key not in {"sources", "data_yaml_sha256"} and current[key] != identity[key]:
            raise ValueError(f"Instruct reuse identity mismatch: {key}")
    destination = run/"instruct_s42"
    if destination.is_symlink():
        if destination.resolve() != source: raise ValueError("Existing reused Instruct points elsewhere")
    elif destination.exists():
        raise ValueError("Refuse to overwrite an existing Instruct output")
    else:
        destination.symlink_to(source, target_is_directory=True)
    atomic_json(run/"instruct_reuse.json", dict(status="PASS", source=str(source),
        trained_in_this_run=False, semantic_qa_identical=True, base_and_recipe_identical=True,
        training_implementation_identical=True, global_step=identity["training_view_padding"]["padded_records"]//48))
    print("[full training] Instruct reused after QA/base/recipe/completion validation; no optimizer steps", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True, type=Path); p.add_argument("--run", required=True, type=Path)
    a = p.parse_args(); reuse(a.source, a.run)
