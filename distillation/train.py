import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration


import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import *
from decord import VideoReader, gpu, cpu
import yaml



MAX_LENGTH = 8192
BATCH_SIZE = 4
OUTPUT_DIR = "./outputs" # path where to save the checkpoints
MODEL_ID = "your_path/LLaVA-NeXT-Video-7B-hf"
REPO_ID = "AoTD_train" # Change to your hf-hub repo
DATA_PATH = "datasets/train_datasets.yaml"


def get_video(video_path):
    vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) # you need to install from source to use gpu ctx
    vlen = len(vr)
    indices = np.linspace(0, vlen, 32, endpoint=False).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batches = []
        for feature in features:
            conversation = feature['conversation']
            conversation = self.processor.apply_chat_template(conversation, add_generation_prompt=False)

            video = get_video(feature['video_path'])
            # import ipdb; ipdb.set_trace()
            batch = self.processor(
                text=conversation,
                videos=video,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            batch['input_ids'][0][-1] = self.processor.tokenizer.eos_token_id
            batches.append(batch)
        padded_inputs = self.processor.tokenizer.pad(
            {   
                "input_ids": [batch['input_ids'][0] for batch in batches],
                "attention_mask": [batch['attention_mask'][0] for batch in batches],
            },
            padding=True,
            return_tensors="pt",
        )
        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat([batch['pixel_values_videos'] for batch in batches], dim=0)

        return padded_inputs

def build_dataset(data_path, seed=42):
    dataset_combined = []

    with open(data_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        datasets = yaml_data.get("datasets")

    for dataset in datasets:
        json_path = dataset.get("json_path")
        dataset = load_dataset("json", data_files=json_path)
        dataset_combined.append(dataset['train'])
    
    dataset_processed = concatenate_datasets(dataset_combined)
    dataset_processed = dataset_processed.shuffle(seed=seed)

    train_dataset = dataset_processed.with_format("torch")

    return train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="AoTD student distillation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_yaml", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--per_device_train_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--save_strategy", default="epoch", choices=("no", "steps", "epoch"))
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    cli = parser.parse_args()
    args = TrainingArguments(

        # args related to training
        output_dir = cli.output_dir,
        # dataloader_drop_last=True,
        do_eval=False,
        eval_strategy = 'no',
        per_device_train_batch_size = cli.per_device_train_batch_size,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = cli.gradient_accumulation_steps,
        max_steps = cli.max_steps,
        seed = cli.seed,
        deepspeed = cli.deepspeed,
        local_rank = cli.local_rank,
        learning_rate = 4e-05,
        num_train_epochs=1, # adjust this depending on your dataset size
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.03,
        remove_unused_columns=False,

        # args related to eval/save
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = cli.save_strategy,
        save_steps = cli.save_steps,
        # save_steps=1000,
        save_total_limit = 1,
        fp16 = False, # we have the model train and eval with fp16 precision
        bf16=True,
        tf32=True,
        fp16_full_eval = False,
        optim = cli.optim,
        report_to = "none", # install wand to use this
        hub_model_id = REPO_ID,
        push_to_hub = False, # wel'll push the model to hub after each epoch
        label_names=["labels"],
        dataloader_num_workers=cli.dataloader_num_workers,
        gradient_checkpointing=True,
        neftune_noise_alpha=0.1
    )

    return cli, args


class TrainingProgress(TrainerCallback):
    """Expose step loss and include the processor in Trainer checkpoints."""
    def __init__(self, processor):
        self.processor = processor
        self.losses = []
        self.trainer = None
        self.runtime_backend = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Callback kwargs expose Trainer.model (the unwrapped HF model). Inspect
        # model_wrapped after accelerator.prepare to audit the actual engine.
        trainer = self.trainer
        engine = trainer.model_wrapped
        self.runtime_backend = {
            "distributed_type": str(trainer.accelerator.distributed_type),
            "model_class": f"{type(engine).__module__}.{type(engine).__qualname__}",
            "world_size": args.world_size,
        }
        if args.deepspeed:
            from deepspeed import DeepSpeedEngine
            if not isinstance(engine, DeepSpeedEngine):
                raise RuntimeError(f"DeepSpeed requested but actual model is {type(engine)!r}")
            self.runtime_backend.update({
                "zero_stage": int(engine.zero_optimization_stage()),
                "bfloat16_enabled": bool(engine.bfloat16_enabled()),
                "world_size": int(engine.world_size),
                "optimizer_class": f"{type(engine.optimizer).__module__}.{type(engine.optimizer).__qualname__}",
            })
        if state.is_world_process_zero:
            print("[train backend] " + json.dumps(self.runtime_backend, sort_keys=True), flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = float(logs["loss"])
            self.losses.append(loss)
            if state.is_world_process_zero:
                print(f"[train] step {state.global_step}/{state.max_steps} loss={loss:.6f}", flush=True)
            if not math.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {state.global_step}: {loss}")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.processor.save_pretrained(Path(args.output_dir) / f"checkpoint-{state.global_step}")





def main():
    cli, args = parse_args()
    processor = AutoProcessor.from_pretrained(cli.model_path, use_fast=False, local_files_only=True)
    processor.tokenizer.padding_side = "right"

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        cli.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=cli.attn_implementation,
        local_files_only=True,
    )
    model.config.use_cache = False
    # Transformers 4.49 detects **lm_kwargs on LlavaNextVideo.forward and
    # assumes num_items_in_batch normalizes its loss. This model instead uses
    # its own mean CrossEntropyLoss and ignores that token-count argument.
    # Let Trainer divide each microbatch loss by gradient_accumulation_steps;
    # DeepSpeed is already called with scale_wrt_gas=False by Trainer.
    model.accepts_loss_kwargs = False
    for k, v in model.named_parameters():
        if "vision_tower" in k or 'image_newline' in k:
            v.requires_grad = False

    train_dataset = build_dataset(cli.data_yaml, seed=cli.seed)
    progress = TrainingProgress(processor)


    trainer = Trainer(
        model = model,
        tokenizer = processor,
        data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset = train_dataset,
        args=args,
        callbacks=[progress],
    )
    progress.trainer = trainer

    result = trainer.train()
    loss = float(result.metrics["train_loss"])
    if not progress.losses or not math.isfinite(loss):
        raise FloatingPointError(f"Training did not report finite losses: train_loss={loss}")
    if trainer.is_world_process_zero():
        print(f"[train] saving full model and processor to {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)
        metrics = {
            **result.metrics,
            "global_step": trainer.state.global_step,
            "finite_loss": all(math.isfinite(value) for value in progress.losses),
            "step_losses": progress.losses,
            "runtime_backend": progress.runtime_backend,
            "model_path": cli.model_path,
            "data_yaml": cli.data_yaml,
            "data_yaml_sha256": hashlib.sha256(Path(cli.data_yaml).read_bytes()).hexdigest(),
            "attn_implementation": cli.attn_implementation,
            "optimizer": cli.optim,
            "deepspeed": cli.deepspeed,
            "world_size": args.world_size,
            "effective_batch_size": (args.per_device_train_batch_size *
                                     args.gradient_accumulation_steps * args.world_size),
            "loss_normalization": "trainer_per_microbatch_mean",
            "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
            "training_scope": "smoke when max_steps is set; six-source author data is not bundled",
        }
        with (Path(args.output_dir) / "training_metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
        print(f"[train] complete: {trainer.state.global_step} steps, finite loss; saved {args.output_dir}", flush=True)


if __name__ == "__main__":

    main()


