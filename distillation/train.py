import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aotd_checkpoints import (CHECKPOINT_MARKER, COMPLETE_MARKER, atomic_json, choose_checkpoint,
                              export_finished_checkpoint, make_padding_plan, publish_checkpoint, publish_completion, training_identity,
                              validate_completion)

import numpy as np
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_callback import PrinterCallback
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration


import torch
from datasets import load_dataset, concatenate_datasets
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
        if not Path(json_path).is_absolute():json_path = str(Path(data_path).resolve().parent/json_path)
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
    parser.add_argument("--save_strategy", default="steps", choices=("no", "steps", "epoch"))
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", default="auto",
                        help="auto chooses the newest complete compatible checkpoint; none requires a fresh output")
    parser.add_argument('--pad_to_effective_batch',action='store_true',
                        help='Full epoch only: repeat real rows to complete the last effective batch without dropping source QA')
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    cli = parser.parse_args()
    if cli.save_steps < 1 or cli.save_total_limit < 2:
        parser.error('save_steps must be positive and save_total_limit must be at least two')
    if cli.pad_to_effective_batch and cli.max_steps != -1:
        parser.error('pad_to_effective_batch is reserved for complete epoch training, without a max_steps limit')
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
        disable_tqdm = True,
        logging_nan_inf_filter = False,
        save_strategy = cli.save_strategy,
        save_steps = cli.save_steps,
        # save_steps=1000,
        save_total_limit = cli.save_total_limit,
        save_only_model = False,
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


def rank_zero_action(trainer, action):
    """Wait for all checkpoint writers, then publish metadata or fail every rank."""
    trainer.accelerator.wait_for_everyone()
    failure = [None]
    if trainer.is_world_process_zero():
        try:action()
        except Exception as exc:failure[0] = f'{type(exc).__name__}: {exc}'
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(failure, src=0)
    if failure[0] is not None:raise RuntimeError(f'Checkpoint publication failed: {failure[0]}')


class TrainingProgress(TrainerCallback):
    """Expose step loss and include the processor in Trainer checkpoints."""
    def __init__(self, processor, identity=None, resume_checkpoint=None):
        self.processor = processor
        self.losses = []
        self.trainer = None
        self.runtime_backend = None
        self.numeric_logs = []
        self.identity = identity
        self.resume_checkpoint = resume_checkpoint
        interval = float(os.environ.get('AOTD_PROGRESS_INTERVAL_SECONDS', '30'))
        if not math.isfinite(interval):
            raise ValueError('AOTD_PROGRESS_INTERVAL_SECONDS must be finite')
        self.interval = max(10.0, min(60.0, interval))
        self.started = None
        self.last_progress = None
        self.start_step = 0

    def print_progress(self, state, force=False):
        if not state.is_world_process_zero:
            return
        now = time.monotonic()
        if not force and now - self.last_progress < self.interval:
            return
        elapsed = now - self.started
        step, total = state.global_step, max(state.max_steps, 1)
        completed = step - self.start_step
        eta = elapsed / completed * max(total-step, 0) if completed else None
        loss = f'{self.losses[-1]:.6f}' if self.losses else 'pending'
        remaining = f'{eta:.1f}s' if eta is not None else 'pending'
        print(f'[train progress] step={step}/{total} percent={100*step/total:.1f}% '
              f'elapsed={elapsed:.1f}s estimated_remaining={remaining} loss={loss}', flush=True)
        self.last_progress = now

    def on_train_end(self, args, state, control, **kwargs):
        self.print_progress(state, force=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # Callback kwargs expose Trainer.model (the unwrapped HF model). Inspect
        # model_wrapped after accelerator.prepare to audit the actual engine.
        self.started = self.last_progress = time.monotonic()
        self.start_step = state.global_step
        # Trainer loads this history before on_train_begin; preserve earlier loss
        # observations instead of reporting only the resumed suffix as the run.
        for row in state.log_history:
            if 'loss' in row:
                loss = float(row['loss'])
                if not math.isfinite(loss):raise FloatingPointError('Saved training history contains non-finite loss')
                self.losses.append(loss)
                self.numeric_logs.append(dict(step=row.get('step'),metrics={key:float(value) for key,value in row.items()
                    if key!='step' and isinstance(value,(int,float)) and not isinstance(value,bool)}))
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
            self.print_progress(state, force=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        numeric = {key: float(value) for key, value in logs.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)}
        self.numeric_logs.append(dict(step=state.global_step, metrics=numeric))
        if "loss" in logs:
            loss = float(logs["loss"])
            self.losses.append(loss)
            if not math.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {state.global_step}: {loss}")
            if "grad_norm" in numeric and not math.isfinite(numeric["grad_norm"]):
                raise FloatingPointError(f"Non-finite gradient norm at step {state.global_step}")
        self.print_progress(state)

    def on_save(self, args, state, control, **kwargs):
        folder = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        def publish():
            self.processor.save_pretrained(folder)
            publish_checkpoint(folder,self.identity,args.world_size,bool(args.deepspeed),self.runtime_backend)
            print(f'[checkpoint] complete resumable state: {folder}',flush=True)
        rank_zero_action(self.trainer,publish)

    def on_step_end(self, args, state, control, **kwargs):
        # The last partial interval still needs optimizer/RNG state before final
        # inference export, so a failure while exporting can resume safely.
        if args.save_strategy.value == 'steps' and state.global_step >= state.max_steps:
            control.should_save = True
        return control





def main():
    cli, args = parse_args()
    output = Path(cli.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    train_dataset = None
    padding = None
    if cli.pad_to_effective_batch:
        train_dataset = build_dataset(cli.data_yaml,seed=cli.seed)
        batch=args.world_size*args.per_device_train_batch_size*args.gradient_accumulation_steps
        padding=make_padding_plan(len(train_dataset),batch)
        extra=train_dataset.select(padding['selected_indices'])
        ids=[str(value) for value in extra['sample_id']]
        padding=make_padding_plan(len(train_dataset),batch,ids)
        if padding['padding_records']:
            train_dataset=concatenate_datasets([train_dataset,extra])
        if args.process_index==0:
            print('[train view] '+json.dumps(padding,sort_keys=True),flush=True)
    identity = training_identity(cli,args,padding)
    provenance = output/'training_provenance.json'
    if provenance.is_file() and json.loads(provenance.read_text()) != identity:
        raise ValueError('Training inputs/base/protocol/recipe changed; preserve the run and use a fresh output directory')
    if (output/COMPLETE_MARKER).is_file():
        validate_completion(output,identity)
        if args.process_index == 0:print('[train] verified completed phase; skipping duplicate training',flush=True)
        return
    resume = choose_checkpoint(output,identity,cli.resume_from_checkpoint)
    if not provenance.is_file() and any(output.glob('model*.safetensors*')):
        raise ValueError('Inference weights without training provenance are not a resumable checkpoint')
    if args.process_index == 0:
        atomic_json(provenance,identity)
        print(f'[train] resume={resume or "fresh"}; checkpoint every {cli.save_steps} steps; retain {cli.save_total_limit}',flush=True)
    if resume is not None:
        state=json.loads((resume/'trainer_state.json').read_text())
        if state['global_step']>=state['max_steps']:
            failure=[None]
            if args.process_index==0:
                try:
                    export_finished_checkpoint(output,resume,identity)
                    print('[train] recovered final export from completed checkpoint; zero additional optimizer steps',flush=True)
                except Exception as exc:failure[0]=f'{type(exc).__name__}: {exc}'
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(failure,src=0)
            if failure[0]:raise RuntimeError(failure[0])
            return
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

    if train_dataset is None:train_dataset = build_dataset(cli.data_yaml, seed=cli.seed)
    progress = TrainingProgress(processor,identity,resume)


    trainer = Trainer(
        model = model,
        tokenizer = processor,
        data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset = train_dataset,
        args=args,
        callbacks=[progress],
    )
    trainer.remove_callback(PrinterCallback)
    progress.trainer = trainer

    if resume is not None and args.deepspeed:
        # DeepSpeed 0.15 uses torch.load without a weights_only argument. Newer
        # PyTorch defaults reject its optimizer metadata. Scope legacy loading
        # exclusively to our provenance-validated local resume directory.
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
        original_load = TorchCheckpointEngine.load
        trusted = resume.resolve()
        def load_trusted(engine,path,map_location=None):
            candidate = Path(path).resolve()
            if trusted in candidate.parents:
                return torch.load(candidate,map_location=map_location,weights_only=False)
            return original_load(engine,path,map_location=map_location)
        TorchCheckpointEngine.load = load_trusted
        try:result = trainer.train(resume_from_checkpoint=str(resume))
        finally:TorchCheckpointEngine.load = original_load
    else:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    loss = float(result.metrics["train_loss"])
    if not progress.losses or not math.isfinite(loss):
        raise FloatingPointError(f"Training did not report finite losses: train_loss={loss}")
    if trainer.is_world_process_zero():
        print(f"[train] saving full model and processor to {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    def publish_final():
        processor.save_pretrained(args.output_dir)
        metrics = {
            **result.metrics,
            'train_loss': sum(progress.losses)/len(progress.losses),
            'current_invocation_metrics': result.metrics,
            "global_step": trainer.state.global_step,
            "finite_loss": all(math.isfinite(value) for value in progress.losses),
            "step_losses": progress.losses,
            "numeric_logs": progress.numeric_logs,
            "progress_interval_seconds": progress.interval,
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
            "training_scope": "smoke when max_steps is set; selected-source full training otherwise",
            'resume_from_checkpoint': str(resume) if resume is not None else None,
            'resumed_from_global_step': progress.start_step,
            'target_steps': trainer.state.max_steps,
            'periodic_checkpoint_steps': cli.save_steps,
            'training_view_padding': padding,
        }
        with (Path(args.output_dir) / "training_metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
        publish_completion(output,identity,trainer.state.global_step,trainer.state.max_steps)
        print(f"[train] complete: {trainer.state.global_step} steps, finite loss; saved {args.output_dir}", flush=True)
    rank_zero_action(trainer,publish_final)


if __name__ == "__main__":

    main()


