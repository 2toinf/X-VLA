# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from peft import LoraConfig, get_peft_model



import logging
import os
import sys

# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=16)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Split param groups by module type with different learning rates."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups: 
        if g["name"] == name: g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name: return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start: return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Elegant group-wise LR scheduler."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    def schedule(step, base_lr):
        return linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio)
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(
        log_with="tensorboard", 
        project_dir=output_dir
    )
    accelerator.init_trackers("XVLA-Training")
    
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")

    # Load model & processor
    model = XVLA.from_pretrained(args.models)
    
    lora_config = LoraConfig(
        lora_alpha=16,
        r=8,
        bias="none",
        target_modules="all-linear",
        modules_to_save=["transformer.soft_prompt_hub", 
                         "transformer.action_encoder", 
                         "transformer.action_decoder"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    processor = XVLAProcessor.from_pretrained(args.models)

    # Iterable dataloader (don't wrap with prepare)
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=args.seed,
    )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)


    # Training loop
    model.train()
    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start training for {args.iters} iterations | world_size={accelerator.num_processes}")
    
    
    
    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        # Update LR per group
        update_group_lrs(optim, global_step, args)

        # Forward & backward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        # Logging
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it)"
                )
        
        # Checkpointing
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
        
        if global_step >= args.iters: break

    accelerator.end_training()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
