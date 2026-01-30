import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, IterableDataset, interleave_datasets
import glob
import argparse
import random
import time

from cmtp_model import LLMWithCMTP

def rank0_print(*args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

class MTPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.loss_meters = {} 
        self.main_loss_meter = 0.0
        self.loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        
        current_loss_val = loss.item()
        self.main_loss_meter += current_loss_val

        if outputs.offset_losses is not None:
            for key, val in outputs.offset_losses.items():
                if key not in self.loss_meters:
                    self.loss_meters[key] = 0.0
                self.loss_meters[key] += val.item()
            
        self.loss_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self.loss_steps > 0:
            for key, total_val in self.loss_meters.items():
                logs[key] = round(total_val / self.loss_steps, 4)
            
            avg_main_loss = self.main_loss_meter / self.loss_steps
            logs["loss"] = round(avg_main_loss, 4)

            self.loss_meters = {}
            self.main_loss_meter = 0.0
            self.loss_steps = 0
        
        if "grad_norm" in logs:
            logs["grad_norm"] = round(logs["grad_norm"], 4)
        
        if "learning_rate" in logs:
            logs["learning_rate"] = float(f"{logs['learning_rate']:.2e}")

        super().log(logs, *args, **kwargs)

def prepare_mixed_streaming_dataset(dataset_configs, tokenizer, block_size=1024):
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    dataset_objs = []
    dataset_probs = []

    rank0_print(f"=== Preparing mixed dataset (Rank {rank}) ===")

    for config in dataset_configs:
        name = config["name"]
        data_path = os.path.expanduser(config["path"])
        text_col = config.get("column", "text")
        prob = config.get("prob", 1.0)
        max_files = config.get("num_files", None)

        if os.path.isdir(data_path):
            all_files = sorted(glob.glob(f"{data_path}/*.parquet"))
        else:
            all_files = sorted(glob.glob(data_path))

        if not all_files:
            print(f"[Warning] Dataset {name} found no files at {data_path}, skipping.")
            continue
            
        if max_files is not None:
            all_files = all_files[:max_files]

        if len(all_files) < world_size:
            my_files = all_files
            need_row_sharding = True
            rank0_print(f"  - [{name}] Number of files is less than number of GPUs, will shard by rows after loading.")
        else:
            my_files = all_files[rank::world_size]
            need_row_sharding = False
            random.shuffle(my_files)
            print(f"  - [{name} Rank {rank}] Assigned {len(my_files)}/{len(all_files)} files.")

        ds = load_dataset(
            "parquet", 
            data_files=my_files, 
            split="train", 
            streaming=True
        )

        if text_col != "text":
            if text_col in ds.features:
                ds = ds.rename_column(text_col, "text")
            else:
                pass 
        
        ds = ds.select_columns(["text"])

        if need_row_sharding and world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)

        dataset_objs.append(ds)
        dataset_probs.append(prob)

    if not dataset_objs:
        raise ValueError("No valid datasets loaded! Please check the paths.")

    mixed_dataset = interleave_datasets(
        dataset_objs, 
        probabilities=dataset_probs, 
        seed=42, 
        stopping_strategy="first_exhausted"
    )

    def constant_length_iterator(dataset, tokenizer, seq_len):
        buffer = []
        for example in dataset:
            text = example["text"]
            if not text: continue
            
            encoded = tokenizer(text)
            input_ids = encoded["input_ids"]
            input_ids.append(tokenizer.eos_token_id) 

            buffer.extend(input_ids)
            
            while len(buffer) >= seq_len:
                yield {
                    "input_ids": buffer[:seq_len],
                    "labels": buffer[:seq_len]
                }
                buffer = buffer[seq_len:]
                
    streaming_dataset = IterableDataset.from_generator(
        constant_length_iterator, 
        gen_kwargs={
            "dataset": mixed_dataset, 
            "tokenizer": tokenizer, 
            "seq_len": block_size
        }
    )
    
    current_seed = int(time.time()) + rank
    rank0_print(f"Global Shuffle Seed: {current_seed}")
    shuffled_dataset = streaming_dataset.shuffle(buffer_size=10000, seed=current_seed)
    
    return shuffled_dataset

import deepspeed

def main():
    deepspeed.init_distributed()
    parser = argparse.ArgumentParser(description="Train MTP for LLM")
    parser.add_argument("--model_name", type=str, default="Qwen3-8B", help="Base model name or path")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--max_steps", type=int, default=3000, help="Max training steps")
    parser.add_argument("--mtp_ckpt_path", type=str, default=None, help="Path to MTP projector checkpoint")
    parser.add_argument("--kd_alpha", type=float, default=0.7, help="Knowledge Distillation loss weight")
    parser.add_argument("--future_offsets", type=int, nargs="+", default=[2, 3, 4], help="List of future offset steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument('--data_path', type=str, default=None, help='Path to training data, some .parquet files under this directory')

    parser.add_argument('--local_rank', type=int, default=-1, help='Parameter automatically passed by DeepSpeed')
    args = parser.parse_args()

    model_name = args.model_name
    per_device_train_batch_size = args.per_device_train_batch_size
    max_steps = args.max_steps
    mtp_ckpt_path = args.mtp_ckpt_path
    kd_alpha = args.kd_alpha
    future_offsets = args.future_offsets
    resume_from_checkpoint = args.resume_from_checkpoint

    rank0_print(f"Config: model_name={model_name}, future_offsets={future_offsets}")

    ds_config_path = "./ds_config.json"
    dataset_configs = [
        {
            "name": "FineWeb-10BT",
            "path": args.data_path,
            "column": "text",
            "prob": 1.0,
            "num_files": None
        }
    ]

    model_path = model_name
    output_dir = "./mtp_output_" + model_name
    warmup_steps = max_steps // 20
    gradient_accumulation_steps = 16 // per_device_train_batch_size

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    rank0_print("Preparing mixed streaming dataset...")
    train_dataset = prepare_mixed_streaming_dataset(
        dataset_configs, 
        tokenizer, 
        block_size=2048
    )
    
    model = LLMWithCMTP(
        model_path, 
        future_offsets=future_offsets,
        kd_alpha=kd_alpha,
        kd_temperature=2.0,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    if mtp_ckpt_path is not None:
        rank0_print(f"Loading MTP projector from {mtp_ckpt_path}")
        model.load_mtp_projector(mtp_ckpt_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,  
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=1000,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        logging_dir=os.path.join(output_dir, "runs"),
        report_to=["tensorboard"],
        deepspeed=ds_config_path
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = MTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    rank0_print("Starting streaming training for MTP head...")
    if resume_from_checkpoint is not None:
        rank0_print(f"Checkpoint detected, resuming training from {resume_from_checkpoint}...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        rank0_print("No checkpoint found, starting fresh training...")
        trainer.train()
    
    rank0_print("Processing model saving...")
    trainer.accelerator.wait_for_everyone()
    unwrapped_model = trainer.accelerator.unwrap_model(model)

    if trainer.accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, model_name + ".bin")
        torch.save(unwrapped_model.state_dict(), save_path)
        rank0_print(f"MTP weights saved to: {save_path}")
        unwrapped_model.base_model.config.save_pretrained(output_dir)

if __name__ == "__main__":
    main()