import argparse
import json
import time
import hashlib
import platform
import torch
from transformers import Trainer, TrainingArguments, set_seed
from .common import ROOT, load_config, resolve
from .data import QADataset, VisionQACollator, validate_dataset
from .model import load_model, load_processor

def build_trainer(cfg, model, processor):
    train_data = QADataset(cfg["train_data"])
    eval_data = QADataset(cfg["eval_data"]) if cfg.get("eval_data") else None
    # Keep encoded images/labels from our collator; no prompt or image loss.
    training_args = TrainingArguments(
        output_dir=str(resolve(cfg["output_dir"])),
        max_steps=cfg.get("max_steps", -1),
        num_train_epochs=cfg["epochs"], per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 1), gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], warmup_ratio=0.05,
        lr_scheduler_type="cosine", weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.get("optim", "adamw_torch"), dataloader_num_workers=cfg.get("num_workers", 0),
        dataloader_pin_memory=cfg.get("pin_memory", False),
        prediction_loss_only=True,
        remove_unused_columns=False, label_names=["labels"], report_to="none", logging_steps=cfg.get("logging_steps",1),
        save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch" if eval_data else "no", seed=cfg["seed"])
    return Trainer(model=model, args=training_args,
        train_dataset=train_data, eval_dataset=eval_data,
        data_collator=VisionQACollator(processor, cfg["max_length"], image_root=cfg.get("image_root")), processing_class=processor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/answer_only.yaml")
    parser.add_argument("--check-data", action="store_true", help="Validate only; no model or training")
    parser.add_argument("--resume", default=None, help="Resume a Trainer checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Short training smoke run; omit for full training")
    parser.add_argument("--output-dir", default=None, help="Override output directory, e.g. for smoke runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        cfg['max_steps'] = args.max_steps
    report = validate_dataset(cfg["train_data"], image_root=cfg.get("image_root"))
    print(json.dumps({"samples": report["samples"], "images": report["unique_images"]}))
    if args.check_data:
        return
    set_seed(cfg["seed"])
    destination = resolve(cfg['output_dir'])
    destination.mkdir(parents=True, exist_ok=True)
    def digest(path):
        return hashlib.sha256(resolve(path).read_bytes()).hexdigest()
    provenance = {'config': cfg, 'train_sha256': digest(cfg['train_data']),
                  'eval_sha256': digest(cfg['eval_data']) if cfg.get('eval_data') else None,
                  'python': platform.python_version(), 'torch': torch.__version__,
                  'resume_checkpoint': args.resume, 'status': 'started'}
    (destination/'run_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    start = time.perf_counter()
    processor = load_processor(cfg)
    model = load_model(cfg, training=True)
    trainer = build_trainer(cfg, model, processor)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=str(resolve(args.resume)) if args.resume else None)
    final = resolve(cfg["output_dir"]) / "final_adapter"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    trainer.save_state()
    provenance.update(status='complete', global_step=trainer.state.global_step,
        wall_seconds_including_model_load=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        gpu=torch.cuda.get_device_name(), trainer_metrics=result.metrics)
    (destination/'run_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == "__main__":
    main()
