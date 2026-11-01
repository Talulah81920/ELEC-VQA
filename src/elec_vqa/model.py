import torch
import os
import bitsandbytes as bnb
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .common import resolve

def compute_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def load_processor(cfg):
    options = {"use_fast": cfg["processor_use_fast"]} if "processor_use_fast" in cfg else {}
    processor = AutoProcessor.from_pretrained(
        str(resolve(cfg["model_path"])), local_files_only=True,
        min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"], **options)
    processor.tokenizer.padding_side = "right"
    return processor

def load_model(cfg, training=False, offload=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run scripts/check_env.py first.")
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if training and free < cfg["minimum_training_free_gib"] * 2**30:
        raise RuntimeError(f"Only {free/2**30:.2f} GiB GPU memory is free. "
                           "Close GPU-heavy apps or choose a lower-memory training profile.")
    kwargs = {"local_files_only": True, "torch_dtype": compute_dtype(),
              "attn_implementation": "sdpa", "low_cpu_mem_usage": True}
    quantization = cfg.get("quantization", "nf4")
    if quantization not in ("nf4", "none"):
        raise ValueError("quantization must be nf4 or none")
    if cfg.get("require_bf16", False) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires BF16 hardware support")
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
        llm_int8_enable_fp32_cpu_offload=offload,
        llm_int8_skip_modules=["visual"])
    if offload:
        # Diagnostic forward only; CPU/disk offload is not used for training.
        kwargs.update(device_map="auto", max_memory={
            device: int(max(1.0, free/2**30 - 1.2) * 2**30),
            "cpu": "12GiB"}, offload_folder=str(resolve(".cache/offload")))
    else:
        kwargs["device_map"] = {"": device}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(resolve(cfg["model_path"])), **kwargs)
    model.config.use_cache = False
    if training and quantization == "nf4":
        # Move large unquantized frozen tensors temporarily to CPU: PEFT's
        # fp32 promotion otherwise creates a transient VRAM spike on an 8GB GPU.
        parked = []
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.numel() >= 262144 and parameter.device.type == "cuda":
                parked.append((parameter, parameter.device, parameter.dtype))
                parameter.data = parameter.data.to("cpu")
        torch.cuda.empty_cache()
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        finally:
            for parameter, device, dtype in parked:
                parameter.data = parameter.data.to(device=device, dtype=dtype)
        # PEFT promotes unquantized weights to float32. Retain small layer norms
        # in fp32, but restore frozen vision and large embedding/head tensors to
        # compute dtype so that 7B QLoRA does not waste several GiB on them.
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                continue
            if "visual" in name:
                for parameter in module.parameters(recurse=False):
                    if parameter.is_floating_point():
                        parameter.data = parameter.data.to(compute_dtype())
            elif isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                for parameter in module.parameters(recurse=False):
                    if parameter.is_floating_point():
                        parameter.data = parameter.data.to(compute_dtype())
    if training:
        # Both profiles keep the vision tower frozen and adapt configured text projections.
        targets = [name for name, module in model.named_modules()
                   if "visual" not in name and name.rsplit(".", 1)[-1] in cfg["lora_targets"]
                   and hasattr(module, "weight")]
        if not targets:
            raise RuntimeError("No LoRA targets found")
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    return model
