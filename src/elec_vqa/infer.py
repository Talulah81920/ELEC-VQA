import argparse
import torch
from PIL import Image
from peft import PeftModel
from .common import load_config, resolve
from .model import load_model, load_processor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--cpu-offload", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    processor = load_processor(cfg)
    model = load_model(cfg, offload=args.cpu_offload)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(resolve(args.adapter)), is_trainable=False)
    model.eval()
    messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text", "text":args.question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(resolve(args.image)) as im:
        batch = processor(text=[text], images=[im.convert("RGB")], return_tensors="pt")
    batch = {k:v.to("cuda:0") for k,v in batch.items()}
    model.config.use_cache = True
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
    tokens = generated[:, batch["input_ids"].shape[1]:]
    print(processor.batch_decode(tokens, skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
