import hashlib
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .common import ROOT, resolve

class QADataset(Dataset):
    def __init__(self, path):
        self.rows = json.loads(resolve(path).read_text(encoding="utf-8"))
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

def validate_dataset(path, image_root=None):
    image_root = resolve(image_root) if image_root else ROOT / "data"
    rows = QADataset(path).rows
    ids = set()
    images = {}
    for r in rows:
        if r["id"] in ids:
            raise ValueError("Duplicate QA ID: " + r["id"])
        ids.add(r["id"])
        msgs = r["conversations"]
        if len(msgs) != 2 or [m["from"] for m in msgs] != ["human", "gpt"]:
            raise ValueError("Expected independent human/gpt turns")
        if len(r["images"]) != 1 or msgs[0]["value"].count("<image>") != 1:
            raise ValueError("Exactly one image and image placeholder required")
        if not msgs[1]["value"].strip():
            raise ValueError("Empty answer")
        rel = Path(r["images"][0])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe/nonportable image path")
        p = image_root / rel
        if str(p) not in images:
            with Image.open(p) as im:
                im.verify()
            images[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"samples": len(rows), "unique_images": len(images), "image_hashes": images}

class VisionQACollator:
    """Mask the complete user/image prefix; train only the assistant answer."""
    def __init__(self, processor, max_length, image_root=None):
        self.processor = processor
        self.max_length = max_length
        self.image_root = resolve(image_root) if image_root else ROOT / "data"

    def __call__(self, rows):
        prompts, full, images = [], [], []
        for r in rows:
            human, assistant = r["conversations"]
            with Image.open(self.image_root / r["images"][0]) as im:
                images.append(im.convert("RGB"))
            q = human["value"].replace("<image>", "", 1).lstrip("\n")
            msg = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": q}]}]
            prompts.append(self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True))
            full.append(self.processor.apply_chat_template(
                msg + [{"role": "assistant", "content": [
                    {"type": "text", "text": assistant["value"]}]}],
                tokenize=False, add_generation_prompt=False))
        batch = self.processor(text=full, images=images, padding=True, return_tensors="pt")
        prefix = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        # Qwen vision tokens must never be truncated without their image grid.
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError("Encoded sample exceeds max_length; reduce max_pixels or increase max_length.")
        labels = batch["input_ids"].clone()
        for i in range(len(rows)):
            n = int(prefix["attention_mask"][i].sum())
            expected = prefix["input_ids"][i][prefix["attention_mask"][i].bool()]
            actual = batch["input_ids"][i][batch["attention_mask"][i].bool()][:n]
            if not expected.equal(actual):
                raise ValueError("Prompt/full token prefixes differ; refusing incorrect loss mask.")
            # The processor is set to right-padding in load_processor.
            labels[i, :n] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not (labels != -100).any(dim=1).all():
            raise ValueError("No supervised assistant tokens")
        batch["labels"] = labels
        return batch
