import json
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model import SignLanguageModel, Config # Ensure this matches your model's actual class name

# Device Enforcement
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"!!! FORCING TRAINING ON: {device.upper()} !!!")

class SignLanguageDataset(Dataset):
    def __init__(self, jsonl_path, label_map, max_seq_len, is_train=False):
        self.jsonl_path = jsonl_path
        self.max_seq_len = max_seq_len
        self.label_map = label_map
        self.is_train = is_train
        self.line_offsets = []
        
        with open(jsonl_path, 'rb') as f:
            while True:
                offset = f.tell()
                if not f.readline(): break
                self.line_offsets.append(offset)

    def _parse(self, data, n):
        if not data: return [0.0] * (n * 3)
        return [val for lm in data for val in (lm['x'], lm['y'], lm['z'])]

    def __len__(self): return len(self.line_offsets)

    def __getitem__(self, idx):
        # Using a context manager inside __getitem__ is multiprocessing-safe
        with open(self.jsonl_path, 'rb') as f:
            f.seek(self.line_offsets[idx])
            sample = json.loads(f.readline().decode('utf-8'))
            
        label = self.label_map.get(sample['text'], 0)
        frames = [self._parse(f.get('face'), 478) + self._parse(f.get('pose'), 33) + 
                  self._parse(f.get('hand1'), 21) + self._parse(f.get('hand2'), 21) for f in sample['frames']]
        frames = torch.tensor(frames, dtype=torch.float32)
        
        if len(frames) > 0:
            nose_coords = frames[:, 1434:1437].repeat(1, 553)
            valid_mask = (frames != 0.0) 
            frames = torch.where(valid_mask, frames - nose_coords, frames)
            
            if self.is_train:
                # 1. TIME-WARPING (Speed up or slow down the video by up to 20%)
                # Done before frame dropout to keep interpolation smooth
                if torch.rand(1).item() < 0.5 and len(frames) > 10:
                    target_len = int(len(frames) * torch.empty(1).uniform_(0.8, 1.2).item())
                    target_len = max(1, target_len)
                    
                    # F.interpolate needs shape (Batch, Channels, Time) -> (1, 1659, T)
                    frames = frames.T.unsqueeze(0)
                    # Using 'linear' to smoothly interpolate coordinate values
                    frames = F.interpolate(frames, size=target_len, mode='linear', align_corners=False)
                    frames = frames.squeeze(0).T

                # 2. SPATIAL SCALING (Simulate larger/smaller people)
                if torch.rand(1).item() < 0.5:
                    scale_factor = torch.empty(1).uniform_(0.8, 1.2).item()
                    frames = frames * scale_factor

                # 3. SPATIAL JITTER (Simulate MediaPipe inaccuracies)
                if torch.rand(1).item() < 0.5:
                    noise = torch.randn_like(frames) * 0.005
                    frames = torch.where(valid_mask, frames + noise, frames)

                # 4. FRAME DROPOUT (Simulate dropped frames / low FPS)
                if torch.rand(1).item() < 0.5 and len(frames) > 5:
                    keep_mask = torch.rand(len(frames)) > 0.10
                    if keep_mask.sum() > 0:
                        frames = frames[keep_mask]

        seq_len = min(len(frames), self.max_seq_len)
        padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        padding_mask[:seq_len] = True
        
        padded_frames = torch.zeros((self.max_seq_len, 1659))
        padded_frames[:seq_len] = frames[:seq_len]
        
        return {"frames": padded_frames, "padding_mask": padding_mask, "labels": torch.tensor(label)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (np.argmax(logits, -1) == labels).mean()}


if __name__ == "__main__":
    cfg = Config()
    save_path = "../model_save"
    vocab_file = os.path.join(save_path, "vocab.json")
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(vocab_file):
        print(f"Loading existing vocab from {vocab_file}")
        with open(vocab_file, 'r') as f: label_map = json.load(f)
    else:
        print("Re-scanning datasets for new vocabulary...")
        with open("../train.jsonl", 'r') as f:
            labels = sorted(list(set(json.loads(l)['text'] for l in f)))
        label_map = {l: i for i, l in enumerate(labels)}
        with open(vocab_file, 'w') as f: json.dump(label_map, f)

    cfg.classes_num = len(label_map)
    train_ds = SignLanguageDataset("../train.jsonl", label_map, cfg.max_seq_len, is_train=True)
    test_ds = SignLanguageDataset("../test.jsonl", label_map, cfg.max_seq_len, is_train=False)

    # Initialize model (Make sure the class name matches what you called it in model.py!)
    model = SignLanguageModel(cfg).to(device) 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    if device == "cuda" and int(torch.__version__.split('.')[0]) >= 2:
        print("Compiling model for faster training...")
        model = torch.compile(model)

    # Calculate optimal workers (leave 1 core free)
    optimal_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=300, 
        save_strategy="steps",
        save_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        
        # 🚀 INCREASED WORKERS: Reading JSONL from disk is slow. This prevents the GPU from waiting.
        dataloader_num_workers=optimal_workers, 
        
        remove_unused_columns=False,
        optim="adamw_torch_fused", 
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()