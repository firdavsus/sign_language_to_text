import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from model import .

# ========================================== #
# 5. JSONL DATASET PARSER
# ========================================== #
class SignLanguageDataset(Dataset):
    def __init__(self, jsonl_path, label_map, max_seq_len):
        self.samples = []
        self.max_seq_len = max_seq_len
        self.label_map = label_map
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def _parse_landmarks(self, data, expected_len):
        if not data:  
            return [0.0] * (expected_len * 3)
        features = []
        for lm in data:
            features.extend([lm['x'], lm['y'], lm['z']])
        return features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.label_map.get(sample['text'], 0) 
        
        frames_tensor = []
        for frame in sample['frames']:
            f_face = self._parse_landmarks(frame.get('face'), 478)
            f_pose = self._parse_landmarks(frame.get('pose'), 33)
            f_h1 = self._parse_landmarks(frame.get('hand1'), 21)
            f_h2 = self._parse_landmarks(frame.get('hand2'), 21)
            frames_tensor.append(f_face + f_pose + f_h1 + f_h2)
            
        frames_tensor = torch.tensor(frames_tensor, dtype=torch.float32) 
        
        seq_len = frames_tensor.shape[0]
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        
        if seq_len < self.max_seq_len:
            pad_size = self.max_seq_len - seq_len
            pad_tensor = torch.zeros((pad_size, 1659), dtype=torch.float32)
            frames_tensor = torch.cat([frames_tensor, pad_tensor], dim=0)
            padding_mask[seq_len:] = False 
        else:
            frames_tensor = frames_tensor[:self.max_seq_len]
            
        return {
            "frames": frames_tensor,
            "padding_mask": padding_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ========================================== #
# 6. HUGGING FACE TRAINER SCRIPT
# ========================================== #
def build_label_map(jsonl_paths):
    unique_labels = set()
    for path in jsonl_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        unique_labels.add(json.loads(line)['text'])
        except FileNotFoundError:
            pass
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}

def save_label_map(label_map, output_dir):
    """Saves the vocabulary mapping to a JSON file for inference later."""
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(f"Vocabulary saved to {vocab_path}")

if __name__ == "__main__":
    config = Config()
    
    print("Building label mapping...")
    label_map = build_label_map(["train.jsonl", "test.jsonl"])
    
    if not label_map:
        print("Files not found. Initializing dummy map for testing.")
        label_map = {"Ё": 0, "А": 1, "Р": 2}
        
    config.classes_num = len(label_map)
    print(f"Total classes found: {config.classes_num}")

    # --- NEW: Save the vocabulary before training starts ---
    save_label_map(label_map, "./sign_language_model")

    try:
        train_dataset = SignLanguageDataset("train.jsonl", label_map, config.max_seq_len)
        test_dataset = SignLanguageDataset("test.jsonl", label_map, config.max_seq_len)
    except FileNotFoundError:
        print("Warning: jsonl files not found. The Trainer will error if executed.")
        train_dataset, test_dataset = [], []

    model = SignLanguageBert(config)
    
    # --- UPDATED: Training Arguments with Cosine Decay and Warmup ---
    training_args = TrainingArguments(
        output_dir="./sign_language_model",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",    # ADDED: Cosine annealing schedule
        warmup_ratio=0.1,              # ADDED: 10% of total steps for warmup
        eval_strategy="epoch",  
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False, 
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # trainer.train()