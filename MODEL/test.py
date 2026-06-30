import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file  # <-- Added this import

# Import your classes from model.py and train.py
from model import SignLanguageBert, Config
from train import SignLanguageDataset

def run_evaluation(checkpoint_path, vocab_path, test_jsonl):
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device.upper()}")

    # 2. Load Vocabulary (Mapping ID -> Text)
    with open(vocab_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    
    # Create the inverse map for human-readable output
    id_to_label = {int(v): k for k, v in label_map.items()}
    
    # 3. Initialize Config and Model
    cfg = Config()
    cfg.classes_num = len(label_map)
    
    model = SignLanguageBert(cfg)
    
    # 4. Load Weights Safely
    sf_path = os.path.join(checkpoint_path, "model.safetensors")
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    if os.path.exists(sf_path):
        print(f"Loading weights from safetensors: {sf_path}")
        state_dict = load_file(sf_path, device=device)
    elif os.path.exists(bin_path):
        print(f"Safetensors not found. Loading from bin: {bin_path}")
        # Added weights_only=True to prevent PyTorch 2.6 security crash
        state_dict = torch.load(bin_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(f"Could not find model weights in {checkpoint_path}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Load Test Dataset
    test_dataset = SignLanguageDataset(test_jsonl, label_map, cfg.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    
    print(f"Starting inference on {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move batch to device
            frames = batch["frames"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(frames, padding_mask=padding_mask)
            logits = outputs["logits"]
            
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Calculate Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    
    print("\n" + "="*30)
    print(f"EVALUATION RESULTS")
    print("="*30)
    print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")
    
    # Show some example predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(all_preds))):
        actual = id_to_label[all_labels[i]]
        pred = id_to_label[all_preds[i]]
        status = "✅" if actual == pred else "❌"
        print(f"{status} Actual: {actual} | Predicted: {pred}")

if __name__ == "__main__":
    # Update these paths to your actual folders
    CHECKPOINT = "../model_save/checkpoint-2400" # Path to your best checkpoint folder
    VOCAB = "../model_save/vocab.json"
    TEST_DATA = "../test.jsonl"
    
    run_evaluation(CHECKPOINT, VOCAB, TEST_DATA)