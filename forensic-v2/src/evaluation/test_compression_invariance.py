"""
Compression Invariance Test

Ensures prediction remains stable regardless of runtime JPEG quality shifts.
Compares inference on Q=90 vs Q=30 for the same real/fake sequences.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import io

from src.datasets.celebdf_dataset import CelebDFDataset
from src.models import ForensicModel
from src.config import config
from src.utils.checkpoint import load_checkpoint
from torchvision import transforms

def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def test_compression_stability():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = ForensicModel().to(device)
    ckpt_path = Path(config.paths.project_root) / "checkpoints" / "best_Stage2_Hybrid.pt"
    if not ckpt_path.exists():
        print(f"Skipping test: No Stage 2 checkpoint found at {ckpt_path}")
        return
        
    load_checkpoint(model, ckpt_path, strict=True)
    model.eval()
    
    # Load a few random raw sequences from Celeb-DF
    dataset = CelebDFDataset(mode='test')
    if len(dataset) == 0:
        print("Skipping test: Celeb-DF dataset empty.")
        return
    
    # We will manually load raw images to apply distinct Q=90 and Q=30
    base_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Take 1 real, 1 fake folder manually as a pure test
    real_folder = next((f for f, l in dataset.samples if l == 0), None)
    fake_folder = next((f for f, l in dataset.samples if l == 1), None)
    
    test_folders = []
    if real_folder: test_folders.append((real_folder, "REAL"))
    if fake_folder: test_folders.append((fake_folder, "FAKE"))
    
    if not test_folders:
        return
        
    print(f"{'='*50}")
    print(f"COMPRESSION INVARIANCE TEST")
    print(f"{'='*50}")
    
    for folder, label_name in test_folders:
        jpgs = sorted(folder.glob("*.jpg"))
        indices = np.linspace(0, len(jpgs) - 1, 16, dtype=int)
        selected_paths = [jpgs[i] for i in indices]
        
        frames_q90 = []
        frames_q30 = []
        
        for path in selected_paths:
            img = Image.open(path).convert("RGB")
            
            img_q90 = jpeg_compress(img, 90)
            img_q30 = jpeg_compress(img, 30)
            
            frames_q90.append(base_tf(img_q90))
            frames_q30.append(base_tf(img_q30))
            
        tensor_q90 = torch.stack(frames_q90).unsqueeze(0).to(device)
        tensor_q30 = torch.stack(frames_q30).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit_q90 = model(tensor_q90).item()
            logit_q30 = model(tensor_q30).item()
            
        prob_q90 = torch.sigmoid(torch.tensor(logit_q90)).item()
        prob_q30 = torch.sigmoid(torch.tensor(logit_q30)).item()
        
        diff = abs(prob_q90 - prob_q30)
        
        print(f"[{label_name}] Sequence: {folder.name}")
        print(f"  Q=90 Prediction: {prob_q90:.4f}")
        print(f"  Q=30 Prediction: {prob_q30:.4f}")
        print(f"  Absolute Drift : {diff:.4f}")
        if diff < 0.05:
            print(f"  Result         : STABLE (Pass)")
        else:
            print(f"  Result         : UNSTABLE (Fail) - Check augmentations")
        print("-" * 50)

if __name__ == "__main__":
    test_compression_stability()
