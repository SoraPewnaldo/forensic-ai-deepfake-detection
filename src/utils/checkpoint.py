"""
Strict checkpoint save/load for forensic reproducibility.
Ensures architecture consistency during loading.
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_val: float,
    path: Path,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract keys for schema assertion
    state_dict = model.state_dict()
    architecture_keys = sorted(list(state_dict.keys()))
    
    checkpoint = {
        'epoch': epoch,
        'model_state': state_dict,
        'optimizer_state': optimizer.state_dict(),
        'metric_val': metric_val,
        'architecture_keys': architecture_keys,
        'meta': meta or {}
    }
    
    torch.save(checkpoint, path)

def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location='cpu')
    
    # Schema assertion
    loaded_keys = checkpoint.get('architecture_keys', [])
    current_keys = sorted(list(model.state_dict().keys()))
    
    if loaded_keys and loaded_keys != current_keys:
        missing = set(current_keys) - set(loaded_keys)
        extra = set(loaded_keys) - set(current_keys)
        
        # Buffers are fixed non-learned tensors - safe to ignore cross-stage.
        # Check both current model buffers AND saved checkpoint buffers.
        current_buffer_names = {n for n, _ in model.named_buffers()}
        saved_state = checkpoint['model_state']
        # Any key in 'extra' that is a buffer in the saved checkpoint is safe to skip
        saved_buffer_names = {k for k in saved_state if not saved_state[k].requires_grad 
                              and k not in dict(model.named_parameters())}
        
        ignorable = current_buffer_names | saved_buffer_names
        missing -= ignorable
        extra   -= ignorable
        
        if missing or extra:
            error_msg = f"Architecture mismatch in {path.name}\nMissing: {missing}\nExtra: {extra}"
            if strict:
                raise RuntimeError(error_msg)
            
    model.load_state_dict(checkpoint['model_state'], strict=False)
    
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    return checkpoint
