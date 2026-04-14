"""
GPU info and VRAM monitoring utilities for Windows environments.
"""
import torch
import logging

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def log_vram_usage(logger: logging.Logger, tag: str = "Status"):
    if not torch.cuda.is_available():
        return
        
    # Get current device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_mb = props.total_memory / (1024**2)
    
    allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    
    logger.info(
        f"[VRAM] {tag} | Total: {total_mb:.0f}MB | "
        f"Allocated: {allocated_mb:.0f}MB | "
        f"Reserved: {reserved_mb:.0f}MB"
    )

def assert_vram_safe(min_free_mb: int = 512):
    """Raise error if available VRAM is below safe threshold."""
    if not torch.cuda.is_available():
        return
        
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory
    
    # Approximation of free memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    
    # We use a conservative estimate: total - reserved (or total - allocated)
    free_estimate = total_vram - reserved
    free_mb = free_estimate / (1024**2)
    
    if free_mb < min_free_mb:
        raise RuntimeError(
            f"VRAM CRITICAL: Only {free_mb:.0f}MB free. "
            f"Minimum required: {min_free_mb}MB. Terminating to prevent OOM crash."
        )
