import os
from pathlib import Path
import random

def audit_celebdf():
    root = Path(r'E:\Pojects\workspace\datasets\celebdf')
    if not root.exists():
        print("ROOT NOT FOUND")
        return

    # In Celeb-DF, video folders usually are like 'celebdf_real_id0_0000'
    # where the last component is the 'track index'. 
    # We want to split by VIDEO ID.
    
    folders = sorted([d.name for d in root.iterdir() if d.is_dir()])
    
    reals = [f for f in folders if 'real' in f]
    synths = [f for f in folders if 'synth' in f]
    
    # We strip the last _0000 to get unique videos
    def get_vid(f):
        # Format: celebdf_real_id0_0000
        # Split by underscore and rejoin except last one
        parts = f.split('_')
        return "_".join(parts[:-1])

    real_videos = sorted(list(set([get_vid(r) for r in reals])))
    synth_videos = sorted(list(set([get_vid(s) for s in synths])))
    
    print(f"Total Folders: {len(folders)}")
    print(f"Unique Real Videos: {len(real_videos)}")
    print(f"Unique Synth Videos: {len(synth_videos)}")
    
    # Selection (Seed 42 for methodology compliance)
    random.seed(42)
    proxy_reals = random.sample(real_videos, min(25, len(real_videos)))
    proxy_synths = random.sample(synth_videos, min(25, len(synth_videos)))
    
    print("\nPROXY REAL SAMPLE (First 5):", proxy_reals[:5])
    print("PROXY SYNTH SAMPLE (First 5):", proxy_synths[:5])
    
    return proxy_reals, proxy_synths

if __name__ == "__main__":
    audit_celebdf()
