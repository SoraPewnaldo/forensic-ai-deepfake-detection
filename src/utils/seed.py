import random
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    P10: Enforce deterministic seeding across ALL sources of randomness.
    Called at the top of every __main__ entrypoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner (non-deterministic)

    # Verification (P10)
    assert torch.initial_seed() == seed, \
        f"Seed verification failed: expected {seed}, got {torch.initial_seed()}"

    print(f"[SEED] Global seed locked to {seed}. Results are fully reproducible.")
