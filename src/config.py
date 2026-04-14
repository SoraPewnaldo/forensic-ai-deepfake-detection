import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    ffpp: Path
    wilddeepfake: Path
    celebdf: Path
    celebdf_raw: Path
    ffpp_json: Path

@dataclass(frozen=True)
class ExtractionConfig:
    n_frames: int
    bbox_thresh: float
    crop_size: int
    margin: float

@dataclass(frozen=True)
class SpatialConfig:
    backbone: str
    pretrained: bool
    clip_hidden_dim: int
    freeze_blocks: int
    gradient_checkpointing: bool

@dataclass(frozen=True)
class FrequencyConfig:
    enabled: bool
    input_size: int
    channels: int

@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool
    frames: int
    d_model: int
    n_heads: int

@dataclass(frozen=True)
class ArchitectureConfig:
    spatial: SpatialConfig
    frequency: FrequencyConfig
    temporal: TemporalConfig

@dataclass(frozen=True)
class LRsConfig:
    classifier_head: float
    unfrozen_backbone: float

@dataclass(frozen=True)
class AugmentationsConfig:
    hflip_p: float
    color_jitter_p: float
    jpeg_p: float
    jpeg_min_q: int
    jpeg_max_q: int

@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    img_size: int
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    lrs: LRsConfig
    epochs: int
    early_stopping_patience: int
    augmentations: AugmentationsConfig
    label_smoothing: float

@dataclass(frozen=True)
class CalibrationConfig:
    method: str
    temperature_init: float
    temperature_min: float
    partition: str
    lr_formula: str

@dataclass(frozen=True)
class ForensicConfig:
    paths: PathConfig
    extraction: ExtractionConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    calibration: CalibrationConfig

def _load_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _build_config(yaml_data: Dict[str, Any]) -> ForensicConfig:
    """Builds typed, frozen dataclass from raw yaml dictionary."""
    
    paths_data = yaml_data['paths']
    paths = PathConfig(
        project_root=Path(paths_data['project_root']).resolve(),
        ffpp=Path(paths_data['datasets']['ffpp']).resolve(),
        wilddeepfake=Path(paths_data['datasets']['wilddeepfake']).resolve(),
        celebdf=Path(paths_data['datasets']['celebdf']).resolve(),
        celebdf_raw=Path(paths_data['datasets']['celebdf_raw']).resolve(),
        ffpp_json=Path(paths_data['splits']['ffpp_json']).resolve()
    )
    
    ext_data = yaml_data['extraction']
    extraction = ExtractionConfig(
        n_frames=ext_data['n_frames'],
        bbox_thresh=ext_data['bbox_thresh'],
        crop_size=ext_data['crop_size'],
        margin=ext_data['margin']
    )
    
    arch_data = yaml_data['architecture']
    architecture = ArchitectureConfig(
        spatial=SpatialConfig(**arch_data['spatial']),
        frequency=FrequencyConfig(**arch_data['frequency']),
        temporal=TemporalConfig(**arch_data['temporal'])
    )
    
    train_data = yaml_data['training']
    training = TrainingConfig(
        seed=train_data['seed'],
        img_size=train_data['img_size'],
        batch_size=train_data['batch_size'],
        gradient_accumulation_steps=train_data['gradient_accumulation_steps'],
        num_workers=train_data['num_workers'],
        lrs=LRsConfig(**train_data['lrs']),
        epochs=train_data['epochs'],
        early_stopping_patience=train_data['early_stopping_patience'],
        augmentations=AugmentationsConfig(**train_data['augmentations']),
        label_smoothing=train_data['label_smoothing']
    )

    calib_data = yaml_data['calibration']
    calibration = CalibrationConfig(**calib_data)
    
    return ForensicConfig(
        paths=paths, 
        extraction=extraction, 
        architecture=architecture, 
        training=training,
        calibration=calibration
    )

# The Global Application Config Instance
# Automatically load 'config.yaml' relative to the project root
_ROOT_DIR = Path(__file__).parent.parent
_YAML_PATH = _ROOT_DIR / "config.yaml"

if not _YAML_PATH.exists():
    raise FileNotFoundError(f"CRITICAL ERROR: config.yaml not found at {_YAML_PATH}. System terminating.")

config = _build_config(_load_yaml(_YAML_PATH))
