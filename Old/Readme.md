project:
name: FORENSIC-AI
owner: Ayush Dakwal
domain: deepfake-forensics
objective: >
Develop a cross-dataset generalizable deepfake detection system
with likelihood ratio-based forensic calibration suitable for
court-admissible digital evidence analysis.

system:
hardware:
gpu: NVIDIA RTX 3060 Laptop
vram_gb: 6

os:
name: Windows
version: 10
architecture: x64

gpu_stack:
driver: nvidia_geforce_driver (latest_stable)
cuda: 12.x
verified: pending

python:
version: 3.11
venv_path: E:\Projects\workspace\venvs\forensic-ai

ml_stack:
pytorch: cu121
torchvision: true
torchaudio: true

paths:
workspace_root: E:\Projects\workspace\forensic-ai

datasets:
ffpp: E:\Projects\workspace\data\ffpp
wilddeepfake: E:\Projects\workspace\data\wilddeepfake
celebdf: E:\Projects\workspace\data\celebdf

outputs:
checkpoints: E:\Projects\workspace\forensic-ai\checkpoints
logs: E:\Projects\workspace\forensic-ai\logs
results: E:\Projects\workspace\forensic-ai\results

constraints:
vram_limit_gb: 6
batch_size_max: 8
mixed_precision: true
gradient_accumulation: true
cuda_required: true
no_dataset_leakage: true
video_level_processing: true
calibration_mandatory: true

datasets:
protocol: staged_cross_dataset

stage1:
name: FaceForensics++
role: base_training
path: E:\Projects\workspace\data\ffpp

stage2:
name: WildDeepfake
role: domain_adaptation
path: E:\Projects\workspace\data\wilddeepfake

stage3:
name: Celeb-DF-v2
role: unseen_evaluation
path: E:\Projects\workspace\data\celebdf

rules:
- no_cross_mixing
- strict_split: [train, validation, calibration, test]

data_pipeline:
frame_sampling:
method: uniform
frames_per_video: 32

preprocessing:
resize: [224, 224]
normalization: imagenet

augmentations:
compression: true
gaussian_noise: true
blur: true

model:
architecture: multimodal_spatiotemporal_frequency

spatial_backbone:
type: vit_base_patch16_224
source: timm
pretrained: true
freeze_initial: true

temporal_module:
type: attention
purpose: frame_weighting
hidden_dim: 512
num_heads: 4

frequency_branch:
enabled: true
transform: dct_fft
cnn:
layers: 3
channels: [32, 64, 128]
purpose: detect_compression_and_synthesis_artifacts

fusion:
method: concatenation
inputs:
- spatial
- temporal
- frequency
output_dim: 512

classifier:
type: mlp
layers: [512, 128, 1]
output: raw_score

training:
strategy: staged_transfer_learning

stage1:
name: base_training
dataset: FaceForensics++
freeze_backbone: true
train_modules:
- temporal_module
- frequency_branch
- classifier
epochs: 10
batch_size: 4
lr: 1e-4

stage2:
name: domain_adaptation
dataset: WildDeepfake
freeze_backbone: partial
unfreeze:
type: top_layers
count: 4
lr: 5e-5
epochs: 5
batch_size: 4

stage3:
name: evaluation_only
dataset: Celeb-DF-v2
training: false

optimization:
optimizer: adamw
weight_decay: 1e-4
scheduler:
type: cosine
warmup_epochs: 1

loss:
type: binary_cross_entropy_with_logits

calibration:
enabled: true
method: logistic_regression
input: raw_scores
output: likelihood_ratio
dataset_split: calibration_only

hypothesis:
H1: manipulated_video
H0: real_video

evaluation:
discrimination:
metrics:
- auc
- hter
- eer

calibration:
metrics:
- cllr
- tippett_plot

cross_dataset_tests:
- train: FaceForensics++
test: WildDeepfake
- train: FaceForensics++
test: Celeb-DF-v2
- train: WildDeepfake
test: Celeb-DF-v2

output:
format:
prediction: [real, fake]
likelihood_ratio: float
interpretation: enabled

lr_interpretation:
strong_fake: ">10"
moderate_fake: "1-10"
inconclusive: "~1"
moderate_real: "0.1-1"
strong_real: "<0.1"

logging:
checkpoints: true
checkpoint_dir: E:\Projects\workspace\forensic-ai\checkpoints
logs: E:\Projects\workspace\forensic-ai\logs
verbosity: high

execution_plan:
steps:
- verify_pytorch_cuda
- implement_dataset_loader
- build_frame_extraction_pipeline
- load_spatial_backbone
- implement_temporal_attention
- implement_frequency_branch
- integrate_feature_fusion
- build_training_pipeline
- implement_calibration_module
- run_cross_dataset_evaluation

data_splitting:
method: subject_disjoint
ratios:
train: 0.7
validation: 0.1
calibration: 0.1
test: 0.1
