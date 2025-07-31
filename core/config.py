from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class VAEConf:
    latent_dim: int= 256
    n_layers: int= 2
    beta_kl: float= 0.1
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100   
@dataclass
class NFConf:
    something: int = 0
@dataclass
class DataConf:
    train_audio_folder: str = 'data/nsynth-train/audio'
    test_audio_folder: str = 'data/nsynth-test/audio'
    val_audio_folder: str = 'data/nsynth-valid/audio'
    sample_rate: int = 16000
    n_mels: int = 128
    hop_length: int = 512

@dataclass
class Config:
    vae: VAEConf = field(default_factory=VAEConf)
    nf: NFConf = field(default_factory=NFConf)
    data: DataConf = field(default_factory=DataConf)

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"
    seed: int = 0

def load_config(config_path: Optional[str] = None):
    """Load configuration from file or create default"""
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.structured(Config())
    
    return cfg

def save_config(cfg: DictConfig, save_path: str) -> None:
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(cfg, save_path)

def get_model_config(cfg: DictConfig, model_type: str):
    """Get model-specific configuration"""
    if model_type.lower() == "vae":
        return cfg.vae
    elif model_type.lower() == "nf":
        return cfg.nf
    else:
        raise ValueError(f"Unknown model type: {model_type}")



