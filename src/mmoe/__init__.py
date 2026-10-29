"""Multi-head mixture-of-experts MLP encoder with expert-choice routing (M-MoE, EC). See README.md."""
from .config import ModelConfig, PhaseConfig, TrainConfig, load_config, model_config, train_config
from .diagnostics import assignments, head_diagnostics
from .encoder import MMoEEncoder
from .prior import abs_corr_prior
from .schedule import PhaseController
from .train import FitResult, build_model, encode, evaluate_recon, fit

__version__ = "0.1.0"

__all__ = ["MMoEEncoder", "PhaseController", "ModelConfig", "PhaseConfig", "TrainConfig", "load_config",
           "model_config", "train_config", "abs_corr_prior", "fit", "FitResult", "build_model", "encode",
           "evaluate_recon", "assignments", "head_diagnostics"]
