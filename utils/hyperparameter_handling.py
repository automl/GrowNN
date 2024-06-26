from typing import Tuple
from omegaconf import DictConfig
import os


def extract_hyperparameters(config: DictConfig) -> Tuple[int, float, float, float, float, float, float, int, int, bool, float]:
    batch_size = config["batch_size"]
    clip_range = config["clip_range"]
    clip_range_vf = None if config["clip_range_vf"] == "None" else config["clip_range_vf"]
    ent_coef = config["ent_coef"]
    gae_lambda = config["gae_lambda"]
    learning_rate = config["learning_rate"]
    max_grad_norm = config["max_grad_norm"]
    n_epochs = config["n_epochs"]
    n_steps = config["n_steps"]
    normalize_advantage = config["normalize_advantage"]
    vf_coef = config["vf_coef"]
    feature_extractor_output_dimension = config["feature_extractor_output_dimension"]
    n_feature_extractor_layers = config["n_feature_extractor_layers"]
    feature_extractor_layer_width = config["feature_extractor_layer_width"]

    return (
        batch_size,
        clip_range,
        clip_range_vf,
        ent_coef,
        gae_lambda,
        learning_rate,
        max_grad_norm,
        n_epochs,
        n_steps,
        normalize_advantage,
        vf_coef,
        feature_extractor_output_dimension,
        n_feature_extractor_layers,
        feature_extractor_layer_width,
    )


def create_model_save_path(model_save_path: str, config: DictConfig, budget, seed) -> str:
    return os.path.join(model_save_path, str(extract_hyperparameters(config)), str(budget), str(seed))
