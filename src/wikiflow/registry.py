"""Human-readable binary model names with explicit historical-ID compatibility."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = json.loads((ROOT / "models.json").read_text())
BY_ID = {model["id"]: model for model in MODELS}


def resolve_model(value):
    for model in MODELS:
        names = [model["id"], model["legacy_id"], model["name"]]
        if value.casefold() in [name.casefold() for name in names if name]:
            return model
    raise ValueError(f"Unknown binary model {value!r}; run 'python run.py models'.")


def main_models():
    return [model for model in MODELS if model["role"] == "main"]


def ablation_models():
    return [BY_ID[name] for name in (
        "traffic_logistic", "text_availability_control", "traffic_text_logistic")]
