from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ["checkpoints", "outputs"]:
        out = Path(cfg["paths"][key])
        out.mkdir(parents=True, exist_ok=True)

    return cfg
