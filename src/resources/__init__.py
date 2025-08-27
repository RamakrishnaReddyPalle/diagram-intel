# src/resources/__init__.py
from pathlib import Path
import yaml

def load_device_catalog(root: str) -> dict:
    """
    Look for resources/device_catalog.yml in common locations:
    - <root>/resources/device_catalog.yml
    - <root>/src/resources/device_catalog.yml
    """
    candidates = [
        Path(root) / "resources" / "device_catalog.yml",
        Path(root) / "src" / "resources" / "device_catalog.yml",
    ]
    for p in candidates:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "device_catalog.yml not found. Looked in:\n  - " +
        "\n  - ".join(str(c) for c in candidates)
    )
