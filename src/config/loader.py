# src/config/loader.py
from pydantic_settings import BaseSettings
from omegaconf import OmegaConf, DictConfig, ListConfig
from dotenv import load_dotenv
from pathlib import Path
import os, warnings

class Settings(BaseSettings):
    DEVICE: str = "cpu"
    PRECISION: str = "float16"
    HF_HOME: str = "./models/cache"
    MODEL_CACHE: str = "./models/cache"
    DATA_ROOT: str = "./data"
    LOG_LEVEL: str = "INFO"

def _abs(root: Path, p: str) -> str:
    pth = Path(p)
    return str((pth if pth.is_absolute() else (root / pth)).resolve())

def _to_container(cfg):
    # resolve interpolations before merging as plain python
    return OmegaConf.to_container(cfg, resolve=True)

def _deep_soft_merge(a, b, path=""):
    """
    Dict-vs-dict → recursive merge.
    Any other type conflict → b replaces a.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = _deep_soft_merge(out[k], v, f"{path}.{k}" if path else k)
            else:
                out[k] = v
        return out
    else:
        # type mismatch or lists or scalars → overlay wins
        return b

def _merge_yaml(cfg, path_obj):
    """
    Load YAML and merge into cfg.
    1) If YAML root is list, wrap under {"overlay": [...]}
    2) Try OmegaConf.merge
    3) If it fails due to type conflicts, soft-merge in python
    """
    p = Path(path_obj)
    if not p.exists():
        return cfg
    y = OmegaConf.load(p)
    if isinstance(y, ListConfig):
        warnings.warn(f"[loader] '{p}' has a top-level list; wrapping under key 'overlay'.")
        y = OmegaConf.create({"overlay": y})

    try:
        return OmegaConf.merge(cfg, y)
    except Exception as e:
        warnings.warn(f"[loader] Hard merge failed for '{p}' ({e.__class__.__name__}); "
                      f"falling back to soft replace-on-mismatch.")
        base = _to_container(cfg) if not isinstance(cfg, (dict, list)) else cfg
        over = _to_container(y)   if not isinstance(y,   (dict, list)) else y
        merged = _deep_soft_merge(base, over)
        return OmegaConf.create(merged)

def load_cfg():
    load_dotenv()

    def _env_resolver(var, default=None):
        return os.environ.get(var, default)
    OmegaConf.register_new_resolver("env", _env_resolver, replace=True)
    OmegaConf.register_new_resolver("oc.env", _env_resolver, replace=True)

    root = Path(__file__).resolve().parents[2]

    # ---- core configs (unchanged order) ----
    conf = OmegaConf.merge(
        OmegaConf.load(root / "configs" / "paths.yaml"),
        OmegaConf.load(root / "configs" / "base.yaml"),
        OmegaConf.load(root / "configs" / "pipeline.yaml"),
        OmegaConf.load(root / "configs" / "models.yaml"),
    )

    # ---- constraints layering (packs + project + legacy) ----
    constraints = _merge_yaml(OmegaConf.create(), root / "configs" / "constraints" / "base.yaml")

    packs_env = os.getenv("CONSTRAINTS_PACKS", "generic")
    for pack in [p.strip() for p in packs_env.split(",") if p.strip()]:
        constraints = _merge_yaml(constraints, root / f"configs/constraints/packs/{pack}.yaml")

    project_env = os.getenv("PROJECT_CONSTRAINTS", "")
    if project_env:
        constraints = _merge_yaml(constraints, root / f"configs/constraints/projects/{project_env}.yaml")

    legacy = root / "configs" / "constraints.yaml"
    if legacy.exists():
        constraints = _merge_yaml(constraints, legacy)

    # Attach into main cfg tree
    conf = OmegaConf.merge(conf, OmegaConf.create({"constraints": constraints}))

    # ---- optional profile overlay (unchanged) ----
    profile = os.environ.get("CFG_PROFILE")
    if profile:
        prof_path = root / "configs" / "profiles" / f"{profile}.yaml"
        if prof_path.exists():
            conf = OmegaConf.merge(conf, OmegaConf.load(prof_path))

    # ---- env passthrough ----
    s = Settings()
    os.environ["HF_HOME"] = s.HF_HOME

    # ---- normalize paths.* ----
    paths_dict = OmegaConf.to_container(conf.paths, resolve=True)
    for k, v in list(paths_dict.items()):
        if isinstance(v, str):
            paths_dict[k] = _abs(root, v)
    conf.paths = OmegaConf.create(paths_dict)

    # ---- RUN_ID workspace redirect (so you don't touch old data) ----
    run_id = os.environ.get("RUN_ID")
    if run_id:
        ws_root = Path(conf.paths.data_root) / "_runs" / run_id
        conf.paths.raw       = str(ws_root / "raw")
        conf.paths.interim   = str(ws_root / "interim")
        conf.paths.processed = str(ws_root / "processed")
        conf.paths.exports   = str(ws_root / "exports")

    # ---- normalize prompts.* ----
    if "prompts" in conf:
        prompts_dict = OmegaConf.to_container(conf.prompts, resolve=True)
        for k, v in list(prompts_dict.items()):
            if isinstance(v, str):
                prompts_dict[k] = _abs(root, v)
        conf.prompts = OmegaConf.create(prompts_dict)

    conf.env = dict(s)
    conf.root = str(root)
    return conf
