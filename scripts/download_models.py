# pulls open weights locally via huggingface_hub
# usage: python scripts/download_models.py
#!/usr/bin/env python

import json, os, sys, pathlib
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    # Merge configs so ${paths.*} interpolations in models.yaml work
    paths_cfg  = OmegaConf.load(ROOT / "configs" / "paths.yaml")
    models_cfg = OmegaConf.load(ROOT / "configs" / "models.yaml")
    cfg = OmegaConf.merge(paths_cfg, models_cfg)

    model_cache = os.path.abspath(os.path.expanduser(cfg.paths.model_cache))
    os.makedirs(model_cache, exist_ok=True)
    os.environ["HF_HOME"] = model_cache

    resolved = {}
    vlm_cfg = cfg.get("vlm", {})

    if not vlm_cfg:
        print("[warn] No 'vlm' section found in configs/models.yaml")
        return

    for name, meta in vlm_cfg.items():
        if not meta.get("enabled", False):
            continue

        repo = meta["hf_repo_id"]

        # Resolve any ${...} in local_path against the merged config
        try:
            local_path = OmegaConf.to_container(meta["local_path"], resolve=True)
        except Exception:
            # Fallback: treat as plain string
            local_path = str(meta["local_path"])

        local_path = os.path.abspath(os.path.expanduser(local_path))
        os.makedirs(local_path, exist_ok=True)

        print(f"[download] {name} ← {repo}")
        p = snapshot_download(
            repo_id=repo,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # safer on Windows
        )
        resolved[name] = {"repo_id": repo, "local_path": p}

    out = ROOT / "models" / "registry.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    sys.exit(main())
