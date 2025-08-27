# pulls open weights locally via huggingface_hub
# usage: python scripts/download_models.py
#!/usr/bin/env python
import json, os, sys, pathlib
from typing import Dict, Any, Tuple, List
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

ROOT = pathlib.Path(__file__).resolve().parents[1]

def resolve_local_path(meta, cfg) -> str:
    # Resolve any ${...} interpolations in local_path
    try:
        return os.path.abspath(os.path.expanduser(OmegaConf.to_container(meta["local_path"], resolve=True)))
    except Exception:
        return os.path.abspath(os.path.expanduser(str(meta["local_path"])))

def is_model_present(local_dir: str) -> bool:
    """Heuristic: if folder has config.json AND any weight file, we consider it present."""
    if not os.path.isdir(local_dir):
        return False
    has_config = os.path.isfile(os.path.join(local_dir, "config.json"))
    # Look for sharded weights or single-file binaries
    try:
        entries = os.listdir(local_dir)
    except Exception:
        return False
    has_weights = any(
        name.endswith(".safetensors") or name.endswith(".bin")
        for name in entries
    )
    return has_config and has_weights

def main():
    paths_cfg  = OmegaConf.load(ROOT / "configs" / "paths.yaml")
    models_cfg = OmegaConf.load(ROOT / "configs" / "models.yaml")
    cfg = OmegaConf.merge(paths_cfg, models_cfg)

    model_cache = os.path.abspath(os.path.expanduser(cfg.paths.model_cache))
    os.makedirs(model_cache, exist_ok=True)
    os.environ["HF_HOME"] = model_cache

    vlm_cfg: Dict[str, Any] = cfg.get("vlm", {})
    if not vlm_cfg:
        print("[warn] No 'vlm' section found in configs/models.yaml")
        return 0

    resolved = {}
    skipped: List[Tuple[str, str]] = []
    downloaded: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    for name, meta in vlm_cfg.items():
        if not meta.get("enabled", False):
            continue

        repo = meta["hf_repo_id"]
        local_path = resolve_local_path(meta, cfg)
        os.makedirs(local_path, exist_ok=True)

        if is_model_present(local_path):
            print(f"[skip] {name} already present at {local_path}")
            skipped.append((name, local_path))
            resolved[name] = {"repo_id": repo, "local_path": local_path, "status": "present"}
            continue

        print(f"[download] {name} \u2190 {repo}")
        try:
            p = snapshot_download(
                repo_id=repo,
                local_dir=local_path,
                resume_download=True,        # resume/incremental
            )
            resolved[name] = {"repo_id": repo, "local_path": p, "status": "downloaded"}
            downloaded.append((name, p))
        except GatedRepoError as e:
            msg = str(e).splitlines()[0]
            print(f"[gated] {name}: {msg}")
            failed.append((name, f"gated: {msg}"))
        except HfHubHTTPError as e:
            print(f"[http] {name}: {e}")
            failed.append((name, f"http: {e}"))
        except Exception as e:
            print(f"[error] {name}: {e}")
            failed.append((name, f"error: {e}"))

    out = ROOT / "models" / "registry.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)
    print(f"\n[ok] wrote {out}")

    # Summary
    print("\nSummary:")
    if skipped:
        for n,p in skipped: print(f"  - skipped:    {n} ({p})")
    if downloaded:
        for n,p in downloaded: print(f"  - downloaded: {n} ({p})")
    if failed:
        for n,why in failed: print(f"  - failed:     {n} ({why})")

    # exit code: 0 if at least one success or skip; 1 only if all failed
    return 0 if (skipped or downloaded) else (1 if failed else 0)

if __name__ == "__main__":
    sys.exit(main())
