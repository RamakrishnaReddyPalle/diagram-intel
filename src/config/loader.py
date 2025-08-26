# merges YAML + .env → structured cfg (pydantic)
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from omegaconf import OmegaConf
from dotenv import load_dotenv
from pathlib import Path
import os

class Settings(BaseSettings):
    DEVICE: str = "cpu"
    PRECISION: str = "float16"
    HF_HOME: str = "./models/cache"
    MODEL_CACHE: str = "./models/cache"
    DATA_ROOT: str = "./data"

def load_cfg():
    load_dotenv()
    root = Path(__file__).resolve().parents[2]
    conf = OmegaConf.merge(
        OmegaConf.load(root/"configs/paths.yaml"),
        OmegaConf.load(root/"configs/base.yaml"),
        OmegaConf.load(root/"configs/pipeline.yaml"),
        OmegaConf.load(root/"configs/models.yaml"),
        OmegaConf.load(root/"configs/constraints.yaml"),
    )
    s = Settings()
    os.environ["HF_HOME"] = s.HF_HOME
    conf.env = dict(s)
    return conf
