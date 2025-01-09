import os
import wandb
from wandbot.configs.vectorstore_config import VectorStoreConfig

from dataclasses import dataclass
import simple_parsing as sp

@dataclass
class Config:
    debug: bool = False  # Debug flag for quick testing
    index_dir: str = None
    artifact_url: str = None

config = sp.parse(Config)
vsconfig = VectorStoreConfig()

if config.index_dir is None:
    config.index_dir = vsconfig.index_dir

if config.artifact_url is None:
    config.artifact_url = vsconfig.artifact_url

api_key = os.getenv("WANDB_API_KEY")
if not api_key:
    raise ValueError("WANDB_API_KEY environment variable is not set")

wandb.login(key=api_key)
api = wandb.Api()
art = api.artifact(config.artifact_url)  # Download vectordb index from W&B
print(f"Downloading index to {config.index_dir}")
os.makedirs(config.index_dir, exist_ok=True)
save_dir = art.download(config.index_dir)

print(f"Downloaded index to {save_dir}") 