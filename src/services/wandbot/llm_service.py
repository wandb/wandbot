# llm-service-adapter/services/wandbot/llm_service.py
from .chat import Chat
from global_config import TEAM, PROJECT, JOB_TYPE
from .default_config import default_config
import wandb
import openai
import os

#TODO: move this to a better place
openai.api_key = os.environ.get("OPENAI_API_KEY")

class WandbotLLMService:
    def __init__(self, config: dict = {}):
        _config = default_config.copy()
        _config.update(config)
        self.wandb_run = self.init_wandb(_config)
        self.chat = Chat(model_name=self.wandb_run.config.model_name, wandb_run=self.wandb_run)

    def init_wandb(self, config: dict):
        return wandb.init(
            entity=TEAM,
            project=PROJECT,
            job_type=JOB_TYPE,
            config=config,
        )

    def chat(self, query: str):
        return self.chat(query)
