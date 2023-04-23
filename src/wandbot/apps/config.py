import pathlib

from pydantic import BaseModel
from wandbot.ingestion.settings import VectorIndexConfig


class ChatConfig(BaseModel):
    model_name: str = "gpt-4"
    max_retries: int = 1
    fallback_model_name: str = "gpt-3.5-turbo"
    max_fallback_retries: int = 6
    chat_temperature: float = 0.0
    chain_type: str = "stuff"
    chat_prompt: pathlib.Path = pathlib.Path("data/prompts/chat_prompt.txt")
    vector_index_config: VectorIndexConfig = VectorIndexConfig(
        wandb_project="wandb_docs_bot_dev"
    )
    vector_index_artifact: str = (
        "parambharat/wandb_docs_bot_dev/wandbot_vectorindex:latest"
    )
    wandb_project: str = "wandb_docs_bot_dev"
    wandb_entity: str = "wandb"
    wandb_job_type: str = "production"
    include_sources: bool = True
    source_score_threshold: float = 1.0
    query_tokens_threshold: int = 1024
