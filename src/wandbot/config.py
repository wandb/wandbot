from types import SimpleNamespace

TEAM = "wandbot"
PROJECT = "wandb_bot"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    faiss_artifact="parambharat/wandb_docs_bot/faiss_store:latest",
    hyde_prompt_artifact="parambharat/wandb_docs_bot/hyde_prompt:latest",
    chat_prompt_artifact="parambharat/wandb_docs_bot/system_prompt:latest",
    model_name="gpt-4-32k",
)
