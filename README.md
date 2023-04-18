# WandBot

A question answering bot for Weights & Biases [documentation](https://docs.wandb.ai/). This bot is built using [langchain](https://python.langchain.com/en/latest/) and llms. We utilize the LLM Service Adapter Framework, designed to work across multiple platforms such as Discord, Slack, and FastAPI. This guide will walk you through the installation and usage of WandBot on various platforms.

To understand more about the code and framework please refer to the [src ReadMe.md](./src/Readme.md).

## Installation

The project uses `python = ">=3.10.0,<3.11"` and uses [poetry](https://python-poetry.org/) for dependency management. To install the dependencies, run the following commands:

```bash
git clone git@github.com:wandb/wandbot.git
pip install poetry
cd wandbot
poetry install
# If anything is missing, run the following command
pip install -r requirements.txt
```

Set the necessary environment variables, such as API keys and tokens, for the desired platform.

## Running WandBot on Different Platforms

```bash
# Needed env variables
WANDB_API_KEY=
OPENAI_API_KEY=
DISCORD_BOT_TOKEN=
SLACK_BOT_TOKEN=
SLACK_APP_TOKEN=
```

### Discord

To run WandBot on Discord:

1. Set the `DISCORD_TOKEN` environment variable with your Discord bot token.
2. Run the following command:

```bash
python main.py --service wandbot --app discord
```

### Slack

To run WandBot on Slack:

1. Set the SLACK_APP_TOKEN and SLACK_BOT_TOKEN environment variables with your Slack app and bot tokens.
2. Run the following command:

```bash
python main.py --service wandbot --app slack
```

### FastAPI

To run WandBot as an API using FastAPI:

1. Run the following command:

```bash
python main.py --service wandbot --app fastapi
```

2. Access the FastAPI server at http://0.0.0.0:8000.
3. POST to http://0.0.0.0:8000/query/

## Usage

### Data Ingestion

To ingest the data you first need to clone the [docodile](https://github.com/wandb/docodile) and [examples](https://github.com/wandb/examples) into the data directory.
The data directory is located at `wandbot/data`.
To clone the repositories, run the following commands:

```bash
cd wandbot/data
git clone git@github.com:wandb/docodile.git
git clone git@github.com:wandb/examples.git
```

To ingest the data, run the following command:

```bash
cd wandbot
poetry run python -m wandbot.ingest --docs_dir data/docodile \
--documents_file data/documents.jsonl \
--faiss_index data/faiss_index \
--hyde_prompt data/prompts/hyde_prompt.txt \
--use_hyde \
--temperature 0.3 \
--wandb_project wandb_docs_bot_dev
```

This will create a `documents.jsonl` file in the `wandbot/data` directory and a `faiss_index` directory.
The `documents.jsonl` file contains the documents that will be used create the faiss index.
The `faiss_index` directory contains the faiss index that will be used to retrieve the documents.
Additionally, the `documents.jsonl`, `faiss_index` and `hyde_prompt` will be uploaded to wandb as artifacts to the `wandb_project` specified in the command.
