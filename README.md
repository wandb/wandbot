# wandbot

A question answering bot for Weights & Biases [documentation](https://docs.wandb.ai/).
This bot is built using [llama-index](https://gpt-index.readthedocs.io/en/stable/) and openai gpt-4.

## Features

- Utilizes retrieval augmented generation, and a fallback mechanism for model selection.
- Efficiently handles user queries and provides accurate, context-aware responses
- Integrated with Discord and Slack, allowing seamless integration into popular collaboration platforms.
- Logging and analysis with Weights & Biases Tables for performance monitoring and continuous improvement.
- Evaluation using a combination of metrics such as retrieval accuracy, string similarity, and model-generated response correctness

## Installation

The project uses `python = ">=3.10.0,<3.11"` and uses [poetry](https://python-poetry.org/) for dependency management. To install the dependencies:

```bash
git clone git@github.com:wandb/wandbot.git
pip install poetry
cd wandbot
poetry install
# Depending on which platform you want to run on run the following command:
# poetry install --extras discord # for discord
# poetry install --extras slack # for slack
# poetry install --extras api # for api
```

## Usage

### Data Ingestion

The data ingestion module pulls code and markdown from Weights & Biases repositories [docodile](https://github.com/wandb/docodile) and [examples](https://github.com/wandb/examples) ingests them into vectorstores for the retrieval augmented generation pipeline.
To ingest the data run the following command from the root of the repository
```bash
poetry run python -m src.wandbot.ingestion
```
You will notice that the data is ingested into the `data/cache` directory and stored in three different directories `raw_data`, `transformed_data`, `retriever_data` with individual files for each step of the ingestion process.
These datasets are also stored as wandb artifacts in the project defined in the environment variable `WANDB_PROJECT` and can be accessed from the [wandb dashboard](https://wandb.ai/wandb/wandbot).


### Running the Q&A Bot

You will need to set the following environment variables:

```bash
OPENAI_API_KEY
SLACK_APP_TOKEN
SLACK_BOT_TOKEN
SLACK_SIGNING_SECRET
WANDB_API_KEY
DISCORD_BOT_TOKEN
WANDBOT_API_URL="http://localhost:8000"
WANDB_TRACING_ENABLED="true"
WANDB_PROJECT="wandbot-dev"
WANDB_ENTITY="wandbot"
```
Then you can run the Q&A bot application, use the following commands:
```bash
(poetry run uvicorn wandbot.api.app:app --host="0.0.0.0" --port=8000 > api.log 2>&1) & \
(poetry run python -m wandbot.apps.slack > slack_app.log 2>&1) & \
(poetry run python -m wandbot.apps.discord > discord_app.log 2>&1)
```

This will start the chatbot applications - the api, the slackbot and the discord bot, allowing you to interact with it and ask questions related to the Weights & Biases documentation.

### Evaluation

To evaluate the performance of the Q&A bot, the provided evaluation script (…) can be used. This script utilizes a separate dataset for evaluation, which can be stored as a W&B Artifact. The evaluation script calculates retrieval accuracy, average string distance, and chat model accuracy.

The evaluation script downloads the evaluation dataset from the specified W&B Artifact, performs the evaluation using the Q&A bot, and then logs the results, such as retrieval accuracy, average string distance, and chat model accuracy, back to W&B. The logged results can be viewed on the W&B dashboard.

To run the evaluation script, use the following commands:

```bash
cd wandbot
poetry run python -m eval
```

## Implementation Overview

1. Document Embeddings with FAISS
2. Building the Q&A Pipeline with llama-index
3. Model Selection and Fallback Mechanism
4. Deploying the Q&A Bot on FastAPI, Discord and Slack
5. Logging and Analysis with Weights & Biases Tables
6. Evaluation of the Q&A Bot

You can track the bot usage in the following project:
https://wandb.ai/wandbot/wandbot_public
