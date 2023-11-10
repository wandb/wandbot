# wandbot

Wandbot is a question-answering bot designed specifically for Weights & Biases [documentation](https://docs.wandb.ai/).
Leveraging the power of [llama-index](https://gpt-index.readthedocs.io/en/stable/) and OpenAI's [gpt-4](https://openai.com/research/gpt-4), it provides precise and context-aware responses
using a combination of [FAISS](https://github.com/facebookresearch/faiss) for RAG and OpenAI's [gpt-4](https://openai.com/research/gpt-4) for generating responses.


## Features

- Wandbot employs Retrieval Augmented Generation with a [FAISS](https://github.com/facebookresearch/faiss) backend, ensuring efficient and accurate responses to user queries by retrieving relevant documents.
- It features periodic data ingestion and report generation, contributing to the bot's continuous improvement. You can view the latest data ingestion report [here](https://wandb.ai/wandbot/wandbot-dev/reportlist).
- The bot is integrated with Discord and Slack, facilitating seamless integration with these popular collaboration platforms.
- Performance monitoring and continuous improvement are made possible through logging and analysis with Weights & Biases Tables. Visit the workspace for more details [here](https://wandb.ai/wandbot/wandbot_public).
- Wandbot has a fallback mechanism for model selection, which is used when GPT-4 fails to generate a response.
- The bot's performance is evaluated using a mix of metrics, including retrieval accuracy, string similarity, and the correctness of model-generated responses.
- Curious about the custom system prompt used by the bot? You can view the full prompt [here](data/prompts/chat_prompt.json).

## Installation

The project is built with Python version `>=3.10.0,<3.11` and utilizes [poetry](https://python-poetry.org/) for managing dependencies. Follow the steps below to install the necessary dependencies:

```bash
git clone git@github.com:wandb/wandbot.git
pip install poetry
cd wandbot
poetry install --all-extras
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
You will notice that the data is ingested into the `data/cache` directory and stored in three different directories `raw_data`, `vectorstore` with individual files for each step of the ingestion process.
These datasets are also stored as wandb artifacts in the project defined in the environment variable `WANDB_PROJECT` and can be accessed from the [wandb dashboard](https://wandb.ai/wandb/wandbot-dev).


### Running the Q&A Bot

Before running the Q&A bot, ensure the following environment variables are set:

```bash
OPENAI_API_KEY
COHERE_API_KEY
SLACK_APP_TOKEN
SLACK_BOT_TOKEN
SLACK_SIGNING_SECRET
WANDB_API_KEY
DISCORD_BOT_TOKEN
COHERE_API_KEY
WANDBOT_API_URL="http://localhost:8000"
WANDB_TRACING_ENABLED="true"
WANDB_PROJECT="wandbot-dev"
WANDB_ENTITY="wandbot"
```

Once these environment variables are set, you can start the Q&A bot application using the following commands:

```bash
(poetry run uvicorn wandbot.api.app:app --host="0.0.0.0" --port=8000 > api.log 2>&1) & \
(poetry run python -m wandbot.apps.slack > slack_app.log 2>&1) & \
(poetry run python -m wandbot.apps.discord > discord_app.log 2>&1)
```

For more detailed instructions on installing and running the bot, please refer to the [run.sh](./run.sh) file located in the root of the repository.

Executing these commands will launch the API, Slackbot, and Discord bot applications, enabling you to interact with the bot and ask questions related to the Weights & Biases documentation.

### Evaluation

To evaluate the performance of the Q&A bot, the provided evaluation script (…) can be used. This script utilizes a separate dataset for evaluation, which can be stored as a W&B Artifact. The evaluation script calculates retrieval accuracy, average string distance, and chat model accuracy.

The evaluation script downloads the evaluation dataset from the specified W&B Artifact, performs the evaluation using the Q&A bot, and then logs the results, such as retrieval accuracy, average string distance, and chat model accuracy, back to W&B. The logged results can be viewed on the W&B dashboard.

To run the evaluation script, use the following commands:

```bash
cd wandbot
poetry run python -m eval
```

## Overview of the Implementation

1. Creating Document Embeddings with FAISS
2. Constructing the Q&A Pipeline using llama-index
3. Selection of Models and Implementation of Fallback Mechanism
4. Deployment of the Q&A Bot on FastAPI, Discord, and Slack
5. Utilizing Weights & Biases Tables for Logging and Analysis
6. Evaluating the Performance of the Q&A Bot

You can monitor the usage of the bot in the following project:
https://wandb.ai/wandbot/wandbot_public
