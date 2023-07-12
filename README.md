# wandbot

A question answering bot for Weights & Biases [documentation](https://docs.wandb.ai/). This bot is built using [langchain](https://python.langchain.com/en/latest/) and llms.

## Features

- Utilizes advanced techniques such as HydeEmbeddings, FAISS index, and a fallback mechanism for model selection
- Efficiently handles user queries and provides accurate, context-aware responses
- Integrated with Discord and Slack, allowing seamless integration into popular collaboration platforms
- Logging and analysis with Weights & Biases Stream Table for performance monitoring and continuous improvement
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

To ingest the data, you first need to clone the [docodile](https://github.com/wandb/docodile) and [examples](https://github.com/wandb/examples) into the data directory.
The data directory is located at `wandbot/data`.
To clone the repositories, follow the commands:

```bash
cd wandbot/data
git clone git@github.com:your_md_repo/your_md_docs.git
git clone git@github.com:your_python_repo/your_python_codebase.git
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

After cloning the repositories, run the data ingestion command provided in the "Data Ingestion" section (…). This will create a `documents.jsonl` file in the `wandbot/data` directory and a `faiss_index` directory.
The `documents.jsonl` file contains the documents that will be used to create the FAISS index.
The `faiss_index` directory contains the FAISS index that will be used to retrieve the documents.
Additionally, the `documents.jsonl`, `faiss_index`, and `hyde_prompt` will be uploaded to W&B as artifacts to the `wandb_project` specified in the command.

### Running the Q&A Bot

To run the Q&A bot, use the provided command:

```bash
cd wandbot
cd discord
# or
# cd slack_app
poetry run python -m main
```

This will start the chatbot application, allowing you to interact with it and ask questions related to the Weights & Biases documentation.

### Evaluation

To evaluate the performance of the Q&A bot, the provided evaluation script (…) can be used. This script utilizes a separate dataset for evaluation, which can be stored as a W&B Artifact. The evaluation script calculates retrieval accuracy, average string distance, and chat model accuracy.

The evaluation script downloads the evaluation dataset from the specified W&B Artifact, performs the evaluation using the Q&A bot, and then logs the results, such as retrieval accuracy, average string distance, and chat model accuracy, back to W&B. The logged results can be viewed on the W&B dashboard.

To run the evaluation script, use the provide the following commands:

```bash
cd wandbot
poetry run python -m eval
```

## Implementation Overview

1. HydeEmbeddings and Langchain for Document Embedding
2. Document Embeddings with FAISS
3. Building the Q&A Pipeline
4. Model Selection and Fallback Mechanism
5. Deploying the Q&A Bot on Discord and Slack
6. Logging and Analysis with Weights & Biases Stream Table
7. Evaluation of the Q&A Bot
