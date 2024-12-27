# wandbot

WandBot is a question-answering bot designed specifically for Weights & Biases Models and Weave [documentation](https://docs.wandb.ai/).

## What's New

### wandbot v1.3.0
Note that to trigger the final initalization after running `run.sh`, a request has to be made to the `/startup` endpoint, as tiggering the heavy initialzations during app startup causes replit to timeout:

```bash
curl https://wandbot.replit.app/startup
```
**New:**

- **Move to uv for package management**: Installs and dependency checks cut down from minutes to seconds
- **Support python 3.11 on replit**
- **Move to lazing loading in app.py to help with startup**: Replit app deployments can't seen to handle the delay from loading the app, despite attempting async or background tasks
- **Add wandb artifacts cache cleanup**
- **Turn off web search**: Currently we don't have a web search provider to use.
- **Small formatting updates for weave.op**
- **Add dotenv in app.py for easy env var loads**


### wandbot v1.2.0

This release introduces a number of exciting updates and improvements:

- **Parallel LLM Calls**: Replaced the llama-index with the LECL, enabling parallel LLM calls for increased efficiency.
- **ChromaDB Integration**: Transitioned from FAISS to ChromaDB to leverage metadata filtering and speed.
- **Query Enhancer Optimization**: Improved the query enhancer to operate with a single LLM call.
- **Modular RAG Pipeline**: Split the RAG pipeline into three distinct modules: query enhancement, retrieval, and response synthesis, for improved clarity and maintenance.
- **Parent Document Retrieval**: Introduced parent document retrieval functionality within the retrieval module to enhance contextuality.
- **Sub-query Answering**: Added sub-query answering capabilities in the response synthesis module to handle complex queries more effectively.
- **API Restructuring**: Redesigned the API into separate routers for retrieval, database, and chat operations.

These updates are part of our ongoing commitment to improve performance and usability.

## Evaluation
English 
| wandbot version  | Comment  | response accuracy |
|---|---|---|
| 1.0.0 | our baseline wandbot |  53.8 % |
| 1.1.0 | improvement over baseline; in production for the longest | 72.5 %  | 
| 1.2.0 | our new enhanced wandbot | 81.6 % |


Japanese
| wandbot version  | Comment  | response accuracy |
|---|---|---|
| 1.2.0 | our new enhanced wandbot | 56.3 % |
| 1.2.1 | add translation process | 71.9 % |

## Features

- WandBot uses:
  - a local ChromaDB vector store
  - OpenAI's v3 embeddings
  - GPT-4 for query enhancement and response synthesis
  - Cohere's re-ranking model
- It features periodic data ingestion and report generation, contributing to the bot's continuous improvement. You can view the latest data ingestion report [here](https://wandb.ai/wandbot/wandbot-dev/reportlist).
- The bot is integrated with Discord and Slack, facilitating seamless integration with these popular collaboration platforms.
- Performance monitoring and continuous improvement are made possible through logging and analysis with Weights & Biases Weave
- Has a fallback mechanism for model selection

## Installation

The project is built with Python version `3.11` and utilizes `uv` for dependency management. Follow the steps below to install the necessary dependencies:

```bash
bash build.sh
```

## Usage

### Running WandBot

Before running the Q&A bot, ensure the following environment variables are set:

```bash
OPENAI_API_KEY
COHERE_API_KEY
SLACK_EN_APP_TOKEN
SLACK_EN_BOT_TOKEN
SLACK_EN_SIGNING_SECRET
SLACK_JA_APP_TOKEN
SLACK_JA_BOT_TOKEN
SLACK_JA_SIGNING_SECRET
WANDB_API_KEY
DISCORD_BOT_TOKEN
COHERE_API_KEY
WANDBOT_API_URL="http://localhost:8000"
WANDB_TRACING_ENABLED="true"
WANDB_PROJECT="wandbot-dev"
WANDB_ENTITY="wandbot"
```

Then build the app to install all dependencies in a virtual env.

```
bash build.sh
```

Start the Q&A bot application using the following commands:

```bash
bash run.sh
```

Then call the endpoint to trigger the final wandbot app initialisation:
```bash
curl http://localhost:8000/startup
```

For more detailed instructions on installing and running the bot, please refer to the [run.sh](./run.sh) file located in the root of the repository.

Executing these commands will launch the API, Slackbot, and Discord bot applications, enabling you to interact with the bot and ask questions related to the Weights & Biases documentation.

### Running the Evaluation pipeline

Make sure to set the environments in your terminal.

```
set -o allexport; source .env; set +o allexport
```

Launch the wandbot with 8 workers. This speeds up evaluation

```
WANDBOT_EVALUATION=1 gunicorn wandbot.api.app:app --bind 0.0.0.0:8000 --timeout=200 --workers=8 --worker-class uvicorn.workers.UvicornWorker
```

Set up for evaluation

wandbot/src/wandbot/evaluation/config.py
- `evaluation_strategy_name` : attribute name in Weave Evaluation dashboard
- `eval_dataset` : 
    - [Latest English evaluation dataset](https://wandb.ai/wandbot/wandbot-eval/weave/datasets?peekPath=%2Fwandbot%2Fwandbot-eval%2Fobjects%2Fwandbot_eval_data%2Fversions%2FeCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU%3F%26): "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU"
    - [Latest Japanese evaluation dataset](https://wandb.ai/wandbot/wandbot-eval-jp/weave/datasets?peekPath=%2Fwandbot%2Fwandbot-eval-jp%2Fobjects%2Fwandbot_eval_data_jp%2Fversions%2FoCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA%3F%26): "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA" 
- `eval_judge_model` : model used for judge
- `wandb_entity` : wandb entity name for record
- `wandb_project` : wandb project name for record

Launch W&B Weave evaluation
```
python src/wandbot/evaluation/weave_eval/main.py
```

### Data Ingestion

The data ingestion module pulls code and markdown from Weights & Biases repositories [docodile](https://github.com/wandb/docodile) and [examples](https://github.com/wandb/examples) ingests them into vectorstores for the retrieval augmented generation pipeline.
To ingest the data run the following command from the root of the repository
```bash
poetry run python -m src.wandbot.ingestion
```
You will notice that the data is ingested into the `data/cache` directory and stored in three different directories `raw_data`, `vectorstore` with individual files for each step of the ingestion process.
These datasets are also stored as wandb artifacts in the project defined in the environment variable `WANDB_PROJECT` and can be accessed from the [wandb dashboard](https://wandb.ai/wandb/wandbot-dev).
