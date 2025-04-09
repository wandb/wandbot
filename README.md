# wandbot

WandBot is a question-answering bot designed specifically for Weights & Biases Models and Weave [documentation](https://docs.wandb.ai/).

## What's New

### wandbot v1.3.0
**New:**

- **Move to uv for package management**: Installs and dependency checks cut down from minutes to seconds
- **Support python 3.12 on replit**
- **Move to lazing loading in app.py to help with startup**: Replit app deployments can't seen to handle the delay from loading the app, despite attempting async or background tasks
- **Add wandb artifacts cache cleanup**: Saved 1.2GB of disk space
- **Turn off web search**: Currently we don't have a web search provider to use.
- **Refactored EvalConfig and evals script**: Switched config to using simple_parsing for free cli arguments. Added n_trials, debug mode. Undid hardcoding of ja weave eval dataset.
- **Removed langchain-cohere**: Started hitting dependency errors, removed it in favor of raw cohere client.
- **wandb Tables Feedback logging disabled in prep for Weave feedback**
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

The project is built with Python version `3.12` and utilizes `uv` for dependency management. Follow the steps below to install the necessary dependencies:

```bash
bash build.sh
```

## Usage

### Running WandBot

Before running the Q&A bot, ensure the following environment variables are set:

```bash
OPENAI_API_KEY
COHERE_API_KEY
WANDB_API_KEY
WANDBOT_API_URL="http://localhost:8000"
WANDB_TRACING_ENABLED="true"
LOG_LEVEL=INFO
WANDB_PROJECT="wandbot-dev"
WANDB_ENTITY= <your W&B entity>

```

If you're running the slack or discord apps you'll also need the following keys/tokens set as env vars:

```
SLACK_EN_APP_TOKEN
SLACK_EN_BOT_TOKEN
SLACK_EN_SIGNING_SECRET
SLACK_JA_APP_TOKEN
SLACK_JA_BOT_TOKEN
SLACK_JA_SIGNING_SECRET
DISCORD_BOT_TOKEN
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

**Eval Config**

Modify the evaluation config file here: `wandbot/src/wandbot/evaluation/config.py`

`evaluation_strategy_name` : attribute name in Weave Evaluation dashboard
`eval_dataset` : 
    - [Latest English evaluation dataset](https://wandb.ai/wandbot/wandbot-eval/weave/datasets?peekPath=%2Fwandbot%2Fwandbot-eval%2Fobjects%2Fwandbot_eval_data%2Fversions%2FeCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU%3F%26): "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU"
    - [Latest Japanese evaluation dataset](https://wandb.ai/wandbot/wandbot-eval-jp/weave/datasets?peekPath=%2Fwandbot%2Fwandbot-eval-jp%2Fobjects%2Fwandbot_eval_data_jp%2Fversions%2FoCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA%3F%26): "weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA" 
`eval_judge_model` : model used for judge
`wandb_entity` : wandb entity name for record
`wandb_project` : wandb project name for record

**Dependencies**

Ensure wandbot is installed by installing the production depenencies, activate the virtual env that was created and then install the evaluation dependencies

```
bash build.sh
source wandbot_venv/bin/activate
uv pip install -r eval_requirements.txt
poetry install
```

**Environment variables**

Make sure to set the environment variables (i.e. LLM provider keys etc) from the `.env` file.

**Launch the wandbot app**
You can either use `uvicorn` or `gunicorn` to launch N workers to be able to serve eval requests in parallel. Note that weave Evaluations also have a limit on the number of parallel calls make, set via the `WEAVE_PARALLELISM` env variable, which is set further down in the `eval.py` file using the `n_weave_parallelism` flag. Launch wandbot with 8 workers for faster evaluation. The `WANDBOT_FULL_INIT` env var triggers the full wandbot app initialization.

`uvicorn`
```bash
WANDBOT_FULL_INIT=1 uvicorn wandbot.api.app:app \
--host 0.0.0.0 \
--port 8000 \
--workers 8 \
--timeout-keep-alive 75 \
--loop uvloop \
--http httptools
```

Testing: You can test that the app is running correctly by making a request to the `chat/query` endpoint, you should receive a response payload back from wandbot after 30 - 90 seconds:

```bash
curl -X POST \
   http://localhost:8000/chat/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I log a W&B artifact?"}'
```

**Debugging**
For debugging purposes during evaluation you can run a single instance of the app by chaning the `uvicorn` command above to use `--workers 1` 
```

**Run the evaluation**

Launch W&B Weave evaluation in the root `wandbot` directory. Ensure that you're virtual envionment is active. By default, a sample will be evaluated 3 times in order to account for both the stochasticity of wandbot and our LLM judge. For debugging, pass the `--debug` flag to only evaluate on a small number of samples. To adjust the number of parallel evaluation calls weave makes use the `--n_weave_parallelism` flag when calling `eval.py` 

```
source wandbot_venv/bin/activate

python src/wandbot/evaluation/eval.py
```

Debugging, only running evals on 1 sample and for 1 trial:

```
python src/wandbot/evaluation/eval.py  --debug --n_debug_samples=1 --n_trials=1
```

Evaluate on Japanese dataset:

```
python src/wandbot/evaluation/eval.py  --lang ja
```

To only evaluate each sample once:

```
python src/wandbot/evaluation/eval.py  --n_trials 1
```


### Data Ingestion

The data ingestion module pulls code and markdown from Weights & Biases repositories [docodile](https://github.com/wandb/docodile) and [examples](https://github.com/wandb/examples) ingests them into vectorstores for the retrieval augmented generation pipeline.

To ingest the data run the following command from the root of the repository

```bash
python -m wandbot.ingestion
```

**Note:**

Pay special attention to the configs in `src/wandbot/configs/vector_store_config.py` and `src/wandbot/configs/ingestion_config` as this is where important settings such as the embedding model, embedding dimensions and hosted vs local vector db are set.

You will notice that the data is ingested into the `data/cache` directory and stored in three different directories `raw_data`, `vectorstore` with individual files for each step of the ingestion process.

These datasets are also stored as wandb artifacts in the project defined in the environment variable `WANDB_PROJECT` and can be accessed from the [wandb dashboard](https://wandb.ai/wandb/wandbot-dev).

#### Ingestion pipeline debugging

To help with debugging, you can use the `steps` and `include_sources` flags to specify only sub-components of the pipeline and only certain documents sources to run. For example if you wanted to stop the pipeline before it creates the vector db and creates the artifacts and W&B report AND you only wanted to process the Weave documentation, you would do the following:

```
python -m wandbot.ingestion --steps prepare preprocess --include_sources "weave_documentation" --debug
```
