<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>

# WandBot

WandBot is a support assistant designed for Weights & Biases' Experiment Tracking and Weave.

## What's New

<details>
<summary>wandbot v1.3.0</summary>

- Up to date wandb docs + code, weave docs + code, example colabs, edu content - using chroma_index:v50
- Gemini flash-2.0 for query expansion (was gpt-4o)
- GPT-4o for composing the response (was gpt-4o)
- Cohere rerank-v3.5 (was rerank-v2.0)
- Hosted Chroma (was locally hosted chroma)
- Turned off web-search for now
- Moved all configs to configs folder
- Removed most langchain dependencies
- Implemented LLM and EmbeddingModel classes
- More robust evaluation pipeline, added retries, error handling and cli args
- Move package management to use uv
- Update to use python 3.12
- Add dotenv for env loading, while developing
- Add new endpoint, removed retriever endpoint for now
- Improved error handling and retries for all apis
</details>

<details>
<summary>wandbot v1.2.0</summary>

This release introduces a number of exciting updates and improvements:

- **Parallel LLM Calls**: Replaced the llama-index with the LECL, enabling parallel LLM calls for increased efficiency.
- **ChromaDB Integration**: Transitioned from FAISS to ChromaDB to leverage metadata filtering and speed.
- **Query Enhancer Optimization**: Improved the query enhancer to operate with a single LLM call.
- **Modular RAG Pipeline**: Split the RAG pipeline into three distinct modules: query enhancement, retrieval, and response synthesis, for improved clarity and maintenance.
- **Parent Document Retrieval**: Introduced parent document retrieval functionality within the retrieval module to enhance contextuality.
- **Sub-query Answering**: Added sub-query answering capabilities in the response synthesis module to handle complex queries more effectively.
- **API Restructuring**: Redesigned the API into separate routers for retrieval, database, and chat operations.

These updates are part of our ongoing commitment to improve performance and usability.
</details>

## Evaluation
English 
| wandbot version  | Comment  | Response Correctness | Num Trials | Data ingestion Report |
|---|---|---| --- | --- |
| 1.0.0 | baseline wandbot |  53.8 % | 1 |  |
| 1.1.0 | improvement over baseline; in production for the longest | 72.5 %  | 1 |  |
| 1.2.0 | our new enhanced wandbot | 81.6 % | 1 |  |
| 1.3.0rc | [1.3.0rc with gpt-4-preview judge](https://wandb.ai/wandbot/wandbot-eval/weave/evaluations?peekPath=%2Fwandbot%2Fwandbot-eval%2Fcalls%2F0196172b-bed6-77e3-8d43-dc1c31fc9a9b%3FhideTraceTree%3D1) | 71.3 % | 5 | [v50](https://wandb.ai/wandbot/wandbot-dev/reports/Prod-v1-3-Wandbot-Data-Ingestion-Report-2025-04-09-15-44-45--VmlldzoxMjIwNzI0Mg) |
| 1.3.0rc | [1.3.0rc with gpt-4o judge](https://wandb.ai/wandbot/wandbot-eval/weave/evaluations?peekPath=%2Fwandbot%2Fwandbot-eval%2Fcalls%2F019619b7-6ca1-7cc1-bdb9-d1053a6386d8%3FhideTraceTree%3D1) |88.8 % | 5 | [v50](https://wandb.ai/wandbot/wandbot-dev/reports/Prod-v1-3-Wandbot-Data-Ingestion-Report-2025-04-09-15-44-45--VmlldzoxMjIwNzI0Mg) |
| 1.3.0 | [v1.3.0 prod, v50 index, gpt-4o judge](https://wandb.ai/wandbot/wandbot-eval/weave/evaluations?peekPath=%2Fwandbot%2Fwandbot-eval%2Fcalls%2F01961c5e-9570-7f93-b3db-572ae83d9dbe%3FhideTraceTree%3D1) | 91.2 % | 5 | [v50](https://wandb.ai/wandbot/wandbot-dev/reports/Prod-v1-3-Wandbot-Data-Ingestion-Report-2025-04-09-15-44-45--VmlldzoxMjIwNzI0Mg)  |
| 1.3.1 | [v1.3.1 prod, v52 index, gpt-4o judge](https://wandb.ai/wandbot/wandbot-eval/weave/calls/01962210-44be-7f53-986d-4dc529660ad1?hideTraceTree=1) | 91.2 % | 5 | [v52](https://wandb.ai/wandbot/wandbot-dev/reports/Wandbot-Data-Ingestion-Report-for-chroma_index-v52-2025-04-10-23-28--VmlldzoxMjIzMDczNQ)
| 1.3.2 | [v1.3.2 prod, v54 index, gpt-4o judge. Knowledge base update](https://wandb.ai/wandbot/wandbot-eval/weave/evaluations?view=evaluations_2025-05-20_09-37-33-681&pin=%7B%22left%22%3A%5B%22summary.weave.trace_name%22%2C%22feedback%22%2C%22output.WandbotCorrectnessScorer.answer_correct.true_fraction%22%2C%22output.model_output.api_call_statuses.chat_error_info.has_error.true_count%22%2C%22output.model_output.api_call_statuses.chat_success.true_count%22%2C%22summary.weave.status%22%5D%2C%22right%22%3A%5B%5D%7D&peekPath=%2Fwandbot%2Fwandbot-eval%2Fcalls%2F0196ea1b-f8cb-7cf2-a22d-a8254ef6f67f%3FhideTraceTree%3D1) | 90.4 % | 5 | [v54](https://wandb.ai/wandbot/wandbot-dev/reports/Prod-v1-3-2-Wandbot-Data-Ingestion-Report-chroma_index-v54-2025-05-19-20-37--VmlldzoxMjg0ODMzNQ)




**Note**
- v1.3.1 uses:
  - claude Sonnet-3.7 for the response synthesizer, updated from gpt-4o-2024-11-20
  - an updated index that exludes korean and japanese versions of the docs as well as excludes the blog posts from Fully Connected.
- `1.3.0rc with gpt-4-preview judge` and `1.3.0rc with gpt-4o judge` are the same wandbot system evaluated with different judges. 
- The ~2.5% improvement between `1.3.0rc (gpt-4o judge)` and `1.3.0 prod` is mostly due to using `reranker-v3.5` (from 2.0) and `flash-2.0-001` (from gpt-4o). However evals previous to the v1.3.0 prod eval had 10-12 errors (out of 490 total calls), so there might be some noise in the results.

Japanese
| wandbot version  | Comment  | response accuracy |
|---|---|---|
| 1.2.0 | our new enhanced wandbot | 56.3 % |
| 1.2.1 | add translation process | 71.9 % |

## Features

- WandBot uses:
  - a hosted ChromaDB vector store
  - OpenAI's v3 embeddings
  - Gemini flash-2.0 for query enhancement
  - GPT-4o for response synthesis
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

Then build the app to install all dependencies in a virtual env, note this is heavily tailored for Replit.

```
bash build.sh
```

Start the Q&A bot application using the following commands:

```bash
bash run.sh
```

Then call the `/startup` endpoint to trigger the final wandbot app initialisation:
```bash
curl http://localhost:8000/startup
```

For more detailed instructions on installing and running the bot, please refer to the [run.sh](./run.sh) file located in the root of the repository.

Executing these commands will launch the API, Slackbot, and Discord bot applications, enabling you to interact with the bot and ask questions related to the Weights & Biases documentation.

### Running the Evaluation pipeline

**Eval Config**

The eval config can be found here and includes cli args to set the number of trials, weave parallelism, weave logging details and debug mode : `wandbot/src/wandbot/evaluation/eval_config.py`

The following evaluation sets are used:

[English evaluation dataset](https://wandb.ai/wandbot/wandbot-eval/weave/datasets?peekPath=%2Fwandbot%2Fwandbot-eval%2Fobjects%2Fwandbot_eval_data%2Fversions%2FeCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU%3F%26)
- ref: `weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU`
    
[Japanese evaluation dataset](https://wandb.ai/wandbot/wandbot-eval-jp/weave/datasets?peekPath=%2Fwandbot%2Fwandbot-eval-jp%2Fobjects%2Fwandbot_eval_data_jp%2Fversions%2FoCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA%3F%26)
- ref: `weave:///wandbot/wandbot-eval-jp/object/wandbot_eval_data_jp:oCWifIAtEVCkSjushP0bOEc5GnhsMUYXURwQznBeKLA`


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
WANDBOT_FULL_INIT=1 uv run uvicorn wandbot.api.app:app \
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

Launch W&B Weave evaluation in the root `wandbot` directory. Ensure that you're virtual envionment is active. By default, a sample will be evaluated 3 times in order to account for both the stochasticity of wandbot and our LLM judge. 

- For debugging, pass the `--debug` flag to only evaluate on a small number of samples. 
- To adjust the number of parallel evaluation calls weave makes use the `--n_weave_parallelism` flag when calling `eval.py` 
- see `eval_config.py` for all evaluation options.

```
source wandbot_venv/bin/activate

uv run src/wandbot/evaluation/eval.py
```

When running evals before prod, we always run 5 trials per eval sample. It can also be a good idea to reduce weave parallelism in order to avoid api rate limiting issues which might skew eval results:

```
caffeinate uv run src/wandbot/evaluation/eval.py --experiment_name <insert eval name> --n_trials 5 --n_weave_parallelism 4
```

Debugging, only running evals on 1 sample and for 1 trial:

```
uv run src/wandbot/evaluation/eval.py  --debug --n_debug_samples=1 --n_trials=1
```

Evaluate on Japanese dataset:

```
uv run src/wandbot/evaluation/eval.py  --lang ja
```

To only evaluate each sample once:

```
uv run src/wandbot/evaluation/eval.py  --n_trials 1
```


### Data Ingestion

The data ingestion module pulls code and markdown from Weights & Biases repositories [docodile](https://github.com/wandb/docodile) and [examples](https://github.com/wandb/examples) ingests them into vectorstores for the retrieval augmented generation pipeline.

To ingest the data run the following command from the root of the repository, see `run_ingestion_config.py` for all available arguments.

```bash
uv run src/wandbot/ingestion/__main__.py
```

**Note:**

Pay special attention to the configs in `src/wandbot/configs/vector_store_config.py` and `src/wandbot/configs/ingestion_config` as this is where important settings such as the embedding model, embedding dimensions and hosted vs local vector db are set.

You will notice that the data is ingested into the `data/cache` directory and stored in three different directories `raw_data`, `vectorstore` with individual files for each step of the ingestion process.

These datasets are also stored as wandb artifacts in the project defined in the environment variable `WANDB_PROJECT` and can be accessed from the [wandb dashboard](https://wandb.ai/wandb/wandbot-dev).

### Evaluating a file with Precomputed Answers 

Instead of hitting the wandbot endpoint, you can also pass a `.json` file of precomputed answers for evaluation by the `WandbotCorrectnessScorer`. To do so, pass a filepath to the `precomputed_answers_json_path` parameter in the `EvalConfig`), it should be a JSON file containing a list of objects. Each object should have a question and a precomputed answer.

The evaluation system will try to match questions from your evaluation dataset (defined by `eval_dataset` in `EvalConfig`) to the `question` field in these objects using exact string matching (after stripping leading/trailing whitespace).

Each object in the JSON list should have the following structure:

**Required Fields:**

*   `question` (string): The question text. This is used to match against questions in the evaluation dataset.
*   `generated_answer` (string): The precomputed answer text. This will be used as `EvalChatResponse.answer`.

**Fields for Contextual Scoring:**
To enable context-based scoring (e.g., by `WandbotCorrectnessScorer`), you can provide the context information through one of the following fields in each JSON object, although its not essential:

*   `retrieved_contexts` (List of Dicts): A list of context documents. Each dictionary in the list should represent a document and ideally have `"source"` (string, URL) and `"content"` (string, text of the document) keys. Minimally, a `"content"` key is needed for the scorer.
    *Example*: `[{"source": "http://example.com/doc1", "content": "Context snippet 1."}, {"content": "Context snippet 2."}]`
*   **OR** `source_documents` (string): A raw string representation of source documents that can be parsed by the system (specifically, by the `parse_text_to_json` function in `eval.py`). This string usually contains multiple documents, each prefixed by something like "source: http://...".

If only `source_documents` (string) is provided and `retrieved_contexts` (list) is not, the system will attempt to parse `source_documents` to populate the `retrieved_contexts` field for the `EvalChatResponse`. If neither is provided, context-based scoring for that precomputed answer will operate with empty context.

**Optional Fields (to fully populate `EvalChatResponse` and mimic live API calls):**

*   `system_prompt` (string): The system prompt used.
*   `sources` (string): A string listing sources (can be similar to `source_documents` or a different format).
*   `model` (string): The name of the model that generated the answer (e.g., "precomputed_gpt-4").
*   `total_tokens` (int): Total tokens used.
*   `prompt_tokens` (int): Prompt tokens used.
*   `completion_tokens` (int): Completion tokens used.
*   `time_taken` (float): Time taken for the call.
*   `api_call_statuses` (dict): Dictionary of API call statuses.
*   `start_time` (string): ISO 8601 formatted start time of the call (e.g., `"2023-10-27T10:00:00Z"`).
*   `end_time` (string): ISO 8601 formatted end time of the call.
*   `has_error` (boolean): Set to `true` if this precomputed item represents an error response.
*   `error_message` (string): The error message if `has_error` is `true`.

**Example Object:**
```json
{
  "question": "What is Weights & Biases?",
  "generated_answer": "Weights & Biases is an MLOps platform.",
  "retrieved_contexts": [
      {"source": "http://example.com/docA", "content": "Content of document A talking about W&B."},
      {"content": "Another piece of context."}
  ],
  "model": "precomputed_from_file",
  "system_prompt": "You are a helpful assistant.",
  "total_tokens": 50,
  "has_error": false
}
```
Or using `source_documents`:
```json
{
  "question": "What is Weights & Biases?",
  "generated_answer": "Weights & Biases is an MLOps platform.",
  "source_documents": "source: http://example.com/docA\\nContent of document A talking about W&B.\\nsource: http://example.com/docB\\nContent of document B.",
  "model": "precomputed_from_file",
  "system_prompt": "You are a helpful assistant.",
  "total_tokens": 50,
  "has_error": false
}
```


### Ingestion pipeline debugging

To help with debugging, you can use the `steps` and `include_sources` flags to specify only sub-components of the pipeline and only certain documents sources to run. For example if you wanted to stop the pipeline before it creates the vector db and creates the artifacts and W&B report AND you only wanted to process the Weave documentation, you would do the following:

```
uv run src/wandbot/ingestion/__main__.py --steps prepare preprocess --include_sources "weave_documentation" --debug
```

#### Note on updating hosted Chroma vector db

A. If you compute a diff between the old dev docs and the new ones
1. You could use delete() then add(), on the same ids if you have consistent ids across updates
2. You could call update() or upsert() on the same ids, but if you changed any metadata schemas and want to drop old keys, you'll have to explicitly do that.

B. If you don't compute a diff or want a simple way to do this
1. You could delete everything in the collection and add it
2. You could create a new collection and insert the new data into that.


## Release Checklist

[] **Evaluation:** Run evaluations, ensure no performance drop or justify why drop is acceptable/expected
[] **Evaluation:** Prefix eval name in Weave evals with "Prod vX.X.X -" for easy identification (in wandbot/wandbot-eval)
[] **Knowlege Update:** If knowlege base is updated then update the ingestion report with "Prod vX.X.X -" for easy identification (in wandbot/wandbot-dev)
[] **Deployment Testing:** Clone the repo to a staging env (e.g. test Replit app). Install and run via shell, sure no issues, ping local endpoint and ensure no errors. 
[] **Deployment Testing:** Then create a staged deployment and again ensure no errors. Ping staged endpoint to ensure correct response is received.
[] **Deployment:** Clone the repo to a prod environment. Deploy updated version. Test via cli and slackbot that the endpoint is working and the correctg response is received.
[] [] **GitHub:** Update evaluation table at top of README with latest eval score, weave Eval link and data ingestion Report link
[] **GitHub:** Update git tag
[] **GitHub:** Create gthub release