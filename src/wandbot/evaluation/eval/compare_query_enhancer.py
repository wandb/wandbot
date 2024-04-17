import wandb
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from wandbot.chat.query_enhancer import QueryHandler
from wandbot.query_handler.query_enhancer import load_query_enhancement_chain
from wandbot.utils import get_logger
from wandbot.chat.config import ChatConfig
from wandbot.chat.schemas import ChatRequest

logger = get_logger(__name__)

df = pd.read_json(
    "data/eval/wandbot_cleaned_annotated_dataset_11-12-2023.jsonl",
    lines=True,
    orient="records",
)
correct_df = df[
    (df["is_wandb_query"] == "YES") & (df["correctness"] == "correct")
]

query_enhancer_v1 = QueryHandler()

config = ChatConfig()
llm = ChatOpenAI(model=config.chat_model_name, temperature=0)
lang_detect_path = "data/cache/models/lid.176.bin"
query_enhancer_v2 = load_query_enhancement_chain(llm, lang_detect_path)

df = pd.DataFrame(
    columns=[
        "question",
        "qe1_condensed_query",
        "qe2_question",
        "qe1_intent",
        "qe2_intent",
        "qe1_keywords",
        "qe2_keywords",
        "qe1_subqueries",
        "qe2_vector_search",
        "qe2_web_answer",
    ]
)

project = "wandbot-eval"
entity = "wandbot"

run = wandb.init(project=project, entity=entity)

for idx, row in tqdm(correct_df.iterrows()):
    query = row["question"]
    chat_request = ChatRequest(
        question=query,
        chat_history=[],
        language="en",
        application="slack",
    )
    complete_query = query_enhancer_v1(chat_request)
    condensed_query = complete_query.condensed_query
    intent = complete_query.intent_hints
    keywords = ", ".join(complete_query.keywords)
    subqueries = "; ".join(complete_query.sub_queries)

    #### 

    result = query_enhancer_v2.invoke({"query": query, "chat_history": []})

    try:
        enhanced_question = result.get('standalone_question')
        enhanced_intent = result.get('intents').get('intent_hints') + "\n" + ", ".join(result.get('intents').get('intent_labels'))
        enhanced_keywords = ", ".join(result['keywords'])  # Join list of keywords into a single string
        vector_search = result['vector_search']
        web_answer = result['web_results']['web_answer']
    except:
        enhanced_question = None
        enhanced_intent = None
        enhanced_keywords = None
        vector_search = None
        web_answer = None

    data_dict = {
        "question": query,
        "qe1_condensed_query": condensed_query,
        "qe2_question": enhanced_question,
        "qe1_intent": intent,
        "qe2_intent": enhanced_intent,
        "qe1_keywords": keywords,
        "qe2_keywords": enhanced_keywords,
        "qe1_subqueries": subqueries,
        "qe2_vector_search": vector_search,
        "qe2_web_answer": web_answer,
    }

    with open("data/eval/compare_query_enhancer.jsonl", "w+") as outfile:
        _data_dict = json.dumps(data_dict)
        outfile.write(_data_dict + "\n")

    new_row = pd.DataFrame([data_dict])
    df = pd.concat([df, new_row], ignore_index=True)

run.log({"Compare Query Enhancers Results": df})
