human_template = """You are an evaluator for the W&B chatbot.
You are given a question, the chatbot's answer, and the original answer, and are asked to score the chatbot's answer as either CORRECT or INCORRECT.
Note that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the best answer. 
You are evaluating the chatbot's answer only. 
Example Format:
QUESTION: question here
CHATBOT ANSWER: student's answer here
ORIGINAL ANSWER: original answer here
GRADE: CORRECT or INCORRECT here
Please remember to grade them based on being factually accurate. Begin!
QUESTION: {query}
CHATBOT ANSWER: {result}
ORIGINAL ANSWER: {answer}
GRADE:"""

cfg = {
    'DEBUG': True,
    'CHAT_MODEL_NAME': "gpt-3.5-turbo",
    'EVAL_MODEL_NAME': "gpt-3.5-turbo",
    'EVAL_ARTIFACT': 'wandbot/wandbbot/eval_dataset:v0',
    'human_template': human_template,
    'system_template': "You are a helpful assistant.",
    'max_retries': 3,
    'retry_delay': 10  # seconds,
}

from chat import Chat
import wandb
import pandas as pd
from tqdm.auto import tqdm
import os
from fuzzywuzzy import fuzz
import time
from pathlib import Path

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain

from config import default_config, TEAM, PROJECT

cfg.update(vars(default_config))

# Similarity score as the distance of the most similar substring as a number between 0 and 100
def calculate_string_distance(row):
    return fuzz.partial_ratio(row.orig_response, row.response)

wandb_run = wandb.init(entity=TEAM, project=PROJECT, job_type="eval", config=cfg)

cfg = wandb.config

# download eval data from artifact
eval_artifact = wandb_run.use_artifact(cfg.EVAL_ARTIFACT)
eval_artifact_dir = Path(eval_artifact.download())
df_questions = pd.read_csv(eval_artifact_dir/'auto-eval-questions.csv')

print("Loading chat model...")
chat = Chat(model_name=cfg.CHAT_MODEL_NAME, wandb_run=wandb_run)
eval_df = pd.DataFrame(columns = ['query', 'orig_response', 'orig_document', 'response', 'documents', 'scores'])

if cfg.DEBUG: df_questions = df_questions.sample(n=3).reset_index(drop=True)

for i in tqdm(range(len(df_questions))):
    query = df_questions['question'].loc[i]
    orig_response = df_questions['response'].loc[i]
    orig_document = df_questions['document'].loc[i]
    for i in range(cfg.max_retries):
        try:
            resp = chat(query, sources = True)
            response = resp[1]
            documents = [x.metadata['source'] for x in resp[3]]
            scores = [x.metadata['score'] for x in resp[3]]
            eval_df = eval_df.append({'query': query, 'orig_response': orig_response, 'orig_document': orig_document, 'response': response, 'documents': documents, 'scores': scores}, ignore_index=True)
            break
        except Exception as e:
            print(f'Error occurred: {e}. Retrying in {cfg.retry_delay} seconds...')
            time.sleep(cfg.retry_delay)
            
eval_df.orig_document = eval_df.orig_document.apply(lambda x: x.replace('../docodile/docs/', 'https://docs.wandb.ai/').replace('.md', ''))
eval_df['retrieval_match'] = eval_df.apply(lambda x: x.orig_document in x.documents, axis=1)
eval_df = eval_df.dropna()
eval_df['string_distance'] = eval_df.apply(calculate_string_distance, axis=1)

retrieval_accuracy = len(eval_df[eval_df['retrieval_match'] == True]) / len(eval_df)
print(f"Retrieval accuracy: {retrieval_accuracy}")
wandb.log({'retrieval_accuracy': retrieval_accuracy})

average_string_distance = eval_df['string_distance'].mean()
print(f"Average string distance: {average_string_distance}")
wandb.log({'average_string_distance': average_string_distance})

print("Loading evaluation model...")
system_message_prompt = SystemMessagePromptTemplate.from_template(cfg.system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(cfg.human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
llm = ChatOpenAI(
            model_name=cfg.EVAL_MODEL_NAME,
            temperature=0,
        )
eval_chain = QAEvalChain.from_llm(llm)

examples = []
predictions = []
for i in range(len(eval_df)):
    # print(f"Example {i}:")
    examples.append({
        'query': eval_df['query'].iloc[i],
        'answer': eval_df['orig_response'].iloc[i],
    })
    predictions.append({
        'query': eval_df['query'].iloc[i],
        'answer': eval_df['orig_response'].iloc[i],
        'result': eval_df['response'].iloc[i],
    })
graded_outputs = eval_chain.evaluate(examples, predictions)
eval_df['model_score'] = [x.get('text', 'None') for x in graded_outputs]

model_accuracy = len(eval_df[eval_df['model_score'] == 'CORRECT']) / len(eval_df)
print(f"Chat model accuracy: {model_accuracy}")
wandb.log({'chat_accuracy': model_accuracy})

eval_df.to_parquet('eval_results.parquet')

artifact = wandb.Artifact('eval_results', type='eval_results')
artifact.add_file('eval_results.parquet')
wandb.log_artifact(artifact)

wandb.log({'eval_results': wandb.Table(dataframe=eval_df.astype({'query': 'string', 'orig_response': 'string', 'orig_document': 'string', 'response': 'string', 'documents': 'string', 'scores': 'string'}))})

wandb_run.log_code()

wandb.finish()