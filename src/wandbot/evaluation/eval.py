import json
import time
from pathlib import Path

import pandas as pd
import wandb
from fuzzywuzzy import fuzz
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from tqdm.auto import tqdm
from wandbot.chat.chat import Chat
from wandbot.chat.prompts import load_eval_prompt
from wandbot.evaluation.config import EvalConfig

eval_config = EvalConfig()


# Similarity score as the distance of the most similar substring as a number between 0 and 100
def calculate_string_distance(row):
    return fuzz.partial_ratio(row.orig_response, row.response)


class Evaluate:
    config: EvalConfig = EvalConfig()

    def __init__(self, config: EvalConfig = None):
        if config is not None:
            self.config = config
        self.chat = Chat(self.config.chat_config)
        self.eval_df = pd.DataFrame(
            columns=[
                "query",
                "orig_response",
                "orig_document",
                "response",
                "documents",
                "scores",
            ]
        )
        self.wandb_run = wandb.init(
            entity=eval_config.wandb_entity,
            project=eval_config.wandb_project,
            job_type=eval_config.wandb_job_type,
            config=eval_config.dict(),
        )

    # download eval data from artifact
    def load_eval_dataframe(self):
        eval_artifact = self.wandb_run.use_artifact(self.config.eval_artifact)
        eval_artifact_dir = Path(eval_artifact.download())
        df_questions = pd.read_csv(eval_artifact_dir / "auto-eval-questions.csv")
        if self.config.debug:
            df_questions = df_questions.sample(n=3).reset_index(drop=True)
        return df_questions

    def __call__(self, config: EvalConfig = None):
        if config is not None:
            self.config = config
        eval_dataframe = self.load_eval_dataframe()

        for i in tqdm(range(len(eval_dataframe))):
            query = eval_dataframe["question"].loc[i]
            orig_response = eval_dataframe["response"].loc[i]
            orig_document = eval_dataframe["document"].loc[i]
            for i in range(self.config.max_retries):
                try:
                    resp = self.chat(query)
                    response = resp.answer
                    document_scores = json.loads(resp.source_documents)
                    documents = [x["document"] for x in document_scores]
                    scores = [x["score"] for x in document_scores]
                    eval_df = self.eval_df.append(
                        {
                            "query": query,
                            "orig_response": orig_response,
                            "orig_document": orig_document,
                            "response": response,
                            "documents": documents,
                            "scores": scores,
                        },
                        ignore_index=True,
                    )
                    break
                except Exception as e:
                    print(
                        f"Error occurred: {e}. Retrying in {self.config.retry_delay} seconds..."
                    )
                    time.sleep(self.config.retry_delay)

        self.eval_df["retrieval_match"] = self.eval_df.apply(
            lambda x: x.orig_document in x.documents, axis=1
        )
        self.eval_df = self.eval_df.dropna()
        self.eval_df["string_distance"] = self.eval_df.apply(
            calculate_string_distance, axis=1
        )

        eval_df.orig_document = eval_df.orig_document.apply(
            lambda x: x.replace("../docodile/docs/", "https://docs.wandb.ai/").replace(
                ".md", ""
            )
        )

        retrieval_accuracy = len(
            self.eval_df[self.eval_df["retrieval_match"] == True]
        ) / len(self.eval_df)
        print(f"Retrieval accuracy: {retrieval_accuracy}")
        wandb.log({"retrieval_accuracy": retrieval_accuracy})

        average_string_distance = self.eval_df["string_distance"].mean()
        print(f"Average string distance: {average_string_distance}")
        wandb.log({"average_string_distance": average_string_distance})

        print("Loading evaluation model...")

        chat_prompt = load_eval_prompt(self.config.eval_prompt)
        llm = ChatOpenAI(
            model_name=self.config.eval_model,
            temperature=0,
        )
        eval_chain = QAEvalChain.from_llm(llm, prompt=chat_prompt)

        examples = []
        predictions = []
        for i in range(len(self.eval_df)):
            # print(f"Example {i}:")
            examples.append(
                {
                    "query": self.eval_df["query"].iloc[i],
                    "answer": self.eval_df["orig_response"].iloc[i],
                }
            )
            predictions.append(
                {
                    "query": self.eval_df["query"].iloc[i],
                    "answer": self.eval_df["orig_response"].iloc[i],
                    "result": self.eval_df["response"].iloc[i],
                }
            )
        graded_outputs = eval_chain.evaluate(examples, predictions)
        self.eval_df["model_score"] = [x.get("text", "None") for x in graded_outputs]

        model_accuracy = len(
            self.eval_df[self.eval_df["model_score"] == "CORRECT"]
        ) / len(self.eval_df)
        print(f"Chat model accuracy: {model_accuracy}")
        wandb.log({"chat_accuracy": model_accuracy})

        self.eval_df.to_parquet("eval_results.parquet")

        artifact = wandb.Artifact("eval_results", type="eval_results")
        artifact.add_file("eval_results.parquet")
        self.wandb_run.log_artifact(artifact)

        self.wandb_run.log(
            {
                "eval_results": wandb.Table(
                    dataframe=self.eval_df.astype(
                        {
                            "query": "string",
                            "orig_response": "string",
                            "orig_document": "string",
                            "response": "string",
                            "documents": "string",
                            "scores": "string",
                        }
                    )
                )
            }
        )

        self.wandb_run.log_code()

        self.wandb_run.finish()
