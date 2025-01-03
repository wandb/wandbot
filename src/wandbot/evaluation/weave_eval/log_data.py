import os

os.environ["WANDB_ENTITY"] = "wandbot"

import wandb
import weave
import pandas as pd
from weave import Dataset

from wandbot.evaluation.config import EvalConfig

config = EvalConfig()

wandb_project = config.wandb_project
wandb_entity = config.wandb_entity

eval_artifact = wandb.Api().artifact(config.eval_artifact)
eval_artifact_dir = eval_artifact.download(root=config.eval_artifact_root)

df = pd.read_json(
    f"{eval_artifact_dir}/{config.eval_annotations_file}",
    lines=True,
    orient="records",
)
df.insert(0, "id", df.index)

correct_df = df[
    (df["is_wandb_query"] == "YES") & (df["correctness"] == "correct")
]

data_rows = correct_df.to_dict("records")

weave.init(wandb_project)

# Create a dataset
dataset = Dataset(
    name="wandbot_eval_data",
    rows=data_rows,
)

# Publish the dataset
weave.publish(dataset)
