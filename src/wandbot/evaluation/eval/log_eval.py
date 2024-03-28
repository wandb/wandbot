import os
import wandb
import json
import argparse
import pandas as pd
from pathlib import Path

from wandbot.utils import load_config


def main(config_path: str, eval_result_path: str) -> None:
    config = load_config(config_path)

    project = "wandbot-eval"
    entity = "wandbot"

    run = wandb.init(project=project, entity=entity, config=config.model_dump())
    
    eval_df = pd.read_json(eval_result_path, lines=True)
    print("Number of eval samples: ", len(eval_df))
    run.log({"Evaluation Results": eval_df})

    score_columns = [col for col in eval_df.columns if col.endswith('_score')]
    mean_scores = eval_df[score_columns].mean()
    mode_scores = eval_df[score_columns].mode()
    percent_grade3 = (eval_df[score_columns] == 3).mean()
    percent_grade2 = (eval_df[score_columns] == 2).mean()
    percent_grade1 = (eval_df[score_columns] == 1).mean()

    # Select columns ending with "_result" and calculate the percentage of True values
    result_columns = [col for col in eval_df.columns if col.endswith('_result')]
    percentage_true_results = (eval_df[result_columns].sum() / eval_df[result_columns].count())

    final_eval_results = {}
    final_eval_results.update(mean_scores.to_dict())
    final_eval_results.update(mode_scores.iloc[0].to_dict())
    final_eval_results.update(percent_grade3.to_dict())
    final_eval_results.update(percent_grade2.to_dict())
    final_eval_results.update(percent_grade1.to_dict())
    final_eval_results.update(percentage_true_results.to_dict())

    print("Final Eval Results")
    print(final_eval_results)

    run.log(final_eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run ingestion process with a specified configuration.'
    )
    parser.add_argument(
        '--config', 
        type=str,
        required=True,
        help='Path to the configuration YAML file.'
    )
    parser.add_argument(
        '--eval_result',
        type=str,
        required=True,
        help='Path to the evaluation result JSONL file.'
    )
    args = parser.parse_args()
    main(args.config, args.eval_result)
