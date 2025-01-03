import weave
import json
import requests
import os
from typing import List, Dict, Any
from tqdm import tqdm

dataset_ref = weave.ref(
    "weave:///wandbot/wandbot-eval/object/wandbot_eval_data:eCQQ0GjM077wi4ykTWYhLPRpuGIaXbMwUGEB7IyHlFU"
).get()
question_rows = dataset_ref.rows
question_rows = [
    {
        "question": row["question"],
        "ground_truth": row["answer"],
        "notes": row["notes"],
        "context": row["context"],
        "correctness": row["correctness"],
        "is_wandb_query": row["is_wandb_query"],
    }
    for row in question_rows
]


def translate_with_openai(text: str) -> str:
    # Get the OpenAI API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Set headers for the OpenAI API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Data payload for GPT-4-turbo (gpt-4o) API request

    # 質問には答えないでというプロンプトを入れても良いかもしれない

    data = {
        "model": "gpt-4o-2024-08-06",  # Updated to GPT-4 Turbo
        "max_tokens": 4000,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional translator. \n\n\
                    Translate the user's text into Japanese according to the specified rules. \n\
                    Rule of translation. \n\
                    - Maintain the original nuance\n\
                    - Use 'run' in English where appropriate, as it's a term used in Wandb.\n\
                    - Translate the terms 'reference artifacts' and 'lineage' into Katakana. \n\
                    - Include specific terms in English or Katakana where appropriate\n\
                    - Keep code unchanged.\n\
                    - Keep URL starting from 'Source:\thttps:', but translate texts after 'Source:\thttps:'\n\
                    - Only return the Japanese translation without any additional explanation",
            },
            {"role": "user", "content": text},
        ],
    }

    # Make the API request to OpenAI
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=data
    )

    # Check if the request was successful
    if response.status_code == 200:
        # Return the translated text
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )


def translate_data(data: List[Dict[str, Any]], output_file: str) -> None:
    total_items = len(data)

    # Check if the file exists and get the last processed index
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            processed_data = json.load(file)
        start_index = len(processed_data)
    else:
        processed_data = []
        start_index = 0

    for i in tqdm(
        range(start_index, total_items), initial=start_index, total=total_items
    ):
        item = data[i]
        translated_item = item.copy()
        for key in ["question", "ground_truth", "notes", "context"]:
            if key in item:
                translated_item[key] = translate_with_openai(item[key])

        processed_data.append(translated_item)

        # Save progress after each item
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(processed_data, file, ensure_ascii=False, indent=2)

    print(f"Translation completed. Results saved in '{output_file}'")


output_file = "translated_data.json"
translate_data(question_rows, output_file)
