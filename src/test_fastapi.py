import requests
import json

url = "http://localhost:8000/query/"
headers = {"Content-Type": "application/json"}

query_data = {
    "query": "How to best finetune an LLM with wandb?",
    "user_id": "anish"
}

response = requests.post(url, headers=headers, data=json.dumps(query_data))

if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.text)
