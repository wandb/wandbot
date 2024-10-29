import json
import weave
from weave import Dataset

def rename_key(item):
    if 'ground_truth' in item:
        item['answer'] = item.pop('ground_truth')
    return item

def create_test_file(json_file_path, test_file_path, num_lines=5):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    test_data = data[:num_lines]
    
    with open(test_file_path, 'w') as file:
        json.dump(test_data, file, indent=2, ensure_ascii=False)
    
    print(f"Test file with {num_lines} lines has been created at {test_file_path}")

def publish_json_to_weave(json_file_path, dataset_name, project_name):
    # Initialize Weave
    weave.init(project_name)

    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Rename 'ground_truth' to 'answer' in each item
    processed_data = [rename_key(item) for item in data]

    # Create a dataset
    dataset = Dataset(name=dataset_name, rows=processed_data)

    # Publish the dataset
    weave.publish(dataset)

    print(f"Dataset '{dataset_name}' has been published to project '{project_name}'.")

# Usage example
json_file_path = 'translated_data.json'
test_file_path = 'test_translated_data.json'
dataset_name = 'wandbot_eval_data_jp'
project_name = 'wandbot/wandbot-eval-jp'

# Create test file
#create_test_file(json_file_path, test_file_path)

# Publish full dataset to Weave
publish_json_to_weave(json_file_path, dataset_name, project_name)