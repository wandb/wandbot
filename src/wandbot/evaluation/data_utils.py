import json
import logging
from typing import Dict, List, Optional, Any

import weave

# From eval.py
def sanitize_precomputed_item_recursive(item: Any) -> Any:
    """Recursively sanitize an item by converting None values to empty strings."""
    if isinstance(item, dict):
        return {k: sanitize_precomputed_item_recursive(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [sanitize_precomputed_item_recursive(elem) for elem in item]
    elif item is None:
        return ""
    return item


def load_and_prepare_precomputed_data(
    file_path: Optional[str], logger: logging.Logger
) -> Optional[Dict[str, Dict]]:
    """Loads, sanitizes, and prepares precomputed answers from a JSON file into a map."""
    if not file_path:
        return None

    logger.info(f"Loading precomputed answers from: {file_path}")
    try:
        with open(file_path, "r") as f:
            loaded_answers_raw = json.load(f)

        if not isinstance(loaded_answers_raw, list):
            raise ValueError("Precomputed answers JSON must be a list of items.")

        loaded_answers_sanitized = []
        for raw_item in loaded_answers_raw:
            if not isinstance(raw_item, dict):
                raise ValueError(f"Skipping non-dictionary item in precomputed answers: {raw_item}")
            sanitized_item = sanitize_precomputed_item_recursive(raw_item)
            loaded_answers_sanitized.append(sanitized_item)
            logger.debug(f"Sanitized precomputed item: {sanitized_item}")

        precomputed_answers_map = {}
        for i, item in enumerate(loaded_answers_sanitized):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Item at original index {i} in precomputed answers (post-sanitization) is not a dictionary."
                )
            
            item_index_str: str
            raw_item_index = item.get("index")

            if raw_item_index is None or str(raw_item_index).strip() == "":
                logger.warning(
                    f"Item (original index {i}) is missing 'index' or index is empty after sanitization. "
                    f"Content: {str(item.get('question', 'N/A'))[:50]+'...'}. Using list index {i} as fallback string key."
                )
                item_index_str = str(i)
            else:
                item_index_str = str(raw_item_index).strip()
                if not item_index_str:
                    logger.warning(
                        f"Item (original index {i}) had whitespace-only 'index' after sanitization. "
                        f"Content: {str(item.get('question', 'N/A'))[:50]+'...'}. Using list index {i} as fallback string key."
                    )
                    item_index_str = str(i)
            
            if item_index_str in precomputed_answers_map:
                logger.warning(
                    f"Duplicate string index '{item_index_str}' found in precomputed answers. "
                    f"Overwriting with item from original list at index {i}."
                )
            precomputed_answers_map[item_index_str] = item
        
        logger.info(
            f"Loaded {len(precomputed_answers_map)} precomputed answers into map from {len(loaded_answers_sanitized)} sanitized items."
        )
        return precomputed_answers_map

    except FileNotFoundError:
        logger.error(f"Precomputed answers JSON file not found: {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid format in precomputed answers JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load or parse precomputed answers JSON: {e}")
        raise


def load_and_prepare_dataset_rows(
    dataset_ref_uri: str, is_debug: bool, n_debug_samples: int, logger: logging.Logger
) -> List[Dict]:
    """Loads dataset rows from a Weave reference, applies debug sampling, and prepares them for evaluation."""
    dataset_ref = weave.ref(dataset_ref_uri).get()
    question_rows = dataset_ref.rows

    if is_debug:
        question_rows = question_rows[:n_debug_samples]

    question_rows_for_eval = []
    for i, row in enumerate(question_rows):
        if not isinstance(row, dict):
            logger.warning(f"Dataset item at original index {i} is not a dictionary, skipping: {row}")
            continue
        
        dataset_row_index_str: str
        raw_dataset_index = row.get("index")
        if raw_dataset_index is None or str(raw_dataset_index).strip() == "":
            logger.warning(
                f"Dataset item (original list index {i}, question: {str(row.get('question', 'N/A'))[:50] + '...'}) "
                f"is missing 'index' or index is empty. Using list index {i} as fallback string key."
            )
            dataset_row_index_str = str(i)
        else:
            dataset_row_index_str = str(raw_dataset_index).strip()

        question = row.get("question")
        ground_truth = row.get("answer")
        notes = row.get("notes")

        if question is None:
            logger.warning(f"Dataset item at index {dataset_row_index_str} is missing 'question'. Using empty string.")
            question = ""
        if ground_truth is None:
            logger.warning(f"Dataset item at index {dataset_row_index_str} is missing 'answer'. Using empty string.")
            ground_truth = ""
        if notes is None:
            logger.warning(f"Dataset item at index {dataset_row_index_str} is missing 'notes'. Using empty string.")
            notes = ""

        question_rows_for_eval.append(
            {
                "index": dataset_row_index_str,
                "question": str(question),
                "ground_truth": str(ground_truth),
                "notes": str(notes),
            }
        )
    return question_rows_for_eval 