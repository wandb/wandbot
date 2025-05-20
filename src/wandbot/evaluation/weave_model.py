import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import weave
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from wandbot.evaluation.eval_schemas import EvalChatResponse  # Assuming EvalChatResponse is defined here or moved
from wandbot.utils import get_logger

logger = get_logger(__name__)


# Moved from eval.py
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=10, max=300),
    retry=retry_if_exception_type(httpx.HTTPError),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
    ),
    after=after_log(logger, logging.ERROR),
)
async def make_request(url: str, question: str, application: str = "api-eval", language: str = "en") -> dict:
    """Make HTTP request to wandbot API with retry logic."""
    request_timeout = 120.0
    request_connect_timeout = 30.0
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=request_timeout, connect=request_connect_timeout)
    ) as client:
        try:
            response = await client.post(
                f"{url}/chat/query", json={"question": question, "application": application, "language": language}
            )
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            logger.error(f"Request timed out after {request_timeout} seconds")
            raise
        except httpx.ConnectTimeout:
            logger.error(f"Connection timed out after {request_connect_timeout} seconds")
            raise


async def get_answer(question: str, wandbot_url: str, application: str = "api-eval", language: str = "en") -> str:
    """Get answer from wandbot API."""
    try:
        result = await make_request(wandbot_url, question, application, language)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Failed to get answer: {str(e)}")
        return json.dumps(
            {
                "error": str(e),
                "answer": "",
                "system_prompt": "",
                "source_documents": "",
                "model": "",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "time_taken": 0,
            }
        )


# Moved from eval.py
def parse_text_to_json(text: str) -> List[Dict[str, str]]:
    # Split the text into documents
    documents = re.split(r"source: https?://", text)[1:]
    result = []
    for doc in documents:
        source_url = "https://" + doc.split("\n")[0].strip()
        content = "\n".join(doc.split("\n")[1:]).strip()
        document = {"source": source_url, "content": content}
        result.append(document)
    return result


# Moved from eval.py
@weave.op
async def get_record(
    question: str, wandbot_url: str, application: str = "api-eval", language: str = "en"
) -> EvalChatResponse:
    _start_time = datetime.now(timezone.utc)
    try:
        response_str = await get_answer(question, wandbot_url=wandbot_url, application=application, language=language)
        response_dict = json.loads(response_str)
        _end_time = datetime.now(timezone.utc)

        if not response_dict or not response_dict.get("answer"):
            _error_msg = "Empty answer from API"
            try:
                error_data = json.loads(response_str)
                _error_msg = error_data.get("error", "Unknown API error or empty answer")
                if not response_dict.get("answer"):
                    _error_msg = "Empty answer from API"
            except json.JSONDecodeError:
                _error_msg = response_str if response_str else "Empty response from API (JSONDecodeError)"

            logger.error(_error_msg)
            return EvalChatResponse(
                question=question,
                has_error=True,
                error_message=_error_msg,
                model="api_error_or_empty_answer",
                start_time=_start_time,  # Provide start_time even for error
                end_time=_end_time,  # Provide end_time for error
            )

        raw_source_documents = response_dict.get("source_documents", "")
        parsed_retrieved_contexts = parse_text_to_json(raw_source_documents)

        return EvalChatResponse(
            question=question,
            system_prompt=response_dict.get("system_prompt", ""),
            answer=response_dict.get("answer", ""),
            sources=response_dict.get("sources", ""),
            source_documents=raw_source_documents,
            retrieved_contexts=parsed_retrieved_contexts,
            response_synthesis_llm_messages=response_dict.get("response_synthesis_llm_messages", []),
            model=response_dict.get("model", ""),
            total_tokens=response_dict.get("total_tokens", 0),
            prompt_tokens=response_dict.get("prompt_tokens", 0),
            completion_tokens=response_dict.get("completion_tokens", 0),
            time_taken=response_dict.get("time_taken", 0.0),
            api_call_statuses=response_dict.get("api_call_statuses", {}),
            start_time=_start_time,
            end_time=_end_time,
            has_error=False,
        )
    except Exception as e:
        _error_msg = f"Error getting response from wandbotAPI: {str(e)}"
        logger.error(_error_msg)
        return EvalChatResponse(
            question=question,
            has_error=True,
            error_message=_error_msg,
            model="exception_in_get_record",
            start_time=_start_time,
            end_time=datetime.now(timezone.utc),
        )


# Moved from eval.py
class WandbotModel(weave.Model):
    language: str = "en"
    application: str = "api-eval"
    wandbot_url: str = "http://0.0.0.0:8000"
    wandbot_config: dict = {}
    precomputed_data_map: Optional[Dict[str, Dict]] = None

    # Utility for parsing datetime strings within predict, can be a static method or defined outside
    @staticmethod
    def _parse_iso_datetime(dt_str: Optional[str], placeholder: str, item_index: str, field_name: str) -> str:
        if dt_str:
            try:
                # Attempt to parse then reformat to ensure consistent ISO with Z
                dt_obj = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
                return dt_obj.isoformat()
            except ValueError:
                logger.warning(
                    f"Predict: Could not parse {field_name} '{dt_str}' for item index {item_index}. Using placeholder."
                )
        return placeholder

    def _create_response_dict_from_precomputed(self, item: Dict, item_index: str) -> Dict:
        _placeholder_datetime_str = datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()

        _precomp_start_time_iso = self._parse_iso_datetime(
            item.get("start_time"), _placeholder_datetime_str, item_index, "start_time"
        )
        _precomp_end_time_iso = self._parse_iso_datetime(
            item.get("end_time"), _placeholder_datetime_str, item_index, "end_time"
        )

        raw_retrieved_contexts = item.get("retrieved_contexts", [])
        final_retrieved_contexts = []
        if isinstance(raw_retrieved_contexts, list):
            for ctx in raw_retrieved_contexts:
                if isinstance(ctx, dict):
                    final_retrieved_contexts.append(
                        {"source": str(ctx.get("source", "")), "content": str(ctx.get("content", ""))}
                    )

        has_error_val = item.get("has_error", False)
        processed_has_error = bool(
            str(has_error_val).lower() == "true" if isinstance(has_error_val, str) else has_error_val
        )

        return {
            "question": str(item.get("question", "")),
            "system_prompt": str(item.get("system_prompt", "")),
            "answer": str(item.get("generated_answer", "")),
            "model": str(item.get("model", "precomputed")),
            "sources": str(item.get("sources", "")),
            "source_documents": str(item.get("source_documents", "")),
            "total_tokens": int(str(item.get("total_tokens", "0") or "0")),
            "prompt_tokens": int(str(item.get("prompt_tokens", "0") or "0")),
            "completion_tokens": int(str(item.get("completion_tokens", "0") or "0")),
            "time_taken": float(str(item.get("time_taken", "0.0") or "0.0")),
            "start_time": _precomp_start_time_iso,
            "end_time": _precomp_end_time_iso,
            "api_call_statuses": item.get("api_call_statuses", {}),
            "response_synthesis_llm_messages": item.get("response_synthesis_llm_messages", []),
            "has_error": str(processed_has_error),  # Ensure boolean is stringified as per original logic
            "error_message": str(item.get("error_message", "")),
            "retrieved_contexts": final_retrieved_contexts,
        }

    @weave.op
    async def predict(self, index: str, question: str) -> dict:
        _current_time_for_error_str = datetime.now(timezone.utc).isoformat()

        if self.precomputed_data_map:
            if index in self.precomputed_data_map:
                precomputed_item = self.precomputed_data_map[index]
                logger.debug(f"Precomputed item at index {index} for question '{question}': {precomputed_item}")

                response_data_dict = self._create_response_dict_from_precomputed(precomputed_item, index)

                if response_data_dict.get("has_error") == "True":  # Check stringified boolean
                    logger.debug(f"Processing precomputed item (error case) for index {index}")
                else:
                    logger.debug(f"Processing precomputed item (success case) for index {index}")

                return response_data_dict
            else:
                _error_msg = f"Index {index} (Question: '{question}') not found in precomputed_answers_json_path"
                logger.warning(_error_msg)
                return {
                    "question": str(question),
                    "system_prompt": "",
                    "answer": "",
                    "model": "precomputed_not_found",
                    "sources": "",
                    "source_documents": "",
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "time_taken": 0.0,
                    "start_time": _current_time_for_error_str,
                    "end_time": _current_time_for_error_str,
                    "api_call_statuses": {},
                    "response_synthesis_llm_messages": [],
                    "has_error": True,
                    "error_message": _error_msg,
                    "retrieved_contexts": [],
                }
        else:
            eval_chat_response_obj = await get_record(
                question, wandbot_url=self.wandbot_url, application=self.application, language=self.language
            )
            return eval_chat_response_obj.model_dump(warnings=False, mode="json")
