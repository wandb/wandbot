import json
from typing import Dict, List

import weave

from wandbot.evaluation.eval_config import EvalConfig
from wandbot.evaluation.eval_metrics.correctness import (
    CorrectnessEvaluationResult,
    WandBotCorrectnessEvaluator,
)
from wandbot.utils import get_logger

logger = get_logger(__name__)

# Moved from eval.py
class WandbotCorrectnessScorer(weave.Scorer):
    config: EvalConfig
    correctness_evaluator: WandBotCorrectnessEvaluator = None
    debug: bool = False
    _all_scored_results: List[Dict]

    def __init__(self, config: EvalConfig):
        super().__init__(config=config)
        self.debug = config.debug
        self.correctness_evaluator = WandBotCorrectnessEvaluator(
            provider=config.eval_judge_provider,
            model_name=config.eval_judge_model,
            temperature=config.eval_judge_temperature,
            max_retries=config.max_evaluator_retries,
            timeout=config.evaluator_timeout,
        )
        self._all_scored_results = []

    @weave.op
    async def score(
        self, index: int, question: str, ground_truth: str, notes: str, model_output: dict  # Expect dict
    ) -> dict:
        if self.debug:
            if model_output is not None:
                if str(model_output.get('has_error')).lower() == 'true':
                    logger.debug(
                        f"In WandbotCorrectnessScorer, model_output is dict (error) with message: {model_output.get('error_message')}"
                    )
                else:
                    logger.debug(
                        f"In WandbotCorrectnessScorer, model_output is dict (success) with answer: {model_output.get('answer', '')[:50] if model_output.get('answer') else 'None'}..."
                    )
            else:
                logger.error("model_output is None (should be a dict)")

        score_result_dict: Dict
        try:
            if str(model_output.get('has_error')).lower() == 'true':
                _error_msg = model_output.get('error_message') or "Unknown error from model_output dict"
                logger.error(f"Model output dict indicates an error: {_error_msg}")
                score_result_dict = CorrectnessEvaluationResult(
                    reason=_error_msg,
                    score=1.0,
                    decision="incorrect",
                    answer_correct=False,
                    has_error=True,
                    error_message=_error_msg
                ).model_dump(warnings=False)
            elif not model_output.get('answer'):
                _error_msg = "Generated answer is empty (from model_output dict)"
                logger.error(_error_msg)
                score_result_dict = CorrectnessEvaluationResult(
                    reason=_error_msg,
                    score=1.0,
                    decision="incorrect",
                    answer_correct=False,
                    has_error=True,
                    error_message=_error_msg
                ).model_dump(warnings=False)
            else:
                actual_answer = model_output.get('answer', '')
                retrieved_contexts_for_eval = model_output.get('retrieved_contexts') or []
                contexts_for_eval = [str(c.get("content", "")) for c in retrieved_contexts_for_eval if isinstance(c, dict)]
                
                evaluator_output = await self.correctness_evaluator.aevaluate(
                    query=question, response=actual_answer, reference=ground_truth,
                    contexts=contexts_for_eval, reference_notes=notes,
                )
                score_result_dict = evaluator_output.model_dump(warnings=False)

        except Exception as e:
            _error_msg = f"Error evaluating answer: {str(e)}"
            logger.error(_error_msg, exc_info=True)
            
            _current_answer_for_error_payload = model_output.get('answer', '') if isinstance(model_output, dict) else ""
            _current_retrieved_contexts = model_output.get('retrieved_contexts') if isinstance(model_output, dict) else []
            _current_contexts_for_error_payload = [str(c.get("content", "")) for c in (_current_retrieved_contexts or []) if isinstance(c, dict)]
            
            _error_message_for_payload = _error_msg
            if isinstance(model_output, dict) and model_output.get('has_error') and model_output.get('error_message'):
                _error_message_for_payload = model_output.get('error_message')

            eval_result_obj = CorrectnessEvaluationResult(
                reason=str(_error_message_for_payload or _error_msg),
                score=1.0,
                decision="incorrect",
                answer_correct=False,
                has_error=True, error_message=str(_error_message_for_payload or _error_msg)
            )
            score_result_dict = eval_result_obj.model_dump(warnings=False)

        score_result_dict["reason"] = str(score_result_dict.get("reason", ""))
        score_result_dict["error_message"] = str(score_result_dict.get("error_message", ""))

        if self.debug:
            logger.debug(f"Final score_result_dict for index {index} before returning to Weave: {score_result_dict}")

        self._all_scored_results.append({
            "index": index, "question": question, "ground_truth": ground_truth,
            "notes": notes, "model_output": json.dumps(model_output),
            "score_output": score_result_dict
        })
        return score_result_dict

    def get_all_scored_results(self) -> List[Dict]:
        return self._all_scored_results 