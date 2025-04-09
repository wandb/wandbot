import json
import logging
from typing import Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EvaluationResult(BaseModel):
    """Result of an evaluation."""
    query: str
    response: str
    reasoning: Optional[str] = None
    score: Optional[float] = None
    passing: Optional[bool] = None
    has_error: bool = False
    error_message: Optional[str] = None
    

async def safe_parse_eval_response(eval_response: str, expected_decision: str) -> Tuple[bool, str, float]:
    """Safely parse the evaluation response."""
    try:
        # Try to find the JSON object in the response
        start = eval_response.find("{")
        end = eval_response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
            
        json_str = eval_response[start:end]
        result = json.loads(json_str)
        
        # Extract values
        passing = result["decision"].lower() == expected_decision.lower()
        reasoning = result["reason"]
        score = float(result["score"])
        
        return passing, reasoning, score
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Failed to parse evaluation response: {str(e)}\nResponse: {eval_response}")
        return False, f"Failed to parse evaluation response: {str(e)}", 1.0
