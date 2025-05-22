from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from wandbot.chat.schemas import ChatResponse


class EvalChatResponse(ChatResponse):
    """
    An evaluation-specific ChatResponse that includes error tracking
    and makes certain fields optional for error scenarios.
    It also includes a field for pre-parsed retrieved_contexts.
    """

    # Fields inherited from ChatResponse, overridden for error handling:
    # question: str  (remains mandatory from ChatResponse)
    system_prompt: Optional[str] = ""
    answer: Optional[str] = ""
    model: Optional[str] = ""
    sources: Optional[str] = ""
    source_documents: Optional[str] = ""  # Raw string from API / precomputed
    total_tokens: Optional[int] = 0
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    time_taken: Optional[float] = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    api_call_statuses: Optional[Dict] = Field(default_factory=dict)
    response_synthesis_llm_messages: Optional[List[Dict[str, str]]] = Field(default_factory=list)

    # New fields for evaluation context and error tracking
    has_error: bool = False
    error_message: Optional[str] = None
    retrieved_contexts: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="Parsed source documents into list of dicts for scorer",
    )

    class Config:
        from_attributes = True
