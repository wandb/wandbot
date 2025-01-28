from typing import Optional
from pydantic import BaseModel, Field

from wandbot.utils import ErrorInfo

class APIStatus(BaseModel):
    """Track status of external API calls."""
    component: str = Field(description="Name of the API component (e.g., 'web_search', 'reranker')")
    success: bool = Field(description="Whether the API call was successful")
    error_info: Optional[ErrorInfo] = Field(default=None, description="Error information if the call failed") 