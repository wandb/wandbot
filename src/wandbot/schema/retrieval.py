from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from wandbot.schema.document import Document
from wandbot.utils import ErrorInfo


class APIStatus(BaseModel):
    """Track status of external API calls during retrieval."""
    component: str = Field(description="Name of the API component (e.g., 'web_search', 'reranker')")
    success: bool = Field(description="Whether the API call was successful")
    error_info: Optional[ErrorInfo] = Field(default=None, description="Error information if the call failed")

class RetrievalResult(BaseModel):
    """Standardized output format for retrievers."""
    documents: List[Document] = Field(description="Retrieved documents")
    retrieval_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics and metadata about the retrieved documents (e.g., number of docs, sources, timing)"
    )
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, idx) -> Document:
        return self.documents[idx] 