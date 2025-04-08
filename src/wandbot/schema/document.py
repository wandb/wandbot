from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""
    page_content: str = Field(description="String text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")
    
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize with page_content as positional or named arg."""
        super().__init__(
            page_content=page_content,
            metadata=metadata or {},
            **kwargs
        )
    
    def __str__(self) -> str:
        """String representation focusing on page_content and metadata."""
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        return f"page_content='{self.page_content}'" 