from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field, field_validator

# Define required metadata fields
REQUIRED_METADATA_FIELDS: Set[str] = {"source", "source_type", "has_code", "id"}


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
    
    @field_validator("metadata")
    @classmethod
    def validate_required_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that required metadata fields are present."""
        # During ingestion/data loading, some metadata might not be set yet
        # so only validate if it appears we're validating a fully formed document
        # (if at least one required field is present, validate all required fields)
        if any(field in metadata for field in REQUIRED_METADATA_FIELDS):
            missing_fields = REQUIRED_METADATA_FIELDS - metadata.keys()
            if missing_fields:
                raise ValueError(f"Required metadata fields missing: {missing_fields}")
        return metadata
    
    def __str__(self) -> str:
        """String representation focusing on page_content and metadata."""
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        return f"page_content='{self.page_content}'"
        
    def ensure_required_fields(self) -> "Document":
        """Ensures all required metadata fields are present with default values if needed.
        
        Returns:
            The document with all required fields populated (self)
        """
        from hashlib import md5
        
        for field in REQUIRED_METADATA_FIELDS:
            if field not in self.metadata:
                if field == "id":
                    # Generate an ID based on content and existing metadata
                    content_str = self.page_content + str(self.metadata)
                    self.metadata["id"] = md5(content_str.encode("utf-8")).hexdigest()
                elif field == "source":
                    self.metadata["source"] = "unknown"
                elif field == "source_type":
                    self.metadata["source_type"] = "unknown"
                elif field == "has_code":
                    # Simple heuristic - check for code blocks or common code patterns
                    has_code = "```" in self.page_content or "def " in self.page_content
                    self.metadata["has_code"] = has_code
        
        return self


def validate_document_metadata(documents: list[Document]) -> list[Document]:
    """Validates and ensures all documents have the required metadata fields.
    
    Args:
        documents: List of documents to validate
        
    Returns:
        List of documents with required fields validated and populated
        
    Raises:
        ValueError: If any document is missing required fields and they cannot be auto-populated
    """
    valid_documents = []
    for doc in documents:
        try:
            # Try to ensure all required fields are present
            doc.ensure_required_fields()
            valid_documents.append(doc)
        except Exception as e:
            # Re-raise with more context
            raise ValueError(f"Failed to validate document: {e}\nDocument: {doc}")
    
    return valid_documents 