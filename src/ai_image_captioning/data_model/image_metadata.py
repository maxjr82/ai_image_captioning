from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ImageMetadata(BaseModel):
    """
    Pydantic model for image metadata with validation
    """

    image_id: str = Field(
        ..., min_length=1, 
        description="Unique identifier for the image"
    )
    filename: str = Field(
        ..., min_length=1,
        description="Original filename of the image"
    )
    local_path: Path = Field(...,
                             description="Local filesystem path to the image")
    generated_caption: Optional[str] = Field(
        None, max_length=500,
        description="AI-generated caption for the image"
    )
    validated_caption: Optional[str] = Field(
        None, max_length=500,
        description="Human-validated or edited caption"
    )
    extracted_text: Optional[str] = Field(
        None, max_length=2000,
        description="Text extracted from image via OCR"
    )
    processed: bool = Field(
        False,
        description="Whether the image has been processed by a human"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when record was last updated",
    )

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, v: Path) -> Path:
        """Validate that the path exists and is a file"""
        if not v.exists():
            raise ValueError(f"Path {v} does not exist")
        if not v.is_file():
            raise ValueError(f"Path {v} is not a file")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename has valid extension"""
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Filename {v} has invalid extension")
        return v

    def update_caption(self, new_caption: str, validated: bool = True) -> None:
        """Update caption and timestamps"""
        if validated:
            self.validated_caption = new_caption
        else:
            self.generated_caption = new_caption
        self.updated_at = datetime.now()

    class Config:
        json_encoders = {Path: str, datetime: lambda v: v.isoformat()}
        extra = "forbid"  # Prevent extra fields
