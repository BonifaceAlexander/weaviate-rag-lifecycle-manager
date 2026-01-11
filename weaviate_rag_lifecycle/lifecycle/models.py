from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from .states import LifecycleState

class RAGDataset(BaseModel):
    id: str = Field(description="Unique identifier for the dataset version")
    name: str = Field(description="Logical name of the dataset")
    version: str = Field(description="Semantic version or tag")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EmbeddingConfig(BaseModel):
    id: str = Field(description="Unique configuration hash or ID")
    model_name: str = Field(description="Name of the embedding model")
    chunk_size: int = Field(description="Size of text chunks")
    chunk_overlap: int = Field(description="Overlap between chunks")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class IndexGeneration(BaseModel):
    id: str = Field(description="Unique generation ID (UUID)")
    dataset_id: str = Field(description="Reference to RAGDataset ID")
    config_id: str = Field(description="Reference to EmbeddingConfig ID")
    status: LifecycleState = Field(default=LifecycleState.DRAFT)
    weaviate_collection_name: str = Field(description="Physical Weaviate collection name")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
