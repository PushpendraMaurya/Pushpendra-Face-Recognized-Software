from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the person")
    role: Optional[str] = Field("Member", max_length=50, description="Role or department")
    image: str = Field(..., description="Base64 encoded image string or data URI")

class RecognizeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string or data URI")
    threshold: Optional[float] = Field(None, ge=0.1, le=0.9, description="Custom cosine similarity threshold")
    save_log: Optional[bool] = Field(False, description="Whether to record a recognition event in the log")

class AddSampleRequest(BaseModel):
    user_id: int
    image: str

class UserResponse(BaseModel):
    id: int
    name: str
    role: str
    thumbnail_path: Optional[str]
    sample_count: int
    created_at: str

class RecognitionLogResponse(BaseModel):
    id: int
    user_id: Optional[int]
    user_name: str
    confidence: float
    similarity_score: float
    snapshot_path: Optional[str]
    created_at: str
