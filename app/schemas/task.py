from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisRequest(BaseModel):
    """Schema for analysis request"""
    query: str = Field(..., description="The financial analysis query")
    analysis_type: Optional[str] = Field(
        default="general",
        description="Type of analysis to perform"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional parameters for the analysis"
    )

class TaskResult(BaseModel):
    """Schema for task result"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None
