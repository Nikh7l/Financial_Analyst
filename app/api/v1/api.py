from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import Dict, Any
import uuid
import time

from ... import tasks
from ..schemas.task import TaskStatus, TaskResult, AnalysisRequest

api_router = APIRouter()

# In-memory storage for tasks (replace with database in production)
tasks_db: Dict[str, Dict[str, Any]] = {}

@api_router.post("/analyze", response_model=Dict[str, str])
async def create_analysis_task(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Create a new analysis task"""
    task_id = str(uuid.uuid4())
    
    # Store initial task data
    tasks_db[task_id] = {
        "status": TaskStatus.PENDING,
        "result": None,
        "created_at": time.time(),
        "request": request.dict()
    }
    
    # Add task to background
    background_tasks.add_task(
        tasks.process_analysis,
        task_id=task_id,
        request=request
    )
    
    return {"task_id": task_id, "status": "Task created", "status_url": f"/api/v1/status/{task_id}"}

@api_router.get("/status/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a task"""
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task.get("result"),
        "created_at": task["created_at"]
    }

@api_router.get("/results/{task_id}", response_model=Dict[str, Any])
async def get_task_results(task_id: str) -> Dict[str, Any]:
    """Get the results of a completed task"""
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not complete. Current status: {task['status']}"
        )
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task["result"],
        "created_at": task["created_at"],
        "completed_at": task.get("completed_at")
    }
