from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid
import time
from typing import Dict, Any

# Import your langgraph app and state
from main import app as langgraph_app, AgentState
from config import config

logger = config.logger

# --- FastAPI App Initialization ---
fast_api_app = FastAPI(title="Financial Analysis Agent API")

# In-memory storage for job statuses and results
analysis_jobs: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class AnalysisRequest(BaseModel):
    query: str

class JobStatus(BaseModel):
    job_id: str
    status: str  # "PENDING", "COMPLETED", "FAILED"
    result: Any = None
    error: str | None = None

# --- Background Task Function ---
def run_langgraph_workflow(job_id: str, query: str):
    try:
        # Create initial state
        initial_state = AgentState(query=query)
        
        # Run the workflow synchronously
        result = langgraph_app.invoke(initial_state, {"recursion_limit": 50})
        
        # Store result
        analysis_jobs[job_id].update({
            "status": "COMPLETED",
            "result": result,
            "completed_at": time.time()
        })
        
    except Exception as e:
        logger.exception(f"Error in workflow for job {job_id}: {e}")
        analysis_jobs[job_id].update({
            "status": "FAILED",
            "error": str(e),
            "completed_at": time.time()
        })

# --- API Endpoints ---
@fast_api_app.post("/analyze", response_model=JobStatus, status_code=202)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new analysis task"""
    if not langgraph_app:
        raise HTTPException(status_code=503, detail="Analysis service not available")
    
    job_id = str(uuid.uuid4())
    analysis_jobs[job_id] = {
        "status": "PENDING",
        "created_at": time.time()
    }
    
    # Start background task
    background_tasks.add_task(run_langgraph_workflow, job_id, request.query)
    
    return {
        "job_id": job_id,
        "status": "PENDING"
    }

@fast_api_app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get status of a job"""
    job = analysis_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error")
    }

# --- Main Application Setup ---
app = FastAPI(title="Financial Analyst API")
app.mount("/api", fast_api_app)

@app.get("/")
async def read_root():
    return {
        "message": "Financial Analyst API is running",
        "docs": "/api/docs"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)