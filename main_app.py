from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid
import time
from typing import Dict, Any, Optional

# --- Your LangGraph Application ---
# Make sure 'app' (compiled graph) is accessible here
# You might need to adjust imports based on your final structure
# It's often good to have a separate module that initializes and returns the compiled graph
from main_graph import app as langgraph_app, AgentState # Assuming app is compiled in main_graph.py
from config import logger

# --- FastAPI App Initialization ---
fast_api_app = FastAPI(title="Financial Analysis Agent API")

# --- (Optional) Mount static files and templates if serving frontend from FastAPI ---
# fast_api_app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# --- In-memory storage for job statuses and results (for simplicity) ---
# For production, use a database (Redis, PostgreSQL, etc.)
analysis_jobs: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models for API Requests/Responses ---
class AnalysisRequest(BaseModel):
    query: str

class JobStatus(BaseModel):
    job_id: str
    status: str # e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED"
    message: Optional[str] = None
    result: Optional[AgentState] = None # Or a more specific result model
    progress: Optional[int] = 0 # Percentage or step number
    current_node: Optional[str] = None

# --- Background Task Function ---
def run_langgraph_workflow(job_id: str, initial_state: AgentState):
    try:
        analysis_jobs[job_id]["status"] = "RUNNING"
        analysis_jobs[job_id]["current_node"] = "Starting..."
        
        # Stream the process to update progress
        # Note: This is a simplified progress update. Real progress might be more complex.
        step_count = 0
        total_nodes_approx_company = 8 # Router, Prepare, 4 data, Predict, Report
        total_nodes_approx_sector = 7  # Router, Prepare, 4 data, Synthesize, Report

        for event in langgraph_app.stream(initial_state, {"recursion_limit": 50}):
            step_count += 1
            node_name = list(event.keys())[0]
            analysis_jobs[job_id]["current_node"] = node_name
            
            # Estimate progress (very rough)
            query_type = analysis_jobs[job_id].get("query_type", initial_state.get("query_type", "unknown"))
            total_nodes = total_nodes_approx_company if query_type == "company" else total_nodes_approx_sector
            if node_name == END: # END is also a node LangGraph reports
                 analysis_jobs[job_id]["progress"] = 100
            else:
                 analysis_jobs[job_id]["progress"] = min(95, int((step_count / total_nodes) * 100))

            # Store intermediate state if needed for more detailed progress, or just final
            # For now, we'll just update the final state at the end.
            # To get the current full state from the stream, you'd need to accumulate updates.
            # current_full_state = {}
            # for k,v in event.items(): current_full_state.update(v) # simplified
            # analysis_jobs[job_id]["intermediate_state"] = current_full_state


        # After streaming, invoke to get the final accumulated state
        final_state = langgraph_app.invoke(initial_state, {"recursion_limit": 50})
        
        analysis_jobs[job_id]["status"] = "COMPLETED"
        analysis_jobs[job_id]["result"] = final_state
        analysis_jobs[job_id]["progress"] = 100
        analysis_jobs[job_id]["current_node"] = "Finished"
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Error running LangGraph workflow for job {job_id}: {e}")
        analysis_jobs[job_id]["status"] = "FAILED"
        analysis_jobs[job_id]["message"] = str(e)
        analysis_jobs[job_id]["progress"] = 100 # Mark as done even if failed for progress bar
        analysis_jobs[job_id]["current_node"] = "Error"


# --- API Endpoints ---

@fast_api_app.post("/analyze", response_model=JobStatus, status_code=202)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Starts a financial analysis workflow for the given query.
    Returns a job ID to track the status.
    """
    if not langgraph_app:
        raise HTTPException(status_code=503, detail="Analysis service is not available (graph not compiled).")

    job_id = str(uuid.uuid4())
    initial_state = AgentState(query=request.query) # Use your main AgentState

    analysis_jobs[job_id] = {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Analysis initiated.",
        "query": request.query, # Store original query
        "result": None,
        "progress": 0,
        "current_node": "Pending"
    }
    
    background_tasks.add_task(run_langgraph_workflow, job_id, initial_state)
    logger.info(f"Analysis job {job_id} started for query: '{request.query}'")
    
    # Return initial job status
    return JobStatus(**analysis_jobs[job_id])


@fast_api_app.get("/status/{job_id}", response_model=JobStatus)
async def get_analysis_status(job_id: str):
    """
    Retrieves the status and result (if completed) of an analysis job.
    """
    job = analysis_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    # Ensure all fields of JobStatus are present, even if None
    return JobStatus(
        job_id=job.get("job_id", job_id),
        status=job.get("status", "UNKNOWN"),
        message=job.get("message"),
        result=job.get("result"), # This will be the full AgentState
        progress=job.get("progress", 0),
        current_node=job.get("current_node", "Unknown")
    )

# --- (Optional) Basic HTML Frontend Endpoint (if serving from FastAPI) ---
# @fast_api_app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     """Serves the main HTML page for the frontend."""
#     return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    # Ensure your main_graph.app is compiled before starting uvicorn
    if not langgraph_app:
        logger.critical("LangGraph application (app) is not compiled or available. FastAPI server cannot start.")
        exit()
    logger.info("Starting FastAPI server...")
    uvicorn.run(fast_api_app, host="0.0.0.0", port=8000)