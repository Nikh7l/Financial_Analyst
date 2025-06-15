import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict, Any
from google import genai

# Load environment variables from .env file in the project root
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

from config import config

logger = config.logger
logger.info(f"Loaded environment variables from: {env_path.absolute()}")

# Verify required environment variables
if not config.GOOGLE_API_KEY:
    logger.error("CRITICAL: GOOGLE_API_KEY environment variable not found")
    sys.exit(1)
from core.state import AgentState , RetrievalSummarizationState,ReportSummariesPayload,StockDataPayload,StockInfoPayload,SentimentAnalysisPayload,CompetitorSubgraphState,SentimentSubgraphState,CompetitorAnalysisPayload,CompetitorDetailPayload,SectorSentimentAnalysisPayload,SectorKeyPlayersPayload,SectorMarketDataPayload,SectorTrendsInnovationsPayload
from agents.router_agent import RouterAgent
from subgraphs.retrieval_summarization_graph import create_retrieval_summarization_graph
from subgraphs.stock_agent_subgraph import stock_data_subgraph_runnable,StockDataSubgraphState
from subgraphs.competetior_anaysis_subgraph import competitor_analysis_subgraph_runnable, CompetitorSubgraphState
from subgraphs.sentiment_agent_subgraph import sentiment_analysis_subgraph_runnable,SentimentSubgraphState,MAX_SENTIMENT_ATTEMPTS 
from nodes.company_analysis import predict_stock_price_node
from nodes.report_generation_node import generate_llm_based_report_node
from subgraphs.sector_keyplayers_subgraph import sector_key_players_subgraph_runnable, SectorKeyPlayersSubgraphState,MAX_KEY_PLAYERS_ATTEMPTS
from subgraphs.sector_sentiment_subgraph import sector_sentiment_analysis_subgraph_runnable, SectorSentimentSubgraphState,MAX_SECTOR_SENTIMENT_ATTEMPTS
from subgraphs.sector_market_data_subgraph import sector_market_data_subgraph_runnable, SectorMarketDataSubgraphState,MAX_MARKET_DATA_ATTEMPTS
from subgraphs.sector_trends_subgraph import sector_trends_subgraph_runnable, SectorTrendsSubgraphState, MAX_TRENDS_ATTEMPTS
from nodes.sector_analysis_node import synthesize_sector_outlook_node
from nodes.sector_report_node import generate_llm_sector_report_node

# Instantiate your agents

logger.info("--- Initializing Main Graph Components ---")

client = genai.Client()
model_name = config.GEMINI_MODEL_NAME
router_agent = RouterAgent(client=client, model_name=model_name)

# Compile the retrieval-summarization graph once
try:
    retrieval_summarization_app = create_retrieval_summarization_graph()
    if not retrieval_summarization_app:
        raise RuntimeError("Failed to compile retrieval_summarization_graph")
    logger.info("Retrieval-Summarization subgraph compiled successfully.")
except Exception as e:
    logger.exception("Failed to compile Retrieval-Summarization subgraph.")
    retrieval_summarization_app = None 

if not stock_data_subgraph_runnable: logger.error("Stock Data subgraph runnable is not available.")
if not competitor_analysis_subgraph_runnable: logger.error("Competitor Analysis subgraph runnable is not available.")
if not sentiment_analysis_subgraph_runnable: logger.error("Sentiment Analysis subgraph runnable is not available.")


def prepare_company_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to prepare for company-specific analysis.
    It acts as a gateway and branching point for parallel company data gathering tasks.
    """
    company_name = state.get("company_name")
    current_query = state.get("query") 
    if not company_name:
        logger.error("[PrepareCompanyAnalysis] Company name is missing in state. Cannot proceed with company analysis.")
        return {
            "error_message": "Company name missing for company analysis pathway.",
            "next_node_override": "handle_error_or_end" 
        }

    logger.info(f"--- Starting Company Analysis Workflow for: '{company_name}' (Query: '{current_query}') ---")

    return {} 

def run_retrieval_summarization_subgraph(state: AgentState) -> Dict[str, Any]:
    company_name = state.get("company_name")
    if not company_name:
        logger.error("Retrieval-Summarization: Company name missing in main state.")
        return {"report_summaries": {"error": "Company name missing"}} # Or however you want to signal main graph error

    logger.info(f"[MainGraph] Invoking Retrieval-Summarization subgraph for: {company_name}")

    # Prepare initial state for the subgraph
    subgraph_initial_messages = [("human", f"Find the latest annual and quarterly financial reports for {company_name}.")]
    subgraph_initial_state: RetrievalSummarizationState = {
        "company_name": company_name,
        "messages": subgraph_initial_messages,
        "document_urls": None,
        "document_summaries": None,
        "subgraph_error": None,
    }

    try:
        subgraph_final_state : RetrievalSummarizationState = retrieval_summarization_app.invoke(subgraph_initial_state, {"recursion_limit": 40})

        payload: ReportSummariesPayload = {
            "document_summaries": subgraph_final_state.get("document_summaries"),
            "retrieved_document_urls": subgraph_final_state.get("document_urls"),
            "error": subgraph_final_state.get("subgraph_error")
            }

        logger.info(f"[RetrievalWrapper] Complete for {company_name}. Error: {payload.get('error')}")
        return {"report_summaries_output": payload} 

    except Exception as e:
        logger.exception(f"[MainGraph] Exception while running Retrieval-Summarization subgraph for {company_name}")
        return {"report_summaries_output": {"error": str(e), "summaries": {}, "urls": []}}



def run_stock_data_subgraph(state: AgentState) -> Dict[str, StockInfoPayload]: 
    company_name = state.get("company_name")
    if not company_name:
        logger.error("[StockDataWrapper] Company name missing.")
        return {"stock_data_output": {"error": "Company name missing", "stock_metrics": None, "source_urls_used": []}}

    logger.info(f"[StockDataWrapper] Invoking for: {company_name}")
    subgraph_initial_state: StockDataSubgraphState = {
        "company_name": company_name, "attempt": 1, "max_attempts": 2, 
        "messages": [], "stock_data": None, "source_urls_used": None, "subgraph_error": None,
    }
    try:
        if not stock_data_subgraph_runnable:
             logger.error("[StockDataWrapper] Subgraph runnable not available.")
             return {"stock_data_output": {"error": "Stock data subgraph not compiled/imported.", "stock_metrics": None, "source_urls_used": []}}

        subgraph_final_state = stock_data_subgraph_runnable.invoke(subgraph_initial_state, {"recursion_limit": 30})

        actual_stock_metrics_payload: Optional[StockDataPayload] = subgraph_final_state.get("stock_data")
        
        payload: StockInfoPayload = {
            "stock_metrics": actual_stock_metrics_payload,
            "source_urls_used": subgraph_final_state.get("source_urls_used"), 
            "error": subgraph_final_state.get("subgraph_error")
        }
        logger.info(f"[StockDataWrapper] Complete for {company_name}. Error: {payload.get('error')}")
        return {"stock_data_output": payload}
    except Exception as e:
        logger.exception(f"[StockDataWrapper] Exception for {company_name}")
        return {"stock_data_output": {"error": f"Wrapper exception: {str(e)}", "stock_metrics": None, "source_urls_used": []}}
  

def run_competitor_analysis_subgraph(state: AgentState) -> Dict[str, Any]:
    company_name = state.get("company_name")
    if not company_name:
        logger.error("Competitor Analysis Subgraph: Company name missing in main state.")
        return {"competitor_info": {"competitors": [], "error": "Company name missing"}}

    logger.info(f"[MainGraph] Invoking Competitor Analysis subgraph for: {company_name}")

    subgraph_initial_state: CompetitorSubgraphState = {
        "company_name": company_name,
        "attempt": 1,
        "max_attempts": 2, # Or configure as needed
        "messages": [],
        "competitors": None,
        "subgraph_error": None,
    }

    try:
        subgraph_final_state : CompetitorSubgraphState= competitor_analysis_subgraph_runnable.invoke(
            subgraph_initial_state,
            {"recursion_limit": 30} # Adjust as needed
        )

        competitors_from_subgraph: Optional[List[CompetitorDetailPayload]] = subgraph_final_state.get("competitors")

        payload: CompetitorAnalysisPayload = {
            "competitors_list": competitors_from_subgraph,
            "error": subgraph_final_state.get("subgraph_error")
        }
        logger.info(f"[CompetitorWrapper] Competitor analysis for {company_name} complete. Found {len(payload.get('competitors_list', []))} competitors. Error: {payload.get('error')}")
        return {"competitor_info_output": payload}

    except Exception as e:
        logger.exception(f"[CompetitorWrapper] Critical exception running competitor subgraph for {company_name}.")
        return {"competitor_info_output": {"error": f"Wrapper exception: {str(e)}"}}

def run_sentiment_analysis_subgraph(state: AgentState) -> Dict[str, Any]:
    """
    Wrapper node to run the sentiment analysis subgraph.
    Takes company_name from the main AgentState, invokes the subgraph,
    and puts the results back into the main AgentState.
    """
    company_name = state.get("company_name")
    if not company_name:
        logger.error("[MainGraph Wrapper] Sentiment Analysis: Company name missing in main state.")
        return {"sentiment_analysis_results": {"error": "Company name missing", "sentiment": None, "detailed_sentiment_report": None}}

    logger.info(f"[MainGraph Wrapper] Invoking Sentiment Analysis subgraph for: {company_name}")

    subgraph_initial_state: SentimentSubgraphState = {
        "company_name": company_name,
        "messages": [], 
        "attempt": 1,
        "max_attempts": MAX_SENTIMENT_ATTEMPTS, 
        "sentiment_analysis": None,
        "subgraph_error": None,
    }

    try:
        subgraph_final_state : SentimentSubgraphState= sentiment_analysis_subgraph_runnable.invoke(
            subgraph_initial_state,
            {"recursion_limit": 40} 
        )

        payload: SentimentAnalysisPayload = {
            "sentiment_data": subgraph_final_state.get("sentiment_analysis"),
            "error": subgraph_final_state.get("subgraph_error")
        }
        logger.info(f"[SentimentWrapper] Sentiment analysis for {company_name} complete. Error: {payload.get('error')}")
        return {"sentiment_analysis_output": payload}

    except Exception as e:
        logger.exception(f"[SentimentWrapper] Critical exception running sentiment subgraph for {company_name}.")
        return {"sentiment_analysis_output": {"error": f"Wrapper exception: {str(e)}"}}    


##################################################################################################################
# Nodes for sector analysis 

def prepare_sector_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to prepare for Sector-specific analysis.
    It acts as a gateway and branching point for parallel Sector data gathering tasks.
    """
    sector_name = state.get("sector_name")
    current_query = state.get("query") 

    if not sector_name:
        logger.error("[PrepareSectorAnalysis] Sector name is missing in state. Cannot proceed with Sector analysis.")
        return {
            "error_message": "Sector name missing for Sector analysis pathway.",
            "next_node_override": "handle_error_or_end" 
        }

    logger.info(f"--- Starting Sector Analysis Workflow for: '{sector_name}' (Query: '{current_query}') ---")

    return {} 


def run_sector_sentiment_wrapper(state: AgentState) -> Dict[str, SectorSentimentAnalysisPayload]:
    sector_name_main = state.get("sector_name")
    if not sector_name_main:
        logger.error("[SectorSentimentWrapper] Sector name missing.")
        return {"sector_news_sentiment_output": {"error": "Sector name missing"}}

    logger.info(f"[SectorSentimentWrapper] Invoking for SECTOR: {sector_name_main}")
    subgraph_initial_input: SectorSentimentSubgraphState = {
        "sector_name": sector_name_main,
        "messages": [],
        "attempt": 1,
        "max_attempts": MAX_SECTOR_SENTIMENT_ATTEMPTS,
    }
    try:
        subgraph_final_state = sector_sentiment_analysis_subgraph_runnable.invoke(
            subgraph_initial_input, {"recursion_limit": 40}
        )
        payload: SectorSentimentAnalysisPayload = {
            "analysis_data": subgraph_final_state.get("sector_sentiment_analysis"),
            "error": subgraph_final_state.get("subgraph_error")
        }
        logger.info(f"[SectorSentimentWrapper] Complete for SECTOR {sector_name_main}. Error: {payload.get('error')}")
        return {"sector_news_sentiment_output": payload}
    except Exception as e:
        logger.exception(f"[SectorSentimentWrapper] Exception for SECTOR {sector_name_main}.")
        return {"sector_news_sentiment_output": {"error": f"Wrapper exception: {str(e)}"}}
    

def run_sector_key_players_wrapper(state: AgentState) -> Dict[str, SectorKeyPlayersPayload]:
    sector_name_main = state.get("sector_name")
    if not sector_name_main:
        logger.error("[SectorKeyPlayersWrapper] Sector name missing.")
        return {"sector_key_players_output": {"error": "Sector name missing"}}

    logger.info(f"[SectorKeyPlayersWrapper] Invoking for SECTOR: {sector_name_main}")
    subgraph_initial_input: SectorKeyPlayersSubgraphState = {
        "sector_name": sector_name_main,
        "messages": [],
        "attempt": 1,
        "max_attempts": MAX_KEY_PLAYERS_ATTEMPTS,
        "identified_key_players_data": None,
        "subgraph_error": None,
    }
    try:
        if not sector_key_players_subgraph_runnable:
            logger.error("[SectorKeyPlayersWrapper] Subgraph runnable not available.")
            return {"sector_key_players_output": {"error": "Key players subgraph not compiled/imported."}}

        subgraph_final_state: SectorKeyPlayersSubgraphState = sector_key_players_subgraph_runnable.invoke(
            subgraph_initial_input, {"recursion_limit": 40}
        )
        
        # The subgraph state 'identified_key_players_data' holds the dict {"key_players": [...], "source_urls_used": [...]}
        data_from_subgraph = subgraph_final_state.get("identified_key_players_data")

        payload: SectorKeyPlayersPayload = {
            "key_players": data_from_subgraph.get("key_players") if data_from_subgraph else None,
            "source_urls_used": data_from_subgraph.get("source_urls_used") if data_from_subgraph else None,
            "error": subgraph_final_state.get("subgraph_error")
        }
        
        if payload.get("key_players"):
             logger.info(f"[SectorKeyPlayersWrapper] Complete for SECTOR {sector_name_main}. Found {len(payload['key_players'])} players. Error: {payload.get('error')}")
        else:
             logger.info(f"[SectorKeyPlayersWrapper] Complete for SECTOR {sector_name_main}. No players found or error occurred. Error: {payload.get('error')}")

        return {"sector_key_players_output": payload}
    except Exception as e:
        logger.exception(f"[SectorKeyPlayersWrapper] Exception for SECTOR {sector_name_main}.")
        return {"sector_key_players_output": {"error": f"Wrapper exception: {str(e)}"}}
    
def run_sector_market_data_wrapper(state: AgentState) -> Dict[str, SectorMarketDataPayload]:
    sector_name_main = state.get("sector_name")
    if not sector_name_main:
        logger.error("[SectorMarketDataWrapper] Sector name missing.")
        return {"sector_market_data_output": {"error": "Sector name missing"}}

    logger.info(f"[SectorMarketDataWrapper] Invoking for SECTOR: {sector_name_main}")
    subgraph_initial_input: SectorMarketDataSubgraphState = {
        "sector_name": sector_name_main,
        "messages": [],
        "attempt": 1,
        "max_attempts": MAX_MARKET_DATA_ATTEMPTS,
        "extracted_market_data": None,
        "subgraph_error": None,
    }

    try:
        if not sector_market_data_subgraph_runnable:
            logger.error("[SectorMarketDataWrapper] Subgraph runnable not available.")
            return {"sector_market_data_output": {"error": "Market data subgraph not compiled/imported."}}

        subgraph_final_state: SectorMarketDataSubgraphState = sector_market_data_subgraph_runnable.invoke(
            subgraph_initial_input, {"recursion_limit": 50} # Allow more steps
        )
        
        # 'extracted_market_data' from subgraph state contains {"market_data": {...}, "source_urls_used": [...]}
        data_from_subgraph = subgraph_final_state.get("extracted_market_data")

        payload: SectorMarketDataPayload = {
            "market_data": data_from_subgraph.get("market_data") if data_from_subgraph else None,
            "source_urls_used": data_from_subgraph.get("source_urls_used") if data_from_subgraph else None,
            "error": subgraph_final_state.get("subgraph_error")
        }
        
        if payload.get("market_data"):
             logger.info(f"[SectorMarketDataWrapper] Complete for SECTOR {sector_name_main}. Data found. Error: {payload.get('error')}")
        else:
             logger.info(f"[SectorMarketDataWrapper] Complete for SECTOR {sector_name_main}. No specific market data found or error. Error: {payload.get('error')}")

        return {"sector_market_data_output": payload}
    except Exception as e:
        logger.exception(f"[SectorMarketDataWrapper] Exception for SECTOR {sector_name_main}.")
        return {"sector_market_data_output": {"error": f"Wrapper exception: {str(e)}"}}

def run_sector_trends_wrapper(state: AgentState) -> Dict[str, SectorTrendsInnovationsPayload]:
    sector_name_main = state.get("sector_name")
    if not sector_name_main:
        logger.error("[SectorTrendsWrapper] Sector name missing.")
        return {"sector_trends_innovations_output": {"error": "Sector name missing"}}

    logger.info(f"[SectorTrendsWrapper] Invoking for SECTOR: {sector_name_main}")
    subgraph_initial_input: SectorTrendsSubgraphState = {
        "sector_name": sector_name_main,
        "messages": [],
        "attempt": 1,
        "max_attempts": MAX_TRENDS_ATTEMPTS,
        "extracted_trends_data": None,
        "subgraph_error": None,
    }

    try:
        if not sector_trends_subgraph_runnable:
            logger.error("[SectorTrendsWrapper] Subgraph runnable not available.")
            return {"sector_trends_innovations_output": {"error": "Trends/Innovations subgraph not compiled/imported."}}

        subgraph_final_state: SectorTrendsSubgraphState = sector_trends_subgraph_runnable.invoke(
            subgraph_initial_input, {"recursion_limit": 60} # Allow more steps for research
        )
        
        # 'extracted_trends_data' from subgraph state contains {"trends_data": {...}, "source_urls_used": [...]}
        data_from_subgraph = subgraph_final_state.get("extracted_trends_data")

        payload: SectorTrendsInnovationsPayload = {
            "trends_data": data_from_subgraph.get("trends_data") if data_from_subgraph else None,
            "source_urls_used": data_from_subgraph.get("source_urls_used") if data_from_subgraph else None,
            "error": subgraph_final_state.get("subgraph_error")
        }
        
        if payload.get("trends_data"):
             logger.info(f"[SectorTrendsWrapper] Complete for SECTOR {sector_name_main}. Trends data found. Error: {payload.get('error')}")
        else:
             logger.info(f"[SectorTrendsWrapper] Complete for SECTOR {sector_name_main}. No specific trends data found or error. Error: {payload.get('error')}")

        return {"sector_trends_innovations_output": payload}
    except Exception as e:
        logger.exception(f"[SectorTrendsWrapper] Exception for SECTOR {sector_name_main}.")
        return {"sector_trends_innovations_output": {"error": f"Wrapper exception: {str(e)}"}}
    



















workflow = StateGraph(AgentState)
logger.info("Adding nodes to the main graph...")
# Add Nodes
workflow.add_node("router", router_agent.run) 
# Company Analysis Nodes
workflow.add_node("prepare_company_analysis", prepare_company_analysis_node)
workflow.add_node("process_financial_reports", run_retrieval_summarization_subgraph) 
workflow.add_node("analyze_market_sentiment", run_sentiment_analysis_subgraph) 
workflow.add_node("gather_stock_data", run_stock_data_subgraph) 
workflow.add_node("find_competitors", run_competitor_analysis_subgraph) 
workflow.add_node("generate_stock_prediction", predict_stock_price_node) 
workflow.add_node("generate_final_report", generate_llm_based_report_node) 

# Sector Analysis Nodes
workflow.add_node("prepare_sector_analysis", prepare_sector_analysis_node)
workflow.add_node("run_sector_sentiment_analysis", run_sector_sentiment_wrapper) 
workflow.add_node("run_sector_key_players_analysis", run_sector_key_players_wrapper)
workflow.add_node("run_sector_market_data_analysis", run_sector_market_data_wrapper) 
workflow.add_node("run_sector_trends_analysis", run_sector_trends_wrapper)
workflow.add_node("synthesize_sector_outlook", synthesize_sector_outlook_node)
workflow.add_node("generate_sector_report", generate_llm_sector_report_node)
# Set Entry Point
workflow.set_entry_point("router")

# Define Conditional Routing from Router
def decide_next_step(state: AgentState):
    query_type = state.get("query_type")
    if not query_type or state.get("main_graph_error_message"): # Check for explicit main graph error
        logger.error(f"Routing Error: Query Type='{query_type}', Error='{state.get('main_graph_error_message')}'. Ending workflow.")
        return END 
    
    logger.info(f"Router decision: query_type='{query_type}'")
    if query_type == "company":
        # Check if company_name exists before proceeding
        if state.get("company_name"):
            return "prepare_company_analysis"
        else:
            logger.error("Routing Error: Query type is 'company' but company_name is missing. Ending.")
            return END
    elif query_type == "sector":
        if state.get("sector_name"):
            return "prepare_sector_analysis"
        else:
            logger.error("Routing Error: Query type is 'Sector' but sector_name is missing. Ending.")
            return END
    else: 
        logger.warning(f"Query type '{query_type}' is unknown or router failed. Ending.")
        return END
    
workflow.add_conditional_edges(
    "router",
    decide_next_step,
    {
        "prepare_company_analysis": "prepare_company_analysis",
        "prepare_sector_analysis": "prepare_sector_analysis", 
        # "error_handler_node": "error_handler_node",
        END: END
    }
)


workflow.add_edge("prepare_company_analysis", "process_financial_reports")
workflow.add_edge("process_financial_reports", "analyze_market_sentiment")
workflow.add_edge("analyze_market_sentiment", "gather_stock_data")
workflow.add_edge("gather_stock_data", "find_competitors")
workflow.add_edge("find_competitors", "generate_stock_prediction")
workflow.add_edge("generate_stock_prediction", "generate_final_report")
workflow.add_edge("generate_final_report", END)

# --- SECTOR ANALYSIS PATHWAY (Placeholder) ---
workflow.add_edge("prepare_sector_analysis","run_sector_sentiment_analysis" ) 
workflow.add_edge("run_sector_sentiment_analysis", "run_sector_key_players_analysis")
workflow.add_edge("run_sector_key_players_analysis", "run_sector_market_data_analysis")
workflow.add_edge("run_sector_market_data_analysis", "run_sector_trends_analysis")
workflow.add_edge("run_sector_trends_analysis", "synthesize_sector_outlook")
workflow.add_edge("synthesize_sector_outlook", "generate_sector_report")
workflow.add_edge("generate_sector_report", END)    

logger.info("Compiling the main graph...")
try:
    app = workflow.compile()
    logger.info("Main graph compiled successfully.")
except Exception as e:
    logger.exception("Failed to compile main graph!")
    app = None

# --- Main Execution Block ---
if __name__ == "__main__":
    # Configure LangSmith if enabled
    if not config.LANGCHAIN_ENABLED:
        logger.warning("LangSmith tracing is not configured. Some features may be limited.")
    
    if not app:
        logger.info("\nERROR: Main application graph failed to compile. Exiting.")
        sys.exit(1)

    logger.info("\n--- Financial Analysis Agent ---")
    logger.info("Enter your query (e.g., 'analyze Apple', 'what are trends in tech sector?'):")
    # Use a loop for interactive testing
    while True:
        user_query = input("> ")
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        initial_state = AgentState(query=user_query)
        final_state = None
        start_time = time.time()

        logger.info("\nProcessing query...")
        logger.info(f"--- Starting Workflow for Query: '{user_query}' ---")
        try:
            # Use stream to see intermediate steps
            for step, event in enumerate(app.stream(initial_state, {"recursion_limit": 50})):
                # event is a dictionary where keys are node names and values are outputs
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                logger.info(f"\n=== Step {step+1}: Node '{node_name}' Completed ===")
                logger.info(f"Output from node '{node_name}': {node_output}") # Log full output for debugging
                # You could update a running state dict here if needed, but invoke gives the final state
                # Optional: Pretty logger.info parts of the output
                # if isinstance(node_output, dict) and node_output.get('final_report_markdown'):
                #     logger.info(">>> Final Report Generated (Snippet):")
                #     logger.info(node_output['final_report_markdown'][:500] + "...")

            # Get the final state after streaming is complete
            # Note: Streaming doesn't directly return the final state easily, invoke is better for that.
            # Re-invoke to get the simple final state dict if needed, or parse from stream events.
            # For simplicity in testing, let's invoke again (might re-run if not cached)
            # Alternatively, build final state from stream:
            # final_state_from_stream = {}
            # for step, event in enumerate(app.stream(initial_state, {"recursion_limit": 50})):
            #      final_state_from_stream.update(event[list(event.keys())[0]])
            
            # Let's just use invoke for the final result after logging stream
            logger.info("\nInvoking graph to get final state...")
            final_state = app.invoke(initial_state, {"recursion_limit": 50})

        except Exception as e:
            logger.exception("An error occurred during graph execution:")
            logger.info(f"\nAn error occurred: {e}")
            # final_state might be partially populated or None

        end_time = time.time()
        logger.info(f"\nProcessing finished in {end_time - start_time:.2f} seconds.")

        if final_state:
            # if final_state.get("query_type") == "sector":
            #     if final_state.get("sector_market_data_output"): # Check if the last expected node ran
            #         logger.info(f"\nWorkflow for sector '{final_state.get('sector_name')}' completed up to market data gathering. Full report not yet implemented.")
            #         logger.info("\nCollected Sector Market Data:")
            #         logger.info(final_state.get("sector_market_data_output"))
            #         # Add similar checks for other sector outputs as they are implemented
            #     else:
            #             print(f"\nSector analysis for '{final_state.get('sector_name')}' ended. See logs for details.")
            logger.info("\n--- Final Result ---")
            if final_state.get("final_report_markdown"):
                logger.info("\nGenerated Report:\n")
                logger.info(final_state["final_report_markdown"])
                logger.info("\n--- End of Report ---")
            elif final_state.get("final_sector_report_markdown"):
                logger.info("\nGenerated Report:\n")
                logger.info(final_state["final_sector_report_markdown"])
                logger.info("\n--- End of Report ---")
            elif final_state.get("main_graph_error_message"):
                 logger.info(f"\nWorkflow failed with error: {final_state['main_graph_error_message']}")
            else:
                 logger.info("\nReport could not be generated. Final state:")
                 logger.info(final_state) # logger.info state if no report
        else:
            logger.info("\nWorkflow did not produce a final state.")

        logger.info("\nEnter another query or type 'exit':")

    logger.info("\nExiting Financial Analysis Agent.")