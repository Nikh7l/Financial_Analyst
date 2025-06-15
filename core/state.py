# state.py
from typing import TypedDict, List, Dict, Any, Optional,Literal
from langchain_core.messages import BaseMessage


class RetrievalSummarizationState(TypedDict):
    company_name: str
    document_urls: Optional[List[str]]
    document_summaries: Optional[Dict[str, str]]
    subgraph_error: Optional[str]
    messages: List[BaseMessage] 


class SentimentSubgraphState(TypedDict, total=False):
    company_name: str 
    messages: List[BaseMessage]
    attempt: int
    max_attempts: int
    sentiment_analysis: Optional[Dict[str, str]]  
    subgraph_error: Optional[str]
    _route_decision: Optional[str]


class StockDataSubgraphState(TypedDict, total=False): # Use total=False for flexibility
    company_name: str # Input
    attempt: int
    max_attempts: int
    messages: List[BaseMessage] 
    stock_data: Optional[Dict[str, Any]] 
    subgraph_error: Optional[str] 
    _route_decision: Optional[str] 


class ReportSummariesPayload(TypedDict, total=False):
    """Data structure for report summaries in MainAgentState."""
    document_summaries: Optional[Dict[str, str]]
    retrieved_document_urls: Optional[List[str]]
    error: Optional[str] 

class SentimentAnalysisPayload(TypedDict, total=False):
    """Data structure for sentiment analysis results in MainAgentState."""
    sentiment_data: Optional[Dict[str, str]] 
    error: Optional[str] 


# --- Stock Data Subgraph State ---
class StockDataPayload(TypedDict, total=False): 
    ticker: Optional[str]
    company_name_from_source: Optional[str]
    current_price: Optional[Any]
    previous_close: Optional[Any]
    currency: Optional[str]
    market_cap: Optional[str]
    pe_ratio_ttm: Optional[Any]
    eps_ttm: Optional[Any]
    error: Optional[str] 

class StockInfoPayload(TypedDict, total=False): 
    """Data structure for stock information in MainAgentState."""
    stock_metrics: Optional[StockDataPayload] 
    source_urls_used: Optional[List[str]] 
    error: Optional[str] 

class CompetitorDetailPayload(TypedDict): 
    name: str
    description: Optional[str]

class CompetitorSubgraphState(TypedDict, total=False):
    company_name: str
    attempt: int
    max_attempts: int
    messages: List[BaseMessage]
    competitors: Optional[List[CompetitorDetailPayload]]
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

class CompetitorAnalysisPayload(TypedDict, total=False):
    """Data structure for competitor analysis results in MainAgentState."""
    competitors_list: Optional[List[CompetitorDetailPayload]]
    error: Optional[str] 

class StockPredictionOutput(TypedDict, total=False):
    """Data structure for the stock prediction output."""
    recommendation: Optional[Literal["BUY", "SELL", "HOLD", "UNCERTAIN"]]
    confidence: Optional[Literal["High", "Medium", "Low", "N/A"]]
    reasoning: Optional[str]
    key_positive_factors: Optional[List[str]]
    key_negative_factors: Optional[List[str]]
    data_limitations: Optional[List[str]]
    error: Optional[str] 

############### SECTOR ANALYSIS ###############
class SectorSentimentSubgraphState(TypedDict, total=False):
    sector_name: str
    messages: List[BaseMessage]
    attempt: int
    max_attempts: int
    sector_sentiment_analysis: Optional[Dict[str, Any]] 
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

class SectorSentimentAnalysisPayload(TypedDict, total=False):
    """Data structure for sector news and sentiment in MainAgentState."""
    analysis_data: Optional[Dict[str, Any]] 
    error: Optional[str]


class SectorKeyPlayerDetail(TypedDict, total=False):
    name: str
    description: Optional[str]
    market_share_estimate: Optional[str] 

class SectorKeyPlayersPayload(TypedDict, total=False):
    key_players: Optional[List[SectorKeyPlayerDetail]]
    source_urls_used: Optional[List[str]] 
    error: Optional[str]

class SectorMarketDataDetail(TypedDict, total=False):
    market_size_estimate: Optional[str]  
    projected_cagr: Optional[str]    
    key_market_segments: Optional[List[str]] 
    key_geographies: Optional[List[str]]   
    primary_growth_drivers: Optional[List[str]] 
    primary_market_challenges: Optional[List[str]] 

class SectorMarketDataPayload(TypedDict, total=False):
    market_data: Optional[SectorMarketDataDetail] 
    source_urls_used: Optional[List[str]] 
    error: Optional[str]

class SectorTrendsInnovationsDetail(TypedDict, total=False):
    major_trends: Optional[List[str]]
    recent_innovations: Optional[List[str]]
    key_challenges: Optional[List[str]]
    emerging_opportunities: Optional[List[str]]

class SectorTrendsInnovationsPayload(TypedDict, total=False):
    trends_data: Optional[SectorTrendsInnovationsDetail]
    source_urls_used: Optional[List[str]] 
    error: Optional[str]

class SectorOutlookDetail(TypedDict, total=False):
    overall_outlook: Optional[Literal["Very Positive", "Positive", "Neutral", "Cautiously Optimistic", "Challenging", "Negative", "Very Negative", "N/A"]]
    outlook_summary: Optional[str]
    key_growth_drivers_summary: Optional[List[str]]
    key_risks_challenges_summary: Optional[List[str]] 
    investment_considerations: Optional[str] 

class SectorOutlookPayload(TypedDict, total=False):
    outlook_data: Optional[SectorOutlookDetail]
    error: Optional[str]

# --- Main AgentState Definition ---
class AgentState(TypedDict, total=False): 
    # --- Initial Inputs & Routing ---
    query: str
    query_type: Optional[Literal["company", "sector", "unknown"]]
    company_name: Optional[str]
    sector_name: Optional[str]

    # --- Output from Company Analysis Node ---
    report_summaries_output: Optional[ReportSummariesPayload]
    sentiment_analysis_output: Optional[SentimentAnalysisPayload]
    stock_data_output: Optional[StockInfoPayload]
    competitor_info_output: Optional[CompetitorAnalysisPayload]
    prediction_output: Optional[StockPredictionOutput]
    final_report_markdown: Optional[str]
    consolidated_source_urls: Optional[List[str]]

    # --- Output from Sector Analysis Node ---
    sector_news_sentiment_output: Optional[SectorSentimentAnalysisPayload]
    sector_key_players_output: Optional[SectorKeyPlayersPayload]
    sector_market_data_output: Optional[SectorMarketDataPayload]
    sector_trends_innovations_output: Optional[SectorTrendsInnovationsPayload]
    sector_outlook_output: Optional[SectorOutlookPayload]
    final_sector_report_markdown: Optional[str]
    sector_report_urls: Optional[List[str]] # URLs used for the sector report

    # --- General error for the main graph if something goes wrong outside a subgraph ---
    main_graph_error_message: Optional[str]