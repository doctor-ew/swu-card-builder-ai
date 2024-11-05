# /app/main.py

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from . import create_search_service
from .models.schemas import (
    StrategyRequest, StrategyResponse,
    CardSearchRequest, CardSearchResponse,
    DeckAnalysisRequest, DeckAnalysisDistribution,
    DeckBuildRequest, DeckBuildingQuery, DeckBuildingResult,
    SynergyRequest, HealthCheckResponse, ErrorResponse
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Initialize metrics registry
registry = CollectorRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize search service
        app.state.search = create_search_service()

        # Verify initialization
        card_count = app.state.search.cards.count_documents({})
        logger.info(f"Initialized with {card_count} cards")

        # Test specific card lookup
        test_card = app.state.search.cards.find_one({"name": "Hunter, Outcast Sergeant"})
        logger.debug(f"Test card lookup: {test_card is not None}")

    except Exception as e:
        logger.error(f"Failed to initialize search service: {e}")
        raise

    yield

    if hasattr(app.state, 'search'):
        app.state.search.client.close()


# Create FastAPI app
app = FastAPI(
    title="Star Wars Unlimited API",
    description="API for Star Wars Unlimited card search and strategy advice",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/metrics")
async def metrics():
    """Endpoint to expose Prometheus metrics"""
    return Response(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health_check():
    """Check the health of the API and its dependencies"""
    try:
        card_count = app.state.search.cards.count_documents({})
        has_strategy = app.state.search.qa_chain is not None

        return {
            "status": "healthy",
            "cards_loaded": card_count,
            "strategy_docs_loaded": has_strategy
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/strategy/advice", response_model=StrategyResponse)
async def get_strategy_advice(request: StrategyRequest) -> StrategyResponse:
    """Get strategy advice for Star Wars Unlimited"""
    try:
        response = app.state.search.get_strategy_advice(
            query=request.query,
            chat_history=request.chat_history
        )
        return StrategyResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cards/search", response_model=CardSearchResponse)
async def search_cards(filters: CardSearchRequest) -> CardSearchResponse:
    """Search for cards based on various criteria"""
    try:
        cards = app.state.search.search_cards(filters.dict(exclude_none=True))
        return CardSearchResponse(cards=cards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deck/analyze", response_model=DeckAnalysisDistribution)
async def analyze_deck(request: DeckAnalysisRequest) -> DeckAnalysisDistribution:
    """Analyze deck composition"""
    try:
        analysis = app.state.search.analyze_deck_composition(request.deck_ids)
        return DeckAnalysisDistribution(**analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deck/build/natural", response_model=DeckBuildingResult)
async def build_deck_natural(request: DeckBuildingQuery) -> DeckBuildingResult:
    """Build a deck from natural language request"""
    try:
        logger.debug(f"|-o-| Received deck build request: {request.query}")
        result = app.state.search.build_deck_around_cards(query=request.query)
        logger.debug(f"|-o-| Deck build result: {result}")
        return DeckBuildingResult(**result)
    except Exception as e:
        logger.error(f"Error building deck: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Check the health of the API and its dependencies"""
    try:
        card_count = app.state.search.cards.count_documents({})
        has_strategy = app.state.search.qa_chain is not None

        return HealthCheckResponse(
            status="healthy",
            cards_loaded=card_count,
            strategy_docs_loaded=has_strategy
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


# Only if running directly (not through uvicorn command)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
