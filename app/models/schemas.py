from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


# Base Models
class DeckRequest(BaseModel):
    leader_name: Optional[str] = Field(None, description="Name of the leader card")
    key_cards: List[str] = Field(default_factory=list, description="List of key cards to include")
    aspects: List[str] = Field(default_factory=list, description="List of aspects/colors to focus on")
    alignment: Optional[str] = Field(None, description="Heroism, Villainy, or Both")


class CostCurveTarget(BaseModel):
    cost_2_or_less: int = Field(default=12, description="Number of cards costing 2 or less")
    cost_3: int = Field(default=12, description="Number of cards costing 3")
    cost_4: int = Field(default=10, description="Number of cards costing 4")
    cost_5: int = Field(default=8, description="Number of cards costing 5")
    cost_6_plus: int = Field(default=8, description="Number of cards costing 6 or more")


# Request Models
class StrategyRequest(BaseModel):
    query: str = Field(..., description="The strategy question to ask")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional chat history for context"
    )


class CardSearchRequest(BaseModel):
    type: Optional[str] = None
    cost: Optional[int] = None
    aspects: Optional[List[str]] = None
    traits: Optional[List[str]] = None
    arena: Optional[str] = None


class DeckAnalysisRequest(BaseModel):
    deck_ids: List[str] = Field(..., description="List of card IDs in the deck")


class DeckBuildRequest(BaseModel):
    leader_name: Optional[str] = None
    key_cards: Optional[List[str]] = None
    aspects: Optional[List[str]] = None


class DeckBuildingQuery(BaseModel):
    query: str = Field(..., description="Natural language deck building request")


class SynergyRequest(BaseModel):
    card_name: str


# Response Models
class Source(BaseModel):
    title: str
    page: str
    content: str


class StrategyResponse(BaseModel):
    answer: str
    sources: List[Source]


class CardModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[str] = Field(None, alias="_id")
    name: str
    type: str
    cost: str
    aspects: List[str]
    traits: List[str]
    arena: Optional[str]
    description: Optional[str]
    rarity: Optional[str]


class CardSearchResponse(BaseModel):
    cards: List[CardModel]


class DeckAnalysisStats(BaseModel):
    id: str = Field(alias="_id")
    count: int


class DeckAnalysisDistribution(BaseModel):
    cost_curve: List[DeckAnalysisStats]
    type_distribution: List[DeckAnalysisStats]
    aspect_distribution: List[DeckAnalysisStats]


# API Response Models
class HealthCheckResponse(BaseModel):
    status: str
    cards_loaded: int
    strategy_docs_loaded: bool


class ErrorResponse(BaseModel):
    detail: str


# Deck Building Response Models
class DeckBuildingResult(BaseModel):
    leader: Optional[CardModel]
    bases: List[CardModel]
    key_cards: List[CardModel]
    recommended_cards: List[CardModel]
    aspects: List[str]
    deck_analysis: Dict[str, Any]