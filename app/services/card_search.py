from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

from ..models.schemas import DeckRequest, CostCurveTarget
from ..models.constants import MECHANIC_PATTERNS, ASPECT_COLORS, CARD_NAME_VARIATIONS
from ..utils.exceptions import CardSearchError, CardNotFoundError, InvalidQueryError
from .logger import SearchLogger
from .validator import SearchResultValidator
from .analyzer import CardRelationshipAnalyzer
from ..utils.metrics import monitor_method, monitor_db
from ..utils.monitoring import monitor_method, monitor_db


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class StarWarsUnlimitedSearch:
    def __init__(self, mongodb_uri: str = "mongodb://localhost:27017/",
                 docs_dir: str = "./data",
                 persist_dir: str = "./chroma_db",
                 cards_json: str = "./data/swudb_card_details.json",
                 openai_api_key: str = None):

        # Add this after your other initializations
        self.deck_request_template = """
        Extract the following from this Star Wars Unlimited deck request:
        - Leader card (if mentioned)
        - Key cards to include
        - Aspects/colors desired
        - Any specific card types (units, events, bases)

        Use exact card names where possible. Current request: {query}

        Return in this format:
        LEADER: [exact leader name or 'None']
        CARDS: [comma-separated list of card names]
        ASPECTS: [comma-separated list of aspects]
        """
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir)
        self.cards_json = Path(cards_json)

        # Initialize services
        self.logger = SearchLogger()
        self.validator = SearchResultValidator()
        self.analyzer = CardRelationshipAnalyzer()

        # MongoDB setup
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.star_wars_unlimited
        self.cards = self.db.cards

        # LLM setup
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key
        )

        # Initialize components
        self._create_indexes()
        self._load_cards()

        # Vector store setup
        self.strategy_store = None
        self.qa_chain = None
        if self.docs_dir.exists():
            self.load_strategy_docs()

        self.cost_curve = CostCurveTarget()

        self.logger.debug("|-o-| StarWarsUnlimitedSearch initialized.")

    def _get_card_alignment(self, card: Dict) -> str:
        """Determine if a card is Heroism, Villainy, or Neither"""
        aspects = card.get("aspects", [])
        if "Heroism" in aspects:
            return "Heroism"
        elif "Villainy" in aspects:
            return "Villainy"
        return "Neutral"

    def _create_indexes(self):
        """Create MongoDB indexes for improved query performance"""
        try:
            # Create indexes for commonly queried fields
            self.cards.create_index([("name", 1)])  # Index for name searches
            self.cards.create_index([("type", 1)])  # Index for card type queries
            self.cards.create_index([("aspects", 1)])  # Index for aspect queries
            self.cards.create_index([("traits", 1)])  # Index for trait queries
            self.cards.create_index([("cost", 1)])  # Index for cost queries

            self.logger.info("Created MongoDB indexes")

        except Exception as e:
            self.logger.error(f"Error creating MongoDB indexes: {e}")
            raise

    def _validate_aspects(self, card_aspects: List[str], deck_aspects: List[str]) -> bool:
        """Check if card aspects are compatible with deck aspects"""
        # Remove alignment aspects for comparison
        card_colors = [a for a in card_aspects if a not in ["Heroism", "Villainy"]]
        deck_colors = [a for a in deck_aspects if a not in ["Heroism", "Villainy"]]

        # If deck has no colors yet, card is valid
        if not deck_colors:
            return True

        # Card must share at least one color with deck
        return any(color in deck_colors for color in card_colors)

    def _parse_cost(self, cost_value: Any) -> int:
        """Safely parse a cost value to integer"""
        try:
            # Early return for bases and cards without cost
            if not cost_value or str(cost_value).strip() == '':
                return 0

            if isinstance(cost_value, int):
                return cost_value

            # Try to parse the string value
            cost_str = str(cost_value).strip()
            return int(cost_str)

        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not parse cost value: {cost_value}, defaulting to 0. Error: {e}")
            return 0  # Default to 0 for unparseable values

    def _analyze_cost_curve(self, cards: List[Dict]) -> Dict[str, int]:
        """Analyze the cost curve of a list of cards"""
        curve = {
            "2_or_less": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6_plus": 0
        }

        for card in cards:
            cost = self._parse_cost(card.get("cost", "0"))
            if cost <= 2:
                curve["2_or_less"] += 1
            elif cost == 3:
                curve["3"] += 1
            elif cost == 4:
                curve["4"] += 1
            elif cost == 5:
                curve["5"] += 1
            else:
                curve["6_plus"] += 1

        return curve

    def _optimize_deck_composition(self, core_cards: List[Dict], available_cards: List[Dict]) -> List[Dict]:
        """Optimize deck composition based on cost curve and strategy"""
        deck = core_cards.copy()
        target_size = 50  # Minimum deck size

        # Calculate remaining slots
        remaining_slots = target_size - len(deck)

        # Group available cards by cost
        cards_by_cost = {
            "2_or_less": [],
            "3": [],
            "4": [],
            "5": [],
            "6_plus": []
        }

        for card in available_cards:
            cost = self._parse_cost(card.get("cost", "0"))
            if cost <= 2:
                cards_by_cost["2_or_less"].append(card)
            elif cost == 3:
                cards_by_cost["3"].append(card)
            elif cost == 4:
                cards_by_cost["4"].append(card)
            elif cost == 5:
                cards_by_cost["5"].append(card)
            else:
                cards_by_cost["6_plus"].append(card)

        # Fill according to target curve
        current_curve = self._analyze_cost_curve(deck)

        for cost_bracket in cards_by_cost.values():
            cost_bracket.sort(key=lambda x: x.get("name", ""))  # Sort by name for consistency
        for cost_range, target_count in {
            "2_or_less": self.cost_curve.cost_2_or_less,
            "3": self.cost_curve.cost_3,
            "4": self.cost_curve.cost_4,
            "5": self.cost_curve.cost_5,
            "6_plus": self.cost_curve.cost_6_plus
        }.items():
            needed = min(target_count - current_curve[cost_range], remaining_slots)
            if needed > 0:
                available = cards_by_cost[cost_range][:needed]
                deck.extend(available)
                remaining_slots -= len(available)

        return deck

    def load_strategy_docs(self):
        """Load all PDF strategy documents from the docs directory"""
        try:
            if not self.docs_dir.exists():
                raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

            pdf_path = self.docs_dir / "pdfs"  # Adjust if your PDFs are in a different subdirectory
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF directory not found: {pdf_path}")

            self.logger.info(f"Loading documents from {pdf_path}")

            # Check for PDF files
            pdf_files = list(pdf_path.glob("*.pdf"))
            if not pdf_files:
                self.logger.warning("No PDF files found in documents directory")
                return

            loader = DirectoryLoader(
                str(pdf_path),
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )

            documents = loader.load()
            if not documents:
                self.logger.warning("No content loaded from PDF files")
                return

            self.logger.info(f"Loaded {len(documents)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            splits = text_splitter.split_documents(documents)
            if not splits:
                self.logger.warning("No text chunks created from documents")
                return

            self.logger.info(f"Created {len(splits)} text chunks")

            # Initialize embeddings
            embeddings = OpenAIEmbeddings()

            # Create vector store
            if self.persist_dir.exists():
                # Try to load existing store
                try:
                    self.logger.info("Attempting to load existing vector store")
                    self.strategy_store = Chroma(
                        persist_directory=str(self.persist_dir),
                        embedding_function=embeddings
                    )
                    self.logger.info("Successfully loaded existing vector store")
                except Exception as e:
                    self.logger.warning(f"Failed to load existing store, creating new one: {e}")
                    self.strategy_store = None

            # Create new store if needed
            if not self.strategy_store:
                self.logger.info("Creating new vector store")
                self.strategy_store = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=str(self.persist_dir)
                )
                self.logger.info("Vector store created successfully")

            # Initialize retriever
            retriever = self.strategy_store.as_retriever(
                search_kwargs={"k": 4}
            )

            # Initialize QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                max_tokens_limit=4000
            )

            self.logger.info("QA chain initialized successfully")

        except Exception as e:
            self.logger.error(f"Error loading strategy documents: {e}", exc_info=True)
            # Don't raise the exception - allow the service to start without strategy docs
            self.strategy_store = None
            self.qa_chain = None

    def process_natural_language_request(self, query: str) -> DeckRequest:
        """Convert natural language deck request into structured format"""
        self.logger.debug(f"|-o-| 1. Processing natural language request: {query}")
        try:
            self.logger.debug(f"|-o-| 2. Processing natural language request: {query}")
            # Get structured response from LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Star Wars Unlimited TCG deck building assistant. 
                    Extract card information precisely. Return ONLY the structured data in this format:
                    LEADER: [exact card name]
                    CARDS: [comma-separated list of exact card names]
                    ASPECTS: [comma-separated list of aspects]"""
                },
                {
                    "role": "user",
                    "content": f"Extract deck building information from this request: {query}"
                }
            ]

            response = self.llm.invoke(messages)
            self.logger.debug(f"LLM Response: {response.content}")

            # Parse the response
            leader_name = None
            key_cards = []
            aspects = []

            for line in response.content.split('\n'):
                line = line.strip()
                if line.startswith('LEADER:'):
                    leader = line.replace('LEADER:', '').strip()
                    if leader.lower() != 'none':
                        leader_name = leader
                        self.logger.debug(f"Extracted leader: {leader_name}")

                elif line.startswith('CARDS:'):
                    cards = line.replace('CARDS:', '').strip()
                    if cards.lower() != 'none':
                        key_cards = [card.strip() for card in cards.split(',') if card.strip()]
                        self.logger.debug(f"Extracted cards: {key_cards}")

                elif line.startswith('ASPECTS:'):
                    aspect_list = line.replace('ASPECTS:', '').strip()
                    if aspect_list.lower() != 'none':
                        aspects = [aspect.strip() for aspect in aspect_list.split(',') if aspect.strip()]
                        self.logger.debug(f"Extracted aspects: {aspects}")

            result = DeckRequest(
                leader_name=leader_name,
                key_cards=key_cards,
                aspects=aspects
            )

            self.logger.info(f"Processed request into: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error processing deck request: {e}")
            raise InvalidQueryError(f"Failed to process deck request: {str(e)}")

    def _calculate_aspect_cost(self, card: Dict, deck_aspects: List[str], bases: List[Dict]) -> int:
        """Calculate the actual resource cost for a card considering aspects and bases"""
        # Bases have no cost
        if card.get("type") == "Base":
            return 0

        card_aspects = [a for a in card.get("aspects", []) if a not in ["Heroism", "Villainy"]]
        base_aspects = []
        for base in bases:
            base_aspects.extend([a for a in base.get("aspects", []) if a not in ["Heroism", "Villainy"]])

        available_aspects = deck_aspects + base_aspects

        # Get base cost, defaulting to 0 if not present or invalid
        base_cost = self._parse_cost(card.get("cost", "0"))

        # If card shares any aspect with available aspects, no penalty
        if any(aspect in available_aspects for aspect in card_aspects):
            return base_cost

        # Otherwise, +2 resource penalty
        return base_cost + 2

    def _analyze_deck_efficiency(self, cards: List[Dict], bases: List[Dict]) -> Dict:
        """Analyze how efficiently cards can be played"""
        analysis = {
            "on_curve_cards": [],
            "penalty_cards": [],
            "base_enabled_cards": [],
            "aspect_distribution": {},
            "total_base_aspects": []
        }

        # Collect all available aspects
        deck_aspects = []
        for card in cards:
            deck_aspects.extend([a for a in card.get("aspects", []) if a not in ["Heroism", "Villainy"]])
        deck_aspects = list(set(deck_aspects))

        # Add base aspects
        for base in bases:
            analysis["total_base_aspects"].extend(
                [a for a in base.get("aspects", []) if a not in ["Heroism", "Villainy"]]
            )

        # Analyze each card
        for card in cards:
            card_aspects = [a for a in card.get("aspects", []) if a not in ["Heroism", "Villainy"]]
            base_enabled = False

            # Check if card is enabled by a base
            if any(aspect in analysis["total_base_aspects"] for aspect in card_aspects):
                base_enabled = True
                analysis["base_enabled_cards"].append(card)

            # Calculate effective cost
            effective_cost = self._calculate_aspect_cost(card, deck_aspects, bases)

            if effective_cost > int(card.get("cost", "0")):
                analysis["penalty_cards"].append({
                    "card": card,
                    "base_cost": int(card.get("cost", "0")),
                    "effective_cost": effective_cost,
                    "enabled_by_base": base_enabled
                })
            else:
                analysis["on_curve_cards"].append(card)

            # Track aspect distribution
            for aspect in card_aspects:
                analysis["aspect_distribution"][aspect] = analysis["aspect_distribution"].get(aspect, 0) + 1

        return analysis

    @monitor_method("normalize_card_name")
    def _normalize_card_name(self, name: str) -> str:
        """Normalize card name with monitoring"""
        context = {"method": "_normalize_card_name", "original_name": name}

        try:
            if not name:
                raise ValueError("Card name cannot be empty")

            # Remove special characters and lowercase
            normalized = ''.join(c.lower() for c in name if c.isalnum() or c.isspace())

            # Apply spelling fixes
            for variant, correct in CARD_NAME_VARIATIONS.items():
                normalized = normalized.replace(variant, correct)

            self.logger.debug(
                "Normalized card name",
                extra={**context, "normalized": normalized}
            )

            return normalized.strip()

        except Exception as e:
            self.logger.error(
                "Failed to normalize card name",
                extra={**context, "error": str(e)},
                exc_info=True
            )
            raise

    @monitor_method("find_card")
    @monitor_method("find_card")
    def _find_card(self, name: str, card_type: str = None) -> Optional[Dict]:
        """Find a card with flexible name matching"""
        try:
            self.logger.debug(f"Searching for card - Name: {name}, Type: {card_type}")

            if not name:
                return None

            # Try exact match first
            query = {"name": {"$regex": f"^{name}$", "$options": "i"}}
            if card_type:
                query["type"] = card_type

            self.logger.debug(f"Trying exact match with query: {query}")
            card = self.cards.find_one(query)

            if card:
                self.logger.debug(f"Found exact match: {card['name']}")
                return self._serialize_mongo_doc(card)

            # Try contains match
            query = {"name": {"$regex": f".*{name}.*", "$options": "i"}}
            if card_type:
                query["type"] = card_type

            self.logger.debug(f"Trying partial match with query: {query}")
            cards = list(self.cards.find(query))

            if cards:
                self.logger.debug(f"Found {len(cards)} partial matches")
                # Sort by name length to prefer closer matches
                cards.sort(key=lambda x: len(x["name"]))
                self.logger.debug(f"Best partial match: {cards[0]['name']}")
                return self._serialize_mongo_doc(cards[0])

            self.logger.debug(f"No matches found for {name}")
            return None

        except Exception as e:
            self.logger.error(f"Error finding card '{name}': {e}")
            return None

    def _find_bases_for_aspects(self, aspects: List[str]) -> List[Dict]:
        """Find bases that provide specific aspects"""
        try:
            query = {
                "type": "Base",
                "aspects": {"$in": aspects}
            }

            bases = list(self.cards.find(query))
            return [self._serialize_mongo_doc(base) for base in bases]

        except Exception as e:
            self.logger.error(f"Error finding bases for aspects {aspects}: {e}")
            return []

    def _find_complementary_cards(self, core_cards: List[Dict], bases: List[Dict],
                                  deck_aspects: List[str], efficiency_analysis: Dict) -> List[Dict]:
        """Find complementary cards for the deck considering aspects and synergies"""
        try:
            # Extract traits and mechanics from core cards
            core_traits = set()
            core_mechanics = set()
            for card in core_cards:
                core_traits.update(card.get("traits", []))
                # Extract keywords from description
                desc = card.get("description", "").lower()
                for keyword in ["attack", "defend", "draw", "resource", "damage", "shield"]:
                    if keyword in desc:
                        core_mechanics.add(keyword)

            # Calculate available aspects (including bases)
            available_aspects = set(deck_aspects)
            for base in bases:
                available_aspects.update([a for a in base.get("aspects", []) if a not in ["Heroism", "Villainy"]])

            # Build query for complementary cards
            query = {
                "type": {"$ne": "Leader"},  # Exclude leaders
                "$or": [
                    # Cards matching our aspects
                    {"aspects": {"$in": list(available_aspects)}},
                    # Cards sharing traits with our core cards
                    {"traits": {"$in": list(core_traits)}}
                ]
            }

            # Get potential complementary cards
            potential_cards = list(self.cards.find(query))
            scored_cards = []

            for card in potential_cards:
                try:
                    score = 0
                    card_cost = self._parse_cost(card.get("cost", "0"))

                    # Base score based on cost effectiveness
                    effective_cost = self._calculate_aspect_cost(card, deck_aspects, bases)
                    if effective_cost == card_cost:
                        score += 3  # Bonus for native aspect cards
                    elif effective_cost <= card_cost + 2:
                        score += 1  # Still usable with penalty

                    # Rest of the scoring logic...
                    # [Previous scoring code remains the same]

                    scored_cards.append((card, score))
                except Exception as card_error:
                    self.logger.warning(f"Error scoring card {card.get('name')}: {card_error}")
                    continue

            # Sort by score and return top cards
            scored_cards.sort(key=lambda x: x[1], reverse=True)

            return [{
                **self._serialize_mongo_doc(card),
                "synergy_score": score,
                "effective_cost": self._calculate_aspect_cost(card, deck_aspects, bases)
            } for card, score in scored_cards[:30]]  # Return top 30 cards

        except Exception as e:
            self.logger.error(f"Error finding complementary cards: {e}")
            raise

    def _find_complementary_bases(self, deck_aspects: List[str], key_cards: List[str] = None) -> List[Dict]:
        """Find bases that would complement the deck's strategy"""
        try:
            # Get aspects needed by key cards
            needed_aspects = set()
            if key_cards:
                for card_name in key_cards:
                    card = self._find_card(card_name)
                    if card:
                        card_aspects = [a for a in card.get("aspects", []) if a not in ["Heroism", "Villainy"]]
                        needed_aspects.update(card_aspects)

            # Remove aspects we already have
            needed_aspects = needed_aspects - set(deck_aspects)

            # Find bases that provide needed aspects
            query = {
                "type": "Base",
                "$or": [
                    {"aspects": {"$in": list(needed_aspects)}},  # Bases with needed aspects
                    {"aspects": {"$in": deck_aspects}}  # Bases that complement existing aspects
                ]
            }

            bases = list(self.cards.find(query))
            scored_bases = []

            for base in bases:
                score = 0
                base_aspects = [a for a in base.get("aspects", []) if a not in ["Heroism", "Villainy"]]

                # Score based on needed aspects
                needed_aspect_matches = len(set(base_aspects).intersection(needed_aspects))
                score += needed_aspect_matches * 3

                # Score based on existing aspect synergy
                existing_aspect_matches = len(set(base_aspects).intersection(deck_aspects))
                score += existing_aspect_matches

                scored_bases.append((base, score))

            # Sort by score
            scored_bases.sort(key=lambda x: x[1], reverse=True)

            # Return serialized bases with scores
            return [{
                **self._serialize_mongo_doc(base),
                "synergy_score": score
            } for base, score in scored_bases]

        except Exception as e:
            self.logger.error(f"Error finding complementary bases: {e}")
            raise

    @monitor_method("build_deck")
    def build_deck_around_cards(self, query: str = None, leader_name: str = None,
                                key_cards: List[str] = None, aspects: List[str] = None) -> Dict:
        """Build a deck based on natural language query or specific parameters"""
        start_time = time.time()
        method_context = {
            "query": query,
            "leader_name": leader_name,
            "key_cards": key_cards,
            "aspects": aspects
        }

        suggestions = {
            "recommended_bases": [],
            "synergy_notes": [],
            "aspect_recommendations": []
        }

        try:
            self.logger.debug("Starting deck build", extra={"context": method_context})

            # Process natural language if provided
            if query:
                request = self.process_natural_language_request(query)
                self.logger.debug("Processed natural language request",
                                  extra={"request": str(request)})
                leader_name = request.leader_name
                key_cards = request.key_cards
                aspects = request.aspects

                self.logger.debug(
                    "After processing request",
                    extra={
                        "leader": leader_name,
                        "cards": key_cards,
                        "aspects": aspects
                    }
                )

            selected_cards = []
            bases = []
            deck_aspects = []

            # Find and validate leader
            if leader_name:
                leader = self._find_card(leader_name, card_type="Leader")
                if not leader:
                    self.logger.error(f"Leader not found: {leader_name}")
                    raise CardNotFoundError(f"Leader '{leader_name}' not found")

                deck_aspects.extend([a for a in leader["aspects"]
                                     if a not in ["Heroism", "Villainy"]])
                selected_cards.append(leader)

                # Suggest complementary bases
                potential_bases = self._find_complementary_bases(deck_aspects, key_cards)
                suggestions["recommended_bases"].extend(potential_bases)

                if potential_bases:
                    suggestions["synergy_notes"].append(
                        f"Recommended bases to support your strategy: {', '.join(base['name'] for base in potential_bases[:3])}"
                    )

                self.logger.info(
                    f"Found leader: {leader['name']}",
                    extra={
                        "leader_aspects": deck_aspects,
                        "leader_alignment": self._get_card_alignment(leader)
                    }
                )

            # Process key cards
            if key_cards:
                for card_name in key_cards:
                    card = self._find_card(card_name)
                    if not card:
                        self.logger.warning(f"Key card not found: {card_name}")
                        continue

                    if card["type"] == "Base":
                        bases.append(card)
                        continue

                    # Calculate effective cost
                    effective_cost = self._calculate_aspect_cost(card, deck_aspects, bases)
                    card_aspects = [a for a in card["aspects"] if a not in ["Heroism", "Villainy"]]

                    # If card would need a penalty, suggest enabling bases
                    base_cost = self._parse_cost(card.get("cost", "0"))
                    if effective_cost > base_cost:
                        missing_aspects = [a for a in card_aspects if a not in deck_aspects]
                        suggestions["aspect_recommendations"].append({
                            "card": card["name"],
                            "base_cost": base_cost,
                            "effective_cost": effective_cost,
                            "missing_aspects": missing_aspects,
                            "suggested_bases": self._find_bases_for_aspects(missing_aspects)
                        })

                    selected_cards.append(card)
                    deck_aspects.extend([a for a in card_aspects if a not in deck_aspects])

            # Analyze deck efficiency
            efficiency_analysis = self._analyze_deck_efficiency(selected_cards, bases)

            # Build remaining deck considering aspect penalties
            complementary_cards = self._find_complementary_cards(
                selected_cards,
                bases,
                deck_aspects,
                efficiency_analysis
            )

            deck_result = {
                "leader": selected_cards[0] if selected_cards else None,
                "bases": bases,
                "key_cards": [c for c in selected_cards[1:] if c["type"] != "Base"],
                "recommended_cards": complementary_cards,
                "aspects": list(set(deck_aspects)),  # Deduplicate aspects
                "deck_analysis": {
                    "efficiency": efficiency_analysis,
                    "suggestions": suggestions,
                    "curve_analysis": self._analyze_cost_curve(complementary_cards),
                    "aspect_costs": {
                        "native_aspects": list(set(deck_aspects)),
                        "base_enabled": list(set(efficiency_analysis["total_base_aspects"])),
                        "penalty_cards": len(efficiency_analysis["penalty_cards"])
                    }
                }
            }

            duration = time.time() - start_time
            self.logger.info(
                "Completed deck building",
                extra={
                    "duration": f"{duration:.2f}s",
                    "total_cards": len(selected_cards),
                    "aspects_used": deck_aspects
                }
            )

            return deck_result

        except Exception as e:
            self.logger.error(
                "Error building deck",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "method": "build_deck_around_cards"
                }
            )
            if isinstance(e, (CardNotFoundError, InvalidQueryError)):
                raise
            raise CardSearchError(f"Error building deck: {str(e)}")

    def _serialize_mongo_doc(self, doc: Dict) -> Dict:
        """Serialize MongoDB document to JSON-compatible format"""
        return json.loads(JSONEncoder().encode(doc))

    def _load_cards(self):
        """Load cards from JSON into MongoDB"""
        try:
            if not self.cards_json.exists():
                self.logger.error(f"Cards JSON file not found: {self.cards_json}")
                return

            with open(self.cards_json, 'r') as f:
                cards_data = json.load(f)
                self.logger.debug(f"Loaded {len(cards_data)} cards from JSON")

            # Clear existing cards and insert new ones
            self.cards.delete_many({})
            if isinstance(cards_data, list):
                self.cards.insert_many(cards_data)
                count = self.cards.count_documents({})
                self.logger.info(f"Loaded {count} cards into MongoDB")

                # Verify some specific cards
                test_cards = ["Hunter, Outcast Sergeant", "Wrecker", "Marauder"]
                for card in test_cards:
                    result = self.cards.find_one({"name": {"$regex": f".*{card}.*", "$options": "i"}})
                    self.logger.debug(f"Test card '{card}': {'Found' if result else 'Not found'}")
            else:
                self.logger.error("Invalid cards data format")

        except Exception as e:
            self.logger.error(f"Error loading cards: {e}")
            raise

    @monitor_method("card_search")
    def search_cards(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fast attribute-based card search using MongoDB"""
        cursor = self.cards.find(filters)
        return [self._serialize_mongo_doc(doc) for doc in cursor]

    def analyze_deck_composition(self, deck_ids: List[str]) -> Dict[str, Any]:
        """Analyze deck using MongoDB aggregation"""
        object_ids = [ObjectId(id_) if isinstance(id_, str) else id_ for id_ in deck_ids]
        pipeline = [
            {"$match": {"_id": {"$in": object_ids}}},
            {"$facet": {
                "cost_curve": [
                    {"$group": {
                        "_id": "$cost",
                        "count": {"$sum": 1}
                    }}
                ],
                "type_distribution": [
                    {"$group": {
                        "_id": "$type",
                        "count": {"$sum": 1}
                    }}
                ],
                "aspect_distribution": [
                    {"$unwind": "$aspects"},
                    {"$group": {
                        "_id": "$aspects",
                        "count": {"$sum": 1}
                    }}
                ]
            }}
        ]
        result = self.cards.aggregate(pipeline).next()
        return self._serialize_mongo_doc(result)
