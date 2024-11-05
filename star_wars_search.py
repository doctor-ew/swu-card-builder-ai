# star_wars_search.py
from typing import List, Dict, Any
from pathlib import Path
import json
import time
import itertools
import re
from bson import ObjectId
from datetime import datetime
import logging
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper Classes
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)




class SearchLogger:
    def __init__(self):
        self.logger = logging.getLogger("card_search")
        self.logger.setLevel(logging.DEBUG)

        # File handler for detailed logging
        fh = logging.FileHandler("card_search.log")
        fh.setLevel(logging.DEBUG)

        # Console handler for basic logging
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        fh.setFormatter(detailed_formatter)
        ch.setFormatter(simple_formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_search(self, query: Dict, results: List[Dict], duration: float):
        """Log search operation details"""
        self.logger.info(f"Search completed in {duration:.2f}s")
        self.logger.debug(f"Query: {query}")
        self.logger.debug(f"Found {len(results)} results")

    def log_card_relationship(self, card: Dict, related_cards: List[Dict]):
        """Log card relationship analysis"""
        self.logger.debug(
            f"Analyzed relationships for {card['name']}: "
            f"Found {len(related_cards)} related cards"
        )

    def log_error(self, error: Exception, context: Dict = None):
        """Log error with context"""
        self.logger.error(
            f"Error: {str(error)}, Context: {context}",
            exc_info=True
        )


class CardRelationshipAnalyzer:
    """Analyzes and tracks relationships between cards"""

    def __init__(self):
        self.relationship_cache = {}
        self.trait_groups = {}
        self.mechanic_patterns = {
            "damage": [r"deal.*damage", r"damage", r"destroy"],
            "control": [r"capture", r"return.*to.*hand", r"discard"],
            "resource": [r"resource", r"generate", r"gain"],
            "defense": [r"shield", r"protect", r"defend"],
            "combat": [r"attack", r"combat", r"fight"],
            "support": [r"draw.*card", r"search", r"reveal"]
        }

    def analyze_card_relationships(self, card: Dict, card_pool: List[Dict]) -> Dict[str, Any]:
        """Analyze how a card relates to other cards"""
        cache_key = str(card.get("_id"))
        if cache_key in self.relationship_cache:
            return self.relationship_cache[cache_key]

        relationships = {
            "trait_synergies": [],
            "aspect_synergies": [],
            "mechanic_synergies": [],
            "cost_synergies": [],
            "primary_mechanics": set(),
            "relationship_score": {}
        }

        # Analyze card mechanics
        card_desc = card.get("description", "").lower()
        for mechanic, patterns in self.mechanic_patterns.items():
            if any(re.search(pattern, card_desc) for pattern in patterns):
                relationships["primary_mechanics"].add(mechanic)

        # Score relationships with other cards
        for other_card in card_pool:
            if other_card.get("_id") == card.get("_id"):
                continue

            score = 0
            reasons = []

            # Trait synergy
            shared_traits = set(card.get("traits", [])) & set(other_card.get("traits", []))
            if shared_traits:
                score += len(shared_traits) * 2
                reasons.append(f"Shared traits: {', '.join(shared_traits)}")

            # Aspect synergy
            shared_aspects = set(card.get("aspects", [])) & set(other_card.get("aspects", []))
            if shared_aspects:
                score += len(shared_aspects)
                reasons.append(f"Shared aspects: {', '.join(shared_aspects)}")

            # Mechanic synergy
            other_desc = other_card.get("description", "").lower()
            shared_mechanics = set()
            for mechanic in relationships["primary_mechanics"]:
                if any(re.search(pattern, other_desc) for pattern in self.mechanic_patterns[mechanic]):
                    shared_mechanics.add(mechanic)
            if shared_mechanics:
                score += len(shared_mechanics) * 1.5
                reasons.append(f"Complementary mechanics: {', '.join(shared_mechanics)}")

            # Cost curve consideration
            cost_diff = abs(int(card.get("cost", "0")) - int(other_card.get("cost", "0")))
            if cost_diff <= 1:
                score += 1
                reasons.append("Complementary cost")

            if score > 0:
                relationships["relationship_score"][other_card["name"]] = {
                    "score": score,
                    "reasons": reasons
                }

        # Sort and categorize relationships
        scored_relationships = sorted(
            relationships["relationship_score"].items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        # Store in cache
        self.relationship_cache[cache_key] = {
            "primary_mechanics": list(relationships["primary_mechanics"]),
            "top_synergies": [
                {
                    "card_name": card_name,
                    "score": details["score"],
                    "reasons": details["reasons"]
                }
                for card_name, details in scored_relationships[:10]  # Top 10 synergies
            ]
        }

        return self.relationship_cache[cache_key]

    def get_card_group_suggestions(self, card: Dict, trait_threshold: int = 2) -> List[str]:
        """Dynamically suggest card groups based on traits and mechanics"""
        card_traits = set(card.get("traits", []))
        suggestions = []

        # Check existing trait groups
        for trait_combo, cards in self.trait_groups.items():
            if len(card_traits & set(trait_combo.split('+'))) >= trait_threshold:
                suggestions.append(f"{' '.join(trait_combo.split('+'))} Group")

        # Look for new trait combinations
        trait_list = list(card_traits)
        for r in range(2, len(trait_list) + 1):
            for combo in itertools.combinations(trait_list, r):
                combo_key = '+'.join(sorted(combo))
                if combo_key not in self.trait_groups:
                    self.trait_groups[combo_key] = set()
                self.trait_groups[combo_key].add(card["name"])
                if len(self.trait_groups[combo_key]) >= trait_threshold:
                    suggestions.append(f"{' '.join(combo)} Group")

        return suggestions


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


class SearchResultValidator:
    """Validates and scores search results"""

    def __init__(self):
        self.required_fields = {"name", "type", "aspects"}
        self.scoring_weights = {
            "exact_match": 10,
            "partial_match": 5,
            "trait_match": 3,
            "aspect_match": 2
        }

    def validate_result(self, result: Dict) -> bool:
        """Validate a single search result"""
        if not result:
            return False

        # Check required fields
        if not all(field in result for field in self.required_fields):
            return False

        # Validate data types
        if not isinstance(result.get("aspects"), list):
            return False

        return True

    def validate_and_score_results(self, query: Dict, results: List[Dict]) -> List[Dict]:
        """Validate and score search results"""
        scored_results = []

        for result in results:
            if not self.validate_result(result):
                continue

            score = 0
            match_reasons = []

            # Score exact matches
            if query.get("name") and query["name"].lower() == result["name"].lower():
                score += self.scoring_weights["exact_match"]
                match_reasons.append("Exact name match")

            # Score partial matches
            elif query.get("name") and query["name"].lower() in result["name"].lower():
                score += self.scoring_weights["partial_match"]
                match_reasons.append("Partial name match")

            # Score trait matches
            if query.get("traits"):
                query_traits = set(query["traits"])
                result_traits = set(result.get("traits", []))
                trait_matches = query_traits & result_traits
                if trait_matches:
                    score += len(trait_matches) * self.scoring_weights["trait_match"]
                    match_reasons.append(f"Matching traits: {', '.join(trait_matches)}")

            # Score aspect matches
            if query.get("aspects"):
                query_aspects = set(query["aspects"])
                result_aspects = set(result.get("aspects", []))
                aspect_matches = query_aspects & result_aspects
                if aspect_matches:
                    score += len(aspect_matches) * self.scoring_weights["aspect_match"]
                    match_reasons.append(f"Matching aspects: {', '.join(aspect_matches)}")

            if score > 0:
                scored_results.append({
                    **result,
                    "search_score": score,
                    "match_reasons": match_reasons
                })

        # Sort by score
        return sorted(scored_results, key=lambda x: x["search_score"], reverse=True)

class StarWarsUnlimitedSearch:
    def _create_indexes(self):
        """Create optimal indexes for common card queries"""
        try:
            self.cards.create_index([("type", 1)])
            self.cards.create_index([("cost", 1)])
            self.cards.create_index([("aspects", 1)])
            self.cards.create_index([("traits", 1)])
            self.cards.create_index([("arena", 1)])
            self.cards.create_index([("type", 1), ("cost", 1)])
            self.cards.create_index([("aspects", 1), ("type", 1)])
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
            raise

    def _load_cards(self):
        """Load cards from JSON into MongoDB"""
        try:
            if not self.cards_json.exists():
                logger.error(f"Cards JSON file not found: {self.cards_json}")
                return

            with open(self.cards_json, 'r') as f:
                cards_data = json.load(f)

            # Clear existing cards and insert new ones
            self.cards.delete_many({})
            if isinstance(cards_data, list):
                self.cards.insert_many(cards_data)
                logger.info(f"Loaded {len(cards_data)} cards into MongoDB")
            else:
                logger.error("Invalid cards data format")

        except Exception as e:
            logger.error(f"Error loading cards: {e}")
            raise

    def __init__(self, mongodb_uri: str = "mongodb://localhost:27017/",
                 docs_dir: str = "./data",
                 persist_dir: str = "./chroma_db",
                 cards_json: str = "./data/swudb_card_details.json"):
        """Initialize the search system"""
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir)
        self.cards_json = Path(cards_json)

        # MongoDB setup
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.star_wars_unlimited
        self.cards = self.db.cards

        # Vector store setup
        self.strategy_store = None
        self.qa_chain = None

        # LLM setup
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.deck_parser = PydanticOutputParser(pydantic_object=DeckRequest)
        self.cost_curve = CostCurveTarget()

        # Initialize
        self._create_indexes()
        self._load_cards()
        if self.docs_dir.exists():
            self.load_strategy_docs()

    def _get_card_alignment(self, card: Dict) -> str:
        """Determine if a card is Heroism, Villainy, or Neither"""
        aspects = card.get("aspects", [])
        if "Heroism" in aspects:
            return "Heroism"
        elif "Villainy" in aspects:
            return "Villainy"
        return "Neutral"

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
            if isinstance(cost_value, int):
                return cost_value
            if not cost_value or not str(cost_value).strip():  # Handle empty or whitespace
                return 0
            return int(str(cost_value).strip())
        except (ValueError, TypeError):
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

            logger.info(f"Loading documents from {self.docs_dir}")

            loader = DirectoryLoader(
                str(self.docs_dir),
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks")

            embeddings = OpenAIEmbeddings()
            self.strategy_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(self.persist_dir)
            )
            logger.info("Vector store created successfully")

            retriever = self.strategy_store.as_retriever(
                search_kwargs={"k": 4}
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                max_tokens_limit=4000
            )

            logger.info("QA chain initialized successfully")

        except Exception as e:
            logger.error(f"Error loading strategy documents: {e}")
            raise

    def process_natural_language_request(self, query: str) -> DeckRequest:
        """Convert natural language deck request into structured format"""
        template = """
        Convert the following Star Wars Unlimited deck building request into a structured format.
        Extract the leader name, key cards, and aspects (colors).
        Remember that valid aspects are: Command (Green), Cunning (Yellow), Aggression (Red), Vigilance (Blue).
        
        Request: {query}
        
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.deck_parser

        result = chain.invoke({
            "query": query,
            "format_instructions": self.deck_parser.get_format_instructions()
        })

        return result

    def _calculate_aspect_cost(self, card: Dict, deck_aspects: List[str], bases: List[Dict]) -> int:
        """Calculate the actual resource cost for a card considering aspects and bases"""
        card_aspects = [a for a in card.get("aspects", []) if a not in ["Heroism", "Villainy"]]
        base_aspects = []
        for base in bases:
            base_aspects.extend([a for a in base.get("aspects", []) if a not in ["Heroism", "Villainy"]])

        available_aspects = deck_aspects + base_aspects

        # If card shares any aspect with available aspects, no penalty
        if any(aspect in available_aspects for aspect in card_aspects):
            return int(card.get("cost", "0"))

        # Otherwise, +2 resource penalty
        return int(card.get("cost", "0")) + 2

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

    def _normalize_card_name(self, name: str) -> str:
        """Normalize card name for searching"""
        # Common spelling variations and fixes
        spelling_variations = {
            'sergent': 'sergeant',
            'sargent': 'sergeant',
            'wreker': 'wrecker',
            'maruader': 'marauder'
        }

        # Remove special characters and lowercase
        normalized = ''.join(c.lower() for c in name if c.isalnum() or c.isspace())

        # Apply spelling fixes
        for variant, correct in spelling_variations.items():
            normalized = normalized.replace(variant, correct)

        return normalized.strip()

    def _find_card(self, name: str, card_type: str = None) -> Dict:
        """Find a card with flexible name matching"""
        try:
            normalized_search = self._normalize_card_name(name)
            logger.info(f"Searching for card: {name} (normalized: {normalized_search})")

            # Build base query
            base_query = {}
            if card_type:
                base_query["type"] = card_type

            # Try exact match first (case insensitive)
            exact_query = {
                **base_query,
                "name": {"$regex": f"^{name}$", "$options": "i"}
            }
            card = self.cards.find_one(exact_query)
            if card:
                logger.info(f"Found exact match: {card['name']}")
                return self._serialize_mongo_doc(card)

            # Try with normalized name
            normalized_query = {
                **base_query,
                "name": {"$regex": f".*{normalized_search}.*", "$options": "i"}
            }
            cards = list(self.cards.find(normalized_query))

            if cards:
                # If we found multiple matches, try to find the best one
                if len(cards) > 1:
                    # First try to match card type if specified
                    if card_type:
                        type_matches = [c for c in cards if c["type"] == card_type]
                        if type_matches:
                            logger.info(f"Found type match: {type_matches[0]['name']}")
                            return self._serialize_mongo_doc(type_matches[0])

                    # Then try to find closest name match
                    for card in cards:
                        card_normalized = self._normalize_card_name(card["name"])
                        if normalized_search in card_normalized.split():
                            logger.info(f"Found name match: {card['name']}")
                            return self._serialize_mongo_doc(card)

                # Otherwise return the first match
                logger.info(f"Found first match: {cards[0]['name']}")
                return self._serialize_mongo_doc(cards[0])

            # Try matching individual words if still no match
            if not cards:
                words = normalized_search.split()
                word_queries = []
                for word in words:
                    if len(word) > 2:  # Only use words longer than 2 characters
                        word_queries.append({
                            "name": {"$regex": f".*{word}.*", "$options": "i"}
                        })
                if word_queries:
                    word_match_query = {
                        **base_query,
                        "$and": word_queries
                    }
                    cards = list(self.cards.find(word_match_query))
                    if cards:
                        logger.info(f"Found word match: {cards[0]['name']}")
                        return self._serialize_mongo_doc(cards[0])

            logger.warning(f"No card found for: {name}")
            return None

        except Exception as e:
            logger.error(f"Error finding card '{name}': {e}")
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
            logger.error(f"Error finding bases for aspects {aspects}: {e}")
            return []


    def process_natural_language_request(self, query: str) -> DeckRequest:
        """Convert natural language deck request into structured format"""
        template = """
        Convert the following Star Wars Unlimited deck building request into a structured format.
        Extract the leader name, key cards, and aspects (colors).

        Important card types to identify:
        - Leaders (explicit leaders mentioned)
        - Key cards (specific cards mentioned)
        - Bases (locations, facilities mentioned)
        - Groups (e.g., "Fringe Clones", "Clone Squad", etc.)
        - Ships/Vehicles (e.g., "Marauder")

        Remember that valid aspects are:
        - Command (Green)
        - Cunning (Yellow)
        - Aggression (Red)
        - Vigilance (Blue)

        Request: {query}

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.deck_parser

        result = chain.invoke({
            "query": query,
            "format_instructions": self.deck_parser.get_format_instructions()
        })

        # Process any group/type requests (like "Fringe Clones"
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
                score = 0

                # Base score based on cost effectiveness
                effective_cost = self._calculate_aspect_cost(card, deck_aspects, bases)
                if effective_cost == int(card.get("cost", "0")):
                    score += 3  # Bonus for native aspect cards
                elif effective_cost <= int(card.get("cost", "0")) + 2:
                    score += 1  # Still usable with penalty

                # Trait synergy score
                card_traits = set(card.get("traits", []))
                trait_matches = len(core_traits.intersection(card_traits))
                score += trait_matches * 2

                # Mechanic synergy score
                desc = card.get("description", "").lower()
                mechanic_matches = sum(1 for mech in core_mechanics if mech in desc)
                score += mechanic_matches

                # Cost curve consideration
                cost = int(card.get("cost", "0"))
                if cost <= 2 and efficiency_analysis["aspect_distribution"].get("2_or_less", 0) < 12:
                    score += 2
                elif cost == 3 and efficiency_analysis["aspect_distribution"].get("3", 0) < 12:
                    score += 2
                elif cost == 4 and efficiency_analysis["aspect_distribution"].get("4", 0) < 10:
                    score += 2

                scored_cards.append((card, score))

            # Sort by score and return top cards
            scored_cards.sort(key=lambda x: x[1], reverse=True)

            # Convert to list of serialized cards with scores
            return [{
                **self._serialize_mongo_doc(card),
                "synergy_score": score,
                "effective_cost": self._calculate_aspect_cost(card, deck_aspects, bases)
            } for card, score in scored_cards[:30]]  # Return top 30 cards

        except Exception as e:
            logger.error(f"Error finding complementary cards: {e}")
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
            logger.error(f"Error finding complementary bases: {e}")
            raise

    def build_deck_around_cards(self, query: str = None, leader_name: str = None,
                                key_cards: List[str] = None, aspects: List[str] = None) -> Dict:
        """Build a deck based on natural language query or specific parameters"""
        try:
            suggestions = {
                "recommended_bases": [],
                "synergy_notes": [],
                "aspect_recommendations": []
            }

            # [Previous query processing remains the same]

            selected_cards = []
            bases = []
            deck_aspects = []

            # Find and validate leader
            if leader_name:
                leader = self._find_card(leader_name, card_type="Leader")
                if not leader:
                    raise ValueError(f"Leader '{leader_name}' not found")

                deck_aspects.extend([a for a in leader["aspects"] if a not in ["Heroism", "Villainy"]])
                selected_cards.append(leader)

                # Suggest complementary bases
                potential_bases = self._find_complementary_bases(deck_aspects, key_cards)
                suggestions["recommended_bases"].extend(potential_bases)

                if potential_bases:
                    suggestions["synergy_notes"].append(
                        f"Recommended bases to support your strategy: {', '.join(base['name'] for base in potential_bases[:3])}"
                    )

            # Process key cards
            if key_cards:
                for card_name in key_cards:
                    card = self._find_card(card_name)
                    if not card:
                        continue

                    if card["type"] == "Base":
                        bases.append(card)
                        continue

                    # Calculate effective cost
                    effective_cost = self._calculate_aspect_cost(card, deck_aspects, bases)
                    card_aspects = [a for a in card["aspects"] if a not in ["Heroism", "Villainy"]]

                    # If card would need a penalty, suggest enabling bases
                    if effective_cost > int(card.get("cost", "0")):
                        missing_aspects = [a for a in card_aspects if a not in deck_aspects]
                        suggestions["aspect_recommendations"].append({
                            "card": card["name"],
                            "base_cost": int(card.get("cost", "0")),
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

            return {
                "leader": selected_cards[0] if selected_cards else None,
                "bases": bases,
                "key_cards": [c for c in selected_cards[1:] if c["type"] != "Base"],
                "recommended_cards": complementary_cards,
                "aspects": deck_aspects,
                "deck_analysis": {
                    "efficiency": efficiency_analysis,
                    "suggestions": suggestions,
                    "curve_analysis": self._analyze_cost_curve(complementary_cards),
                    "aspect_costs": {
                        "native_aspects": deck_aspects,
                        "base_enabled": efficiency_analysis["total_base_aspects"],
                        "penalty_cards": len(efficiency_analysis["penalty_cards"])
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error building deck: {e}")
            raise

    def _serialize_mongo_doc(self, doc: Dict) -> Dict:
        """Serialize MongoDB document to JSON-compatible format"""
        return json.loads(JSONEncoder().encode(doc))

    def _load_cards(self):
        """Load cards from JSON into MongoDB"""
        try:
            if not self.cards_json.exists():
                logger.error(f"Cards JSON file not found: {self.cards_json}")
                return

            with open(self.cards_json, 'r') as f:
                cards_data = json.load(f)

            # Clear existing cards and insert new ones
            self.cards.delete_many({})
            if isinstance(cards_data, list):
                self.cards.insert_many(cards_data)
                logger.info(f"Loaded {len(cards_data)} cards into MongoDB")
            else:
                logger.error("Invalid cards data format")

        except Exception as e:
            logger.error(f"Error loading cards: {e}")
            raise

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

    def _load_cards(self):
        """Load cards from JSON into MongoDB"""
        try:
            if not self.cards_json.exists():
                logger.error(f"Cards JSON file not found: {self.cards_json}")
                return

            with open(self.cards_json, 'r') as f:
                cards_data = json.load(f)

            # Clear existing cards and insert new ones
            self.cards.delete_many({})
            if isinstance(cards_data, list):
                self.cards.insert_many(cards_data)
                logger.info(f"Loaded {len(cards_data)} cards into MongoDB")
            else:
                logger.error("Invalid cards data format")

        except Exception as e:
            logger.error(f"Error loading cards: {e}")
            raise


def load_strategy_docs(self):
    """Load all PDF strategy documents from the docs directory"""
    try:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        logger.info(f"Loading documents from {self.docs_dir}")

        # Configure document loader for the directory
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )

        # Load documents
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Split documents
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} text chunks")

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.strategy_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(self.persist_dir)
        )
        logger.info("Vector store created successfully")

        # Initialize LLM and retrieval chain
        llm = ChatOpenAI(temperature=0)
        retriever = self.strategy_store.as_retriever(
            search_kwargs={"k": 4}  # Just specify number of documents to retrieve
        )

        # Create the conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,  # Include source documents in response
            max_tokens_limit=4000  # Adjust based on your needs
        )

        logger.info("QA chain initialized successfully")

    except Exception as e:
        logger.error(f"Error loading strategy documents: {e}")
        raise


def get_strategy_advice(self, query: str, chat_history: List = None) -> Dict[str, Any]:
    """
    Get strategy advice with source documents

    Args:
        query: The strategy question
        chat_history: Optional list of previous QA pairs

    Returns:
        Dictionary containing answer and source documents
    """
    if not self.qa_chain:
        raise ValueError("Strategy documents not loaded. Call load_strategy_docs first")

    try:
        # Prepare chat history if not provided
        if chat_history is None:
            chat_history = []

        # Get response from chain
        response = self.qa_chain({
            "question": query,
            "chat_history": chat_history
        })

        # Extract source documents
        sources = []
        if "source_documents" in response:
            sources = [
                {
                    "title": Path(doc.metadata["source"]).name,
                    "page": str(doc.metadata.get("page", "Unknown")),  # Ensure string
                    "content": doc.page_content[:200] + "..."  # Preview of content
                }
                for doc in response["source_documents"]
            ]

        return {
            "answer": response["answer"],
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error getting strategy advice: {e}")
        raise


def find_synergistic_cards(self, card_name: str) -> Dict[str, Any]:
    """
    Find cards that work well with a specific card

    Args:
        card_name: Name of the card to find synergies for
    """
    try:
        # Find the base card
        base_card = self.cards.find_one({
            "name": {"$regex": f".*{card_name}.*", "$options": "i"}
        })

        if not base_card:
            return {"error": f"Card '{card_name}' not found"}

        base_card = self._serialize_mongo_doc(base_card)

        # Find cards with matching aspects
        aspect_matches = list(self.cards.find({
            "aspects": {"$in": base_card.get("aspects", [])},
            "_id": {"$ne": base_card["_id"]}
        }).limit(10))

        # Find cards with matching traits
        trait_matches = list(self.cards.find({
            "traits": {"$in": base_card.get("traits", [])},
            "_id": {"$ne": base_card["_id"]}
        }).limit(10))

        return {
            "base_card": base_card,
            "aspect_synergies": [self._serialize_mongo_doc(card) for card in aspect_matches],
            "trait_synergies": [self._serialize_mongo_doc(card) for card in trait_matches]
        }

    except Exception as e:
        logger.error(f"Error finding synergistic cards: {e}")
        raise


def main():
    # Initialize with your document directory
    search = StarWarsUnlimitedSearch(
        docs_dir="./data",  # Directory containing your PDFs
        persist_dir="./vector_store"  # Directory to store embeddings
    )

    try:
        # Example strategy query
        response = search.get_strategy_advice(
            "How do I effectively use capture mechanics in my deck?"
        )

        # Print response with sources
        print("Answer:", response["answer"])
        print("\nSources:")
        for source in response["sources"]:
            print(f"- {source['title']} (Page {source['page']})")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
