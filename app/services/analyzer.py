import re
from typing import Dict, List, Any, Set


class CardRelationshipAnalyzer:
    def __init__(self):
        self.relationship_cache = {}
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
