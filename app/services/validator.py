from typing import Dict, List


class SearchResultValidator:
    def __init__(self):
        self.required_fields = {"name", "type", "aspects"}
        self.scoring_weights = {
            "exact_match": 10,
            "partial_match": 5,
            "trait_match": 3,
            "aspect_match": 2
        }

    def validate_result(self, result: Dict) -> bool:
        if not result:
            return False
        return all(field in result for field in self.required_fields)

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
