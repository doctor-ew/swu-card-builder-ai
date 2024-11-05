# /app/__init__.py

import os
from .services.card_search import StarWarsUnlimitedSearch


def create_search_service(
        mongodb_uri: str = None,
        docs_dir: str = None,
        persist_dir: str = None,
        cards_json: str = None,
        openai_api_key: str = None
) -> StarWarsUnlimitedSearch:
    """Create and initialize the search service with configuration"""

    # Use environment variables as defaults if not provided
    mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    docs_dir = docs_dir or os.getenv("DOCS_DIR", "./data")
    persist_dir = persist_dir or os.getenv("PERSIST_DIR", "./chroma_db")
    cards_json = cards_json or os.getenv("CARDS_JSON", "./data/swudb_card_details.json")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OpenAI API key is required")

    return StarWarsUnlimitedSearch(
        mongodb_uri=mongodb_uri,
        docs_dir=docs_dir,
        persist_dir=persist_dir,
        cards_json=cards_json,
        openai_api_key=openai_api_key
    )