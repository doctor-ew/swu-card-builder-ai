class CardSearchError(Exception):
    """Base exception for card search errors"""
    pass

class CardNotFoundError(CardSearchError):
    """Raised when a card cannot be found"""
    pass

class InvalidQueryError(CardSearchError):
    """Raised when a search query is invalid"""
    pass