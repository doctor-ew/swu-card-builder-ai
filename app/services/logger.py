import logging
from typing import Dict, List, Any, Optional

class SearchLogger:
    def __init__(self):
        self.logger = logging.getLogger("card_search")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler("card_search.log")
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        fh.setFormatter(detailed_formatter)
        ch.setFormatter(simple_formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _format_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format extra parameters for logging"""
        if extra is None:
            return {}
        return {k: v for k, v in extra.items() if k != 'args'}

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Debug level logging"""
        self.logger.debug(msg, extra=self._format_extra(extra))

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Info level logging"""
        self.logger.info(msg, extra=self._format_extra(extra))

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Warning level logging"""
        self.logger.warning(msg, extra=self._format_extra(extra))

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Error level logging"""
        self.logger.error(msg, extra=self._format_extra(extra))