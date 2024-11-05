from functools import wraps
import time
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log application performance metrics"""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.logger = logging.getLogger(app_name)

    def monitor(self, method_name: str):
        """Decorator to monitor method execution time and log results"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.logger.debug(
                        f"Method {method_name} completed",
                        extra={
                            "duration": f"{duration:.2f}s",
                            "status": "success",
                            "method": method_name
                        }
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(
                        f"Method {method_name} failed: {str(e)}",
                        extra={
                            "duration": f"{duration:.2f}s",
                            "status": "error",
                            "method": method_name,
                            "error": str(e)
                        }
                    )
                    raise

            return wrapper

        return decorator

    def record_db_operation(self, collection_name: str):
        """Decorator to monitor database operations"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.logger.debug(
                        f"DB operation on {collection_name} completed",
                        extra={
                            "duration": f"{duration:.2f}s",
                            "collection": collection_name,
                            "status": "success"
                        }
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(
                        f"DB operation on {collection_name} failed: {str(e)}",
                        extra={
                            "duration": f"{duration:.2f}s",
                            "collection": collection_name,
                            "status": "error",
                            "error": str(e)
                        }
                    )
                    raise

            return wrapper

        return decorator


# Create function-based decorators for simpler usage
def monitor_method(method_name: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(
                    f"Method {method_name} completed",
                    extra={
                        "duration": f"{duration:.2f}s",
                        "status": "success",
                        "method": method_name
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Method {method_name} failed: {str(e)}",
                    extra={
                        "duration": f"{duration:.2f}s",
                        "status": "error",
                        "method": method_name,
                        "error": str(e)
                    }
                )
                raise

        return wrapper

    return decorator


def monitor_db(collection_name: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(
                    f"DB operation on {collection_name} completed",
                    extra={
                        "duration": f"{duration:.2f}s",
                        "collection": collection_name,
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"DB operation on {collection_name} failed: {str(e)}",
                    extra={
                        "duration": f"{duration:.2f}s",
                        "collection": collection_name,
                        "status": "error",
                        "error": str(e)
                    }
                )
                raise
            return wrapper

        return decorator

    return decorator