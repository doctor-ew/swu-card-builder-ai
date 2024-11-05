from .metrics import PerformanceMonitor, monitor_method, monitor_db

# Create a singleton instance if needed
monitor = PerformanceMonitor("swu_search")

# Export the decorators - use either the class-based or function-based ones
__all__ = ['monitor_method', 'monitor_db', 'monitor']