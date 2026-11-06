"""
Wrapper functions for multiprocessing with logging support.
"""

import sys

# Global variable to store current worker logger
_current_logger = None


def get_current_worker_logger():
    """Get the current worker logger."""
    return getattr(sys.modules[__name__], '_current_logger', None)
