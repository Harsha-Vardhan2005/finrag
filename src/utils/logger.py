"""
src/utils/logger.py
===================
Centralized logger for the entire project using loguru.
Import this everywhere instead of stdlib logging.
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Console handler — colored, readable
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File handler — full details for debugging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger.add(
    log_dir / "rag_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    compression="zip",
)

__all__ = ["logger"]
