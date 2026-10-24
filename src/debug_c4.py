"""Opt-in C4 diagnostics. Enabled only when QH_DEBUG_C4=1."""
from __future__ import annotations
import logging, os
from pathlib import Path

_LOGGER = None

def debug_enabled() -> bool:
    return os.environ.get("QH_DEBUG_C4", "0").strip().lower() in {"1", "true", "yes", "on"}

def get_logger():
    global _LOGGER
    if not debug_enabled():
        return None
    if _LOGGER is None:
        path = Path(os.environ.get("QH_DEBUG_C4_LOG", "debug_logs/c4_debug.log"))
        path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("qh_ctde_c4_debug")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        _LOGGER = logger
    return _LOGGER

def log(message: str, *args):
    logger = get_logger()
    if logger is not None:
        logger.debug(message, *args)
