import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    P11: ASCII-only, Windows cp1252-safe logger.
    No emoji. No Unicode box-drawing characters.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler - stdout, UTF-8 forced
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # ASCII-only format (P11: no emoji, no box-drawing)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
