import logging
import os
import sys
from datetime import datetime


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def _colors_enabled() -> bool:
    return sys.stderr.isatty() and os.getenv("NO_COLOR") is None


def style(text, color=None, bold=False, dim=False, italic=False, underline=False):
    """Apply ANSI terminal styles when supported by the current terminal."""
    if not _colors_enabled():
        return text

    codes = []
    if color:
        codes.append(color)
    if bold:
        codes.append(_Ansi.BOLD)
    if dim:
        codes.append(_Ansi.DIM)
    if italic:
        codes.append(_Ansi.ITALIC)
    if underline:
        codes.append(_Ansi.UNDERLINE)

    return "".join(codes) + str(text) + _Ansi.RESET


class StyledFormatter(logging.Formatter):
    """Fancy console formatter with Linux-friendly colors and symbols."""

    _LEVEL_TOKENS = {
        logging.DEBUG: ("DEBUG", _Ansi.BLUE),
        logging.INFO: ("INFO ", _Ansi.GREEN),
        logging.WARNING: ("WARN ", _Ansi.YELLOW),
        logging.ERROR: ("ERROR", _Ansi.RED),
        logging.CRITICAL: ("CRIT ", _Ansi.RED),
    }

    def format(self, record):
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level_token, color = self._LEVEL_TOKENS.get(record.levelno, (record.levelname, _Ansi.WHITE))
        left = style(f"[{ts}]", color=_Ansi.WHITE, dim=True)
        level = style(level_token, color=color, bold=True)
        msg = record.getMessage()
        return f"{left} {level} {msg}"


def banner(logger, title, subtitle=None):
    line = style("=" * 68, color=_Ansi.GREEN, bold=True)
    logger.info(line)
    logger.info(style(str(title), color=_Ansi.GREEN, bold=True))
    if subtitle:
        logger.info(style(str(subtitle), color=_Ansi.GREEN, dim=True))
    logger.info(line)


def section(logger, title, icon=None):
    logger.info(style(str(title), color=_Ansi.GREEN, bold=True))


def success(logger, message, *args):
    prefix = style("complete", color=_Ansi.GREEN, bold=True)
    logger.info(f"{prefix} {message}", *args)


def setup_logger(log_dir=None):
    logger = logging.getLogger('lane_detection')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    # Console handler (styled)
    ch = logging.StreamHandler()
    ch.setFormatter(StyledFormatter())
    logger.addHandler(ch)

    # File handler (plain text)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{timestamp}.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    return logger


def get_logger():
    logger = logging.getLogger('lane_detection')
    if not logger.handlers:
        return setup_logger()
    return logger