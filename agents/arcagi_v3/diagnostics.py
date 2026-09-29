from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger()


class DiagnosticsLogger:
    """Configurable structured diagnostics for development and tracing."""

    def __init__(self, prefix: str = "[V3]", enabled: bool = True) -> None:
        self.prefix = prefix
        self.enabled = enabled

    def log(self, category: str, message: str) -> None:
        if self.enabled:
            logger.info(f"{self.prefix}[{category}] {message}")

    def observe(self, msg: str) -> None:
        self.log("OBSERVE", msg)

    def objects(self, msg: str) -> None:
        self.log("OBJECTS", msg)

    def state(self, msg: str) -> None:
        self.log("STATE", msg)

    def transition(self, msg: str) -> None:
        self.log("TRANSITION", msg)

    def goal(self, msg: str) -> None:
        self.log("GOAL", msg)

    def plan(self, msg: str) -> None:
        self.log("PLAN", msg)

    def action(self, msg: str) -> None:
        self.log("ACTION", msg)

    def result(self, msg: str) -> None:
        self.log("RESULT", msg)

    def model_update(self, msg: str) -> None:
        self.log("MODEL UPDATE", msg)
