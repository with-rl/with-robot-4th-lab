"""Centralized error definitions for the MLDT-based planner."""

from __future__ import annotations

from typing import Any, Mapping

# status descriptions:
# 400: Bad Request - The request could not be understood or was missing required parameters.
# 422: Unprocessable Entity - The request was well-formed but was unable to be
#      followed due to semantic errors.
# 500: Internal Server Error - An error occurred on the server.
# 502: Bad Gateway - The server was acting as a gateway or proxy and received
#      an invalid response from the upstream server.


class BaseServiceError(Exception):
    """Base exception that provides structured metadata for logging and APIs."""

    default_code = "UNKNOWN"
    default_status = 500
    default_domain = "core"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        domain: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code or self.default_code
        self.status = status or self.default_status
        self.domain = domain or self.default_domain
        self.details = dict(details) if details else {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error_code": self.code,
            "error_message": self.message,
            "status": self.status,
            "domain": self.domain,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigError(BaseServiceError):
    """Raised when configuration files are missing or invalid."""

    default_code = "CONFIG_ERROR"


class PromptLoadError(BaseServiceError):
    """Raised when prompt templates cannot be loaded."""

    default_code = "PROMPT_LOAD_ERROR"


class ParsingError(BaseServiceError):
    """Raised when parser cannot extract structured data."""

    default_code = "PARSING_ERROR"
    default_status = 422


class LLMError(BaseServiceError):
    """Raised when LLM calls fail or responses are unusable."""

    default_code = "LLM_ERROR"
    default_status = 502


class RateLimitExceededError(LLMError):
    """Raised when upstream LLM returns a rate-limit or quota error."""

    default_code = "RATE_LIMIT_EXCEEDED"
    default_status = 429


class GraphExecutionError(BaseServiceError):
    """Raised when the planner graph cannot complete execution."""

    default_code = "GRAPH_EXECUTION_ERROR"
    default_status = 500


class UtilsValidationError(BaseServiceError):
    """Raised when validation of inputs or state fails."""

    default_code = "VALIDATION_ERROR"
    default_status = 400


class UtilsConfigurationError(BaseServiceError):
    """Raised when there is a configuration error in utils."""

    default_code = "UTILS_CONFIGURATION_ERROR"
    default_status = 500


class GraphInitializeError(BaseServiceError):
    """Raised when the Graph is in an invalid state."""

    default_code = "GRAPH_INITIALIZE_ERROR"
    default_status = 500


__all__ = (
    "BaseServiceError",
    "ConfigError",
    "PromptLoadError",
    "ParsingError",
    "LLMError",
    "RateLimitExceededError",
    "GraphInitializeError",
)
