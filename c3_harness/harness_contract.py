"""Backward-compatible contract module for pluggable harness providers."""

from __future__ import annotations

from .provider import (
    HarnessCapabilities,
    HarnessProvider,
    HarnessProviderConfig,
    HarnessProviderFactory,
)

__all__ = [
    "HarnessCapabilities",
    "HarnessProvider",
    "HarnessProviderConfig",
    "HarnessProviderFactory",
]
