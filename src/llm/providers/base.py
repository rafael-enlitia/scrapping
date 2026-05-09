"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """Send a system + user message pair and return the raw text response."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier being used."""
