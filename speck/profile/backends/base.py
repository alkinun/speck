"""backend contracts for prepared artifacts and runtime sessions."""

from abc import ABC, abstractmethod


class RuntimeSession(ABC):
    @abstractmethod
    def allocate_state(self, batch_size, length, cache_dtype):
        raise NotImplementedError

    @abstractmethod
    def prefill(self, tokens, state):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens, state):
        raise NotImplementedError

    @abstractmethod
    def synchronize(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class BackendPlugin(ABC):
    @property
    @abstractmethod
    def identity(self):
        raise NotImplementedError

    @abstractmethod
    def supports(self, config, scenario):
        raise NotImplementedError

    @abstractmethod
    def prepare(self, config, scenario, state_dict=None):
        raise NotImplementedError

    @abstractmethod
    def load(self, artifact, scenario):
        raise NotImplementedError
