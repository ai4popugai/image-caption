from abc import ABC, abstractmethod


class BaseIterationPolicy(ABC):
    @abstractmethod
    def evaluate(self, global_step: int) -> float:
        raise NotImplementedError('Method should be implemented in child class')

