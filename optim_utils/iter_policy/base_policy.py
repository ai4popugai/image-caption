from abc import ABC, abstractmethod


class BaseIterationPolicy(ABC):
    @abstractmethod
    def step(self, global_step: int) -> float:
        raise NotImplementedError('Method should be implemented in child class')

