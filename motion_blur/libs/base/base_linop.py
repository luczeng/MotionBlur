from abc import ABC, abstractmethod


class linop(ABC):
    @abstractmethod
    def __mul__(self, other):
        pass
