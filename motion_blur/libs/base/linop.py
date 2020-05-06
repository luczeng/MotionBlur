from abc import ABC

def linop(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __mul__(self,other):
        pass
