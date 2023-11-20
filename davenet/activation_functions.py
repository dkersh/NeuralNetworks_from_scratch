from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def function(self):
        pass

    @abstractmethod
    def derivative(self):
        pass

@dataclass
class Sigmoid(ActivationFunction):
    name: str = 'sigmoid'

    def function(self, x):
        x = np.clip(x, -500, 500 )
        return 1/(1 + np.exp(-x))
    
    def derivative(self, x):
        return self.function(x)*(1-self.function(x))

@dataclass
class ReLU(ActivationFunction):
    name: str = 'relu'

    def function(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)