from dataclasses import dataclass
from .activation_functions import ActivationFunction
import numpy as np

@dataclass
class Neuron():
    input_shape: int
    activation_function: ActivationFunction
    weights: np.ndarray[np.float64] = None
    bias: float = np.random.uniform(0, 1, size=1)
    error: float = 0
    delta: np.ndarray = np.array([])
    delta_weights: np.ndarray = np.array([])
    weights_gradient = 0
    bias_gradient = 0
    z: float = None
    a: float = None
    inputs = None

    def __post_init__(self):
        """Initialise the weights. We do this post_initialisation because we depend
        on the input shape.
        """
        self.weights = np.random.uniform(-1, 1, size=self.input_shape)

    def forward(self, x: np.ndarray) -> float:
        self.inputs = np.squeeze(x)
        self.z = np.dot(self.weights, x) + self.bias
        self.a = self.activation_function.function(self.z)

        return self.a
    
    def backward(self, delta: np.ndarray):
        self.error = delta * self.activation_function.derivative(self.z)
        self.weights_gradient +=  np.outer(delta, self.inputs)
        self.bias_gradient += delta

        return self.error
    
    def update_weights(self, lr=0.05, batch_size = 1):
        self.weights -= (np.squeeze(self.weights_gradient) * lr/batch_size)
        self.bias -= (self.bias_gradient * lr/batch_size)

        self.weights_gradient = 0
        self.bias_gradient = 0