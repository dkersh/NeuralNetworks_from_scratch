from .activation_functions import ActivationFunction
from typing import List
from dataclasses import dataclass
import numpy as np
from .neuron import Neuron


@dataclass
class Layer:
    num_neurons: int
    activation: ActivationFunction
    input_shape: np.ndarray = None
    neurons: List[Neuron] = None
    inputs: np.ndarray = None
    previous_layer: "Layer" = None

    def __post_init__(self):
        if self.input_shape is not None:
            self.initialise_layer()

    def __call__(self, layer):
        self.input_shape = layer.num_neurons
        self.initialise_layer()
        self.previous_layer = layer
        return self

    # Note, we make the weights a property to retain the vectorizability.
    @property
    def weights(self):
        return np.array([n.weights for n in self.neurons])

    @weights.setter
    def weights(self, weights):
        for w, n in zip(weights, self.neurons):
            n.weights = w

    @property
    def bias(self):
        return np.array([n.bias for n in self.neurons])

    @property
    def a(self):
        return np.array([n.a for n in self.neurons])

    @property
    def z(self):
        return np.array([n.z for n in self.neurons])

    @property
    def error(self):
        return np.array([n.error for n in self.neurons])

    @property
    def delta(self):
        return np.array([n.delta for n in self.neurons])

    @property
    def l_inputs(self):
        return np.array([n.inputs for n in self.neurons])

    def initialise_layer(self):
        self.neurons = [
            Neuron(self.input_shape, self.activation) for _ in range(self.num_neurons)
        ]

    def forward(self, x):
        self.inputs = x
        for n in self.neurons:
            n.forward(x)
        return self.a

    def backward(self, error_term):
        for n, delta in zip(self.neurons, error_term):
            n.backward(delta)

    def update_weights(self, lr=0.05, batch_size=1):
        for n in self.neurons:
            n.update_weights(lr, batch_size)
