from .activation_functions import ActivationFunction
from typing import List
from dataclasses import dataclass
import numpy as np
from .neuron import Neuron

@dataclass
class Layer():
    num_neurons: int
    activation: ActivationFunction
    input_shape: np.ndarray = None
    neurons: List[Neuron] = None
    inputs: np.ndarray = None
    previous_layer: 'Layer' = None

    # Note, we make the weights a property to retain the vectorizability.
    @property
    def weights(self):
        return np.array([n.weights for n in self.neurons])
    
    @weights.setter
    def weights(self, weights):
        for w, n in zip(weights, self.neurons):
            n.weights = w

    # TODO these probably don't need to be properties
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

    def __post_init__(self):
        if self.input_shape is not None:
            self.initialise_layer()

    def __call__(self, layer):
        self.input_shape = layer.num_neurons
        self.initialise_layer()
        self.previous_layer = layer
        return self
    
    def initialise_layer(self):
        self.neurons = [Neuron(self.input_shape, self.activation) for _ in range(self.num_neurons)]

    def forward(self, x):
        self.inputs = x
        for n in self.neurons: n.forward(x)
        return self.a
    
    #def backward(self, error, previous_output, lp1 = None):
    #    [n.backward(error, p) for n, p in zip(self.neurons, previous_output)]
    #    if lp1 is not None:
    #        error = compute_error_term(self, lp1)
    #        return error
        
    def backward(self, error_term):
        for n, delta in zip(self.neurons, error_term): 
            n.backward(delta)

    def update_weights(self, lr=0.05):
        for n in self.neurons:
            n.update_weights(lr)

def compute_error_term(layer, layerp1):
    error = np.zeros(len(layer.neurons))
    for i in range(len(layer.neurons)):
        for j in range(len(layerp1.neurons)):
            error[i] += layerp1.neurons[j].weights[i] * layerp1.neurons[j].error # compute the dot product iteratively.
    
    return error