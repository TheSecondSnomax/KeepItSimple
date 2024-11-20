from typing import Optional

from numpy import array, ones

from snoscience._layers import BasicLayer, Layer
from snoscience._neurons import Neuron, NeuronShared


def setup_neuron(neuron: Neuron) -> None:
    """
    Set all neuron weights and its bias to 0.5 to simplify calculations.
    """
    neuron.weights = 0.5 * ones(shape=neuron.weights.shape)
    neuron.bias = 0.5


def setup_neuron_shared(neuron: NeuronShared) -> None:
    """
    Same as "setup_neuron" function, but then for NeuronShared instances.
    """
    neuron.set_weights(weights=0.5 * ones(shape=neuron.weights))
    neuron.set_bias(bias=array([0.5]))


def setup_layer(layer: Layer, layer_prev: Optional[Layer] = None, layer_next: Optional[Layer] = None) -> None:
    """
    Add 2 neurons to the layer and define the previous and next layers.
    """
    neuron_1 = Neuron(weights=2)
    neuron_2 = Neuron(weights=2)

    setup_neuron(neuron=neuron_1)
    setup_neuron(neuron=neuron_2)

    layer.previous = BasicLayer() if layer_prev is None else layer_prev
    layer.next = BasicLayer() if layer_next is None else layer_next
    layer.neurons = [neuron_1, neuron_2]
