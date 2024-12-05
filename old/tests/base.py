from time import time

from numpy import load
from numpy import max as numpy_max
from snoscience.metrics import calculate_accuracy, calculate_mse
from snoscience.networks import NeuralNetwork


def load_binary() -> (tuple, tuple):
    """
    Load binary classification data and normalise it per column.

    Returns
    -------
    (tuple, tuple):
        Binary classification training and test set.
    """
    # Load data.
    data = load(file="./data/binary_breast.npz")

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Normalise input data per column.
    normalise = 1 / numpy_max(x_train, axis=0)

    x_train = normalise * x_train
    x_test = normalise * x_test

    return (x_train, y_train), (x_test, y_test)


def load_multi() -> (tuple, tuple):
    """
    Load multi classification data and normalise it.

    Returns
    -------
    (tuple, tuple):
        Multi classification training and test set.
    """
    # Load data.
    data = load(file="./data/multi_mnist.npz")

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Normalise input data.
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


def load_regression() -> (tuple, tuple):
    """
    Load regression data and normalise it per column.

    Returns
    -------
    (tuple, tuple):
        Regression training and test set.
    """
    # Load data.
    data = load(file="./data/regression_mpg.npz")

    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Normalise input data per column.
    normalise = 1 / numpy_max(x_train, axis=0)

    x_train = normalise * x_train
    x_test = normalise * x_test

    # Normalise output data per column.
    normalise = 1 / numpy_max(y_train)

    y_train = normalise * y_train
    y_test = normalise * y_test

    return (x_train, y_train), (x_test, y_test)


def run_self(
    data: tuple, inputs: int, layers: list, epochs: int, samples: int, classify: bool
) -> (float, float, float):
    """
    Run own binary classification model.

    Parameters
    ----------
    data: tuple
        Training and test sets.
    inputs: int
        Number of input for the network
    layers: list
        Neurons per layer.
    epochs: int
        Number of epochs used for training.
    samples: int
        Number of samples to use per epoch.
    classify: bool
        Use classification.

    Returns
    -------
    (float, float, float):
        Elapsed time, model accuracy, and model MSE on the test set.
    """
    # Get training data.
    (x_train, y_train), (x_test, y_test) = data

    # Create model.
    model = NeuralNetwork(inputs=inputs)

    # Create layers.
    for layer in layers:
        model.add_layer(neurons=layer, activation="sigmoid")

    # Define optimizer hyperparameters.
    rate = 0.01

    # Train model.
    start = time()
    model.train(x=x_train, y=y_train, epochs=epochs, samples=samples, rate=rate)

    # Make predictions.
    y_calc = model.predict(x=x_test, classify=classify)

    # Calculate performance.
    accuracy = calculate_accuracy(calc=y_calc, true=y_test)
    mse = calculate_mse(calc=y_calc, true=y_test)[0]
    end = time()
    elapsed = round(end - start, 1)

    return elapsed, accuracy, mse
