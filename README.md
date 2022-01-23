# NNK-EP

> Implementation of the Non-Negative Kernel regression - Edge Pooling algorithm as well as a framework to test the algorithm.

This project consists of a [Telegram](https://www.upc.edu/ca)®  bot developed in Python which answers either
with text or images to questions related to geometric graphs defined over the
[Bicing](https://www.bicing.barcelona/es)® stations in the city of Barcelona.



## File Architecture

The repository contains two folders.

[`utils`](./utils) contains the implementation of the algorithm:
* [`nnk.py`](./utils/nnk.py) is the main file of the implementation of the NNK-EP algorithm.
* [`graph_utils.py`](./utils/data.py) has some auxiliary functions used in the NNK-EP implementation.
* [`non_neg_qpsolver.py`](./utils/non_neg_qpsolver.py) is a solver for solving non negative quadratic programs with positive definite matrices.

[`train`](./train) contains the scripts necessary for testing the algorithm, including the training setup, auxiliary functions and the graph models:
* [`graph_model.py`](./train/graph_model.py) Graph models used for the testing of the algorthm.
* [`metrics.py`](./train/metrics.py) File containing the metrics class, used to keep track of the performance of the models in training and testing.
* [`tran_proteins.py`](./train/tran_proteins.py) Testing framework for the PROTEINS dataset, a graph classification dataset.
* [`tran_cifar.py`](./train/tran_cifar.py) Testing framework for the CIFAR-10 dataset, a graph classification dataset.
* [`tran_mnist.py`](./train/tran_mnist.py) Testing framework for the MNIST dataset, a graph classification dataset.
* [`tran_cora.py`](./train/tran_cora.py) Testing framework for the CORA dataset, a node classification dataset.


*Note: The rest of the files in the master branch are auxiliary or license related*

## License

[MIT License](./LICENSE)
