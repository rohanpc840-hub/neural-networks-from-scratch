# Neural Networks from Scratch

Small, runnable neural network examples built with NumPy. The project works through forward propagation, loss calculation, backpropagation, and gradient-descent updates without a deep-learning framework.

## Examples

| Script | Problem | Network | What it shows |
| --- | --- | --- | --- |
| `neural.py` | XOR | 2 -> 3 -> 1 | Binary classification with sigmoid activations and binary cross-entropy. Prints training loss, predictions, expected values, and learned parameters. The random seed is fixed for repeatable initialization. |
| `iris.py` | Iris flower classification | 4 -> 8 -> 3 | Multiclass classification with a softmax output and cross-entropy loss. Prints accuracy and plots the loss history. |

Both examples store samples in columns, so the matrix operations process the whole dataset in each training step.

## Getting started

Use Python 3 and install the dependencies:

```bash
python -m pip install numpy scikit-learn matplotlib
