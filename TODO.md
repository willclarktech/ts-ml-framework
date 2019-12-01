# TODO

## Datasets/Benchmarks

- ...

## Features

- Pooling
- Combined layers (eg linear+nonlinear, linear+softmax+cross-entropy)
- Dropout
- Customisable bias initialisation (eg good to set to 1 when using ReLU)
- GAN
- Momentum
- Weight decay
- LSTM
- Transformer
- Hyperparameter optimisation

## Refactoring/Performance/Testing

- Combine backpropagate and update
- Refactor activate/backprop/update in terms of matrix multiplication
- More unit tests
- Refactor function layers not to store copies of functions
- Impure functions for performance
