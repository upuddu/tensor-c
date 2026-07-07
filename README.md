# tensor-c

A feedforward neural-network (multilayer perceptron) library written in pure C99, with backpropagation and stochastic gradient descent.

## Overview

`tensor-c` implements the core building blocks needed to define, train, and run
fully connected feedforward neural networks in standard C99 with no external
dependencies beyond the C math library. It supports arbitrary numbers of hidden
layers, per-layer activation functions, mini-batch training, model finalization
for inference, and binary persistence of learned parameters.

The design is deliberately **neuron-centric**: the forward and backward passes
iterate explicitly over neurons and their connections rather than over batched
matrix operations. This keeps the code readable and easy to follow, at the cost
of the raw throughput a vectorized (BLAS-style) implementation would provide.

All floating-point computation uses a single configurable type, `pfloat`
(currently `typedef`'d to `long double` in `nn.h`), so precision and memory use
can be tuned in one place.

## Features

- Fully connected feedforward networks with any number of hidden layers.
- Activation functions: sigmoid, hyperbolic tangent, ReLU, and identity (linear).
- Loss functions: mean squared error (MSE) and binary cross-entropy (BCE).
- Training via stochastic gradient descent, with configurable mini-batch size
  and optional per-epoch shuffling.
- Backpropagation implemented as an explicit reverse pass over the network.
- Variance-scaled random weight initialization, with a seed for reproducible runs.
- `FinalizedModel`: a compact, inference-only representation that drops all
  training state so a trained network can run (and be freed) independently.
- Binary save/load of parameters (weights, biases, and layer dimensions) with a
  magic number and version header for integrity checking.

## Build and run

A `Makefile` is provided. It compiles the library together with each example
program using `gcc -Wall -O2 -std=c99` and links against `-lm`.

```bash
make        # build all example programs
make test   # build and run every example
make clean  # remove build artifacts and generated parameter files
```

Each example builds to a `nn_<name>` binary (for example, `nn_xor_test`).

## Usage

A minimal end-to-end example: define a network, train it, and read predictions.

```c
#include "nn.h"

// 1. Configure the optimizer and training parameters.
Optimizer *opt = make_optimizer(0.1L);   // learning rate
Params params = {
    .loss = &nnBCE,             // binary cross-entropy
    .optimizer = opt,
    .seed = 42,                 // > 0 for reproducible weights
    .log_frequency_epochs = 10000,
    .randomize_batches = false,
};

// 2. Build the model: 2 inputs -> 4 ReLU -> 1 sigmoid output.
//    The variadic layer list is terminated by a (size_t)0.
Model *model = make_model(2, 1, &params,
                          4, &nnRelu,
                          1, &nnSigmoid,
                          0);

// 3. Train over a flattened dataset.
model_train(model, inputs, targets, num_samples, /*epochs=*/100000, /*batch=*/4);

// 4. Predict (returns an internal buffer valid until the next call).
const pfloat *out = forward(model, sample);

// 5. Clean up.
free_model(model);
free(opt);
```

For deployment, convert the trained network into an inference-only model:

```c
FinalizedModel *fm = model_finalize(model);
free_model(model);                       // training state no longer needed

pfloat output[1];
finalized_model_predict(fm, sample, output);

free_finalized_model(fm);
```

## API summary

Defined in `nn.h`:

- **Construction / teardown** — `make_model`, `free_model`, `make_optimizer`,
  `make_layer` / `make_neuron` and their `free_*` counterparts.
- **Training** — `model_train` (full training loop), plus the lower-level
  `forward`, `backward`, `model_zero_grads`, `model_apply_gradients`, and
  `model_calculate_average_loss`.
- **Inference** — `forward` for a live `Model`, or `model_finalize` +
  `finalized_model_predict` for a compact deployment model.
- **Persistence** — `model_save_params` / `model_load_params` and the
  `finalized_model_*` equivalents. Only weights, biases, and dimensions are
  stored; the identical architecture (layer sizes and activations) must be
  reconstructed before loading.
- **Constants** — activations `nnSigmoid`, `nnTanh`, `nnRelu`, `nnIdentity`;
  losses `nnMSE`, `nnBCE`.

## How it works

Training minimizes a loss function by gradient descent. Each step runs a forward
pass to compute the network's output, evaluates the loss against the target,
then propagates the error backward through the layers to accumulate gradients
for every weight and bias. `model_apply_gradients` updates the parameters:

```
theta_new = theta_old - (learning_rate / batch_size) * gradient
```

Gradients are accumulated across a mini-batch and applied once per batch;
`model_train` handles epoch iteration, batching, optional shuffling, and
periodic loss logging.

## Repository structure

- `nn.h` — public header: types, function prototypes, and activation/loss constants.
- `nn.c` — implementation of the full library.
- `xor_test.c` — XOR classification (BCE loss, sigmoid output).
- `parab_test.c` — fitting `y = x^2` (MSE regression).
- `abs_test.c` — fitting `y = |x|` (MSE regression).
- `complex_root_test.c` — approximating the complex square root (one input,
  two outputs for the real and imaginary parts).
- `Makefile` — builds and runs the examples.

## Limitations

- The neuron-centric design is considerably slower than a vectorized,
  matrix-based implementation for large networks or datasets.
- Only SGD and mini-batch SGD are provided; there are no advanced optimizers
  (Adam, RMSprop), regularization (dropout, L1/L2), or non-dense layer types
  (convolutional, recurrent).
- Execution is single-threaded on the CPU.
- Loading saved parameters requires rebuilding the exact model architecture
  first, since activation functions and training configuration are not stored.
