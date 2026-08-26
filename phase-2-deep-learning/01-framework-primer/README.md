# 01 - Deep Learning Frameworks Primer

## Objective

Understand the role of computation graphs, reverse-mode automatic differentiation, and optimization loops, then compare the same tensor and gradient-descent operations in TensorFlow and PyTorch.

## Concepts

- Tensors, shapes, data types, and element-wise operations
- Computation graphs and dependency tracking
- Backpropagation through the chain rule
- Automatic differentiation in TensorFlow and PyTorch
- Parameters, gradients, and optimization loops
- Framework services that would be required in a custom implementation

## Tasks

- [x] Explain computation graphs
- [x] Explain backpropagation
- [x] Describe what deep learning frameworks provide
- [x] Create tensors from lists of floats in TensorFlow and PyTorch
- [x] Compare four mathematical tensor operations
- [x] Implement gradient descent in pure Python
- [x] Optimize the same function with TensorFlow
- [x] Optimize the same function with PyTorch
- [x] Compare convergence and verify the analytical minimum

## Notes

The experiment minimizes `x² - 12x + 36`, which is `(x - 6)²` and therefore has its unique minimum at `x = 6`. All three implementations start from `x = -8`, use a learning rate of `0.1`, and run for 80 updates so their optimization paths are directly comparable.

TensorFlow and PyTorch both reproduced the analytical gradient of `-12` at `x = 0`. Pure Python, TensorFlow, and PyTorch then reached `x = 5.999999752641` with an objective value of `6.395e-14`, and their recorded loss histories agreed within `1e-10`.

The NVIDIA GeForce RTX 3050 Ti was detected, while the installed framework builds expose CPU execution only. CPU is the appropriate device for this single-scalar workload because accelerator transfer and launch overhead would dominate the calculation.

The executed notebook contains the tensor comparisons, automatic-differentiation checks, convergence plot, assertions, and final observations: [`notebooks/framework_primer.ipynb`](notebooks/framework_primer.ipynb).

## Resources

- [TensorFlow: Introduction to tensors](https://www.tensorflow.org/guide/tensor)
- [TensorFlow: Automatic differentiation](https://www.tensorflow.org/guide/autodiff)
- [PyTorch: Introduction to tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch: Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
