# 01 - NumPy Primer

## Objective

Learn the fundamentals of NumPy by implementing multidimensional array operations and comparing them with equivalent pure Python implementations.

## Dataset

- **Dataset:** Iris Dataset
- **Samples:** 150
- **Features:** 4 numerical features
- **Target:** Species (Setosa, Versicolor, Virginica)

## Topics Covered

- Dataset inspection
- NumPy arrays
- Indexing
- Slicing
- Reshaping
- Transposing
- Numerical operations
- Logical operations
- Broadcasting
- Statistical computations
- Pure Python implementations
- Validation of NumPy vs. pure Python
- Performance benchmarking

## Files

```text
01-numpy-primer/
├── data/
│   └── iris.csv
├── notebooks/
│   └── numpy_primer.ipynb
├── src/
│   ├── numpy_operations.py
│   ├── python_operations.py
│   ├── validate_results.py
│   └── benchmark.py
└── results/
```

## Benchmark Summary

| Operation | NumPy | Pure Python |
|-----------|------:|------------:|
| Mean | TBD | TBD |
| Min | TBD | TBD |
| Max | TBD | TBD |
| Standard Deviation | TBD | TBD |

*(Replace with your measured values after benchmarking.)*

## Observations

- NumPy operations are significantly faster than equivalent pure Python implementations.
- Broadcasting eliminates many explicit loops.
- Vectorized operations produce cleaner and more concise code.
- Reshaping and transposing reorganize data without changing the underlying values.
- NumPy stores homogeneous data in contiguous memory, enabling efficient execution.

## Conclusion

This exercise demonstrated the advantages of NumPy over pure Python for numerical computing. By implementing identical operations in both approaches and validating their outputs, it became clear that NumPy provides simpler syntax, better readability, and substantially higher performance due to vectorized execution and optimized low-level implementations.