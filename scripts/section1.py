#!/usr/bin/env python3

import numpy as np

if __name__ == "__main__":
    print("Hello world")

    np.random.seed(42)
    A = np.random.normal(size=(4, 4))
    B = np.random.normal(size=(4, 2))
    print(A@B)

    np.random.seed(42)
    x = np.random.normal(size=(4, 10))

    result1 = np.square(x[:, None, :] - x[None, :, :]).sum(axis=2)
    print(result1)