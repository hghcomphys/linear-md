# Linear MD

This repository contains experiments on implementing linear-scaling molecular dynamics (MD) algorithms. 
The focus is on improving performance by using **cell list** and **neighbor list** methods, which reduce the computational complexity of force calculations from *O(N²)* to approximately *O(N)* for short-range interactions.

The repo includes two implementations:

* C++ 
* Python + Numba 


## C++ Implementation


### Build and Run

Compile the code and run the simulation using:

```bash
make
make run
```

This will:

1. Compile `md.cpp`
2. Produce the executable `md.x`
3. Run the simulation with default parameters

You can adjust simulation parameters directly in `md.cpp`.


### Performance Profiling

To analyze performance on Linux systems, you can profile the executable using `perf`:

```bash
make profile
```

This typically runs `perf record` and/or `perf report` on `md.x`.


## Python (Numba) Implementation

The Python version uses **Numba** to JIT-compile performance-critical sections.

### Environment Setup

To set up and activate the environment:

```bash
pixi shell
```

This will install and activate all required Python packages, including Numba and NumPy.

### Run the Simulation

Once the environment is active, run:

```bash
python md.py
```

This executes the MD simulation using the Numba-accelerated implementation.


## Results
The performance evaluation shows that the Python Numba implementation achieves performance comparable to the C++ version, demonstrating that JIT compilation can effectively narrow the gap between high-level and low-level languages for molecular dynamics workloads.
That said, the C++ implementation remains approximately 1.5× faster than the Python version for a system of 4000 atoms and 1000 time steps. 

Profiling indicates that this advantage is primarily due to the lower number of executed instructions in the C++ code path, rather than fundamental differences in memory behavior or CPU efficiency. 
Despite the higher instruction count in Python, key hardware-level metrics remain remarkably similar between the two implementations. 
In particular, instructions per cycle (IPC) are comparable (around 1.5×), and cache miss rates fall within the same range (approximately 11–15%). 
This suggests that both versions exhibit similar data access patterns and cache locality, and that the performance gap is largely attributable to language-level overhead rather than architectural inefficiencies.