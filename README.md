# Linear MD

This repository shows my experiments on implementing linear-scaling molecular dynamics (MD) simulator. 
The focus is on improving performance by using **cell list** and **neighbor list** methods in comparison to a [simple MD](https://github.com/hghcomphys/simulational-physics/tree/master/simple_md), which reduce the computational complexity of force calculations from *O(N²)* to approximately *O(N)* for short-range interactions.

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

This typically runs linux `perf` command on `md.x` executable.

Profiling results for a simulation with 4000 atoms and 1000 time steps:

```text
 Performance counter stats for './md.x':

    19.647.672.034      cycles
    35.184.297.731      instructions              #    1,79  insn per cycle
       476.082.499      cache-references
        79.637.529      cache-misses              #   16,728 % of all cache refs

       4,929873145 seconds time elapsed

       4,929639000 seconds user
       0,000000000 seconds sys
```


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

### Performance Profiling

Hardware-level profiling

```bash
perf stat -e cycles,instructions,cache-references,cache-misses python md.py 
```

Profiling results for a simulation with 4000 atoms and 1000 time steps:

```text
 Performance counter stats for 'python md.py':

    34.785.338.573      cycles
    52.160.003.366      instructions              #    1,50  insn per cycle
       600.427.540      cache-references
       141.996.840      cache-misses              #   23,649 % of all cache refs

       8,257291531 seconds time elapsed

       9,022870000 seconds user
       0,104217000 seconds sys
```


Python specific profiling using `scalene`

```bash
scalene python md.py
```

## Benchmarks

Run benchmark via running the following python script
```bash
python run_benchmark.py
```

This will generate lattices with varying numbers of atoms (`argon.xyz`) and run molecular dynamics simulations for 1000 time steps using both the C++ and Python (Numba) implementations.

Using the Python implementation, it is feasible to simulate one million atoms on a standard laptop.

### Results

(Linear) scaling

<img src="benchmark/scaling.png" width="400">

Performance comparison

<img src="benchmark/performance.png" width="400">


