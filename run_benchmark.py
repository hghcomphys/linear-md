import subprocess
import time
import csv


# Define the benchmark cases here
SCALING_BENCHMARKS = [
    ((6, 6, 6), 864),
    ((7, 7, 7), 1372),
    ((8, 8, 8), 2048),
    ((9, 9, 9), 2916),
    ((10, 10, 10), 4000),
    ((20, 10, 10), 8000),
    # ((20, 20, 10), 16000),
    # ((20, 20, 20), 32000),
    # ((40, 20, 20), 64000),
    # ((40, 40, 20), 128000),
    # ((40, 40, 40), 256000),
    # ((80, 40, 40), 512000),
    # ((80, 80, 40), 1024000),
]
PERFORMANCE_BENCHMARKS = [
    # ((6, 6, 6), 864),
    #((7, 7, 7), 1372),
    ((8, 8, 8), 2048),
    ((9, 9, 9), 2916),
    ((10, 10, 10), 4000),
    ((20, 10, 10), 8000),
    ((20, 20, 10), 16000),
    ((20, 20, 20), 32000),
    ((40, 20, 20), 64000),
    # ((40, 40, 20), 128000),
    # ((40, 40, 40), 256000),
    # ((80, 40, 40), 512000),
    # ((80, 80, 40), 1024000),
]

# Output files
SCALING_OUTPUT_CSV = "benchmark_results_scaling.csv"
PERFORMANCE_OUTPUT_CSV = "benchmark_results_performance.csv"

# Commands
GENERATE_LATTICE = ["python", "generate_lattice.py"]
LINEAR_MD_COMMAND = ["pixi", "run", "python", "md.py"]
SIMPLE_MD_COMMAND = ["pixi", "run", "python", "simple_md.py"]
PYTHON_COMMAND = ["pixi", "run", "python", "md.py"]
CPP_COMMAND = ["make", "run"]


def run_command(cmd, cwd=None):
    """
    Run a shell command and measure elapsed wall-clock time.
    Raises CalledProcessError if the command fails.
    """
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    end = time.perf_counter()
    return end - start


def run_scaling_benchmarks():
    results = []

    for (nx, ny, nz), natoms in SCALING_BENCHMARKS:
        print(f"\n=== Benchmark nx={nx}, ny={ny}, nz={nz} | atoms={natoms} ===")

        # 1. Generate lattice
        print("Running generate_lattice.py...")
        run_command(GENERATE_LATTICE + ["--lattice", str(nx), str(ny), str(nz)])

        # 2. Run linear md 
        print(f"Running {' '.join(LINEAR_MD_COMMAND)}")
        linear_md_time = run_command(LINEAR_MD_COMMAND)
        print(f"Linear MD elapsed time: {linear_md_time:.6f} s")

        # 3. Run simple md 
        print(f"Running {' '.join(SIMPLE_MD_COMMAND)}")
        simple_md_time = run_command(SIMPLE_MD_COMMAND)
        print(f"Simple MD elapsed time: {simple_md_time:.6f} s")

        results.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "atoms": natoms,
                "linear_md_time_sec": linear_md_time,
                "simple_md_time_sec": simple_md_time,
            }
        )

    print(f"\nWriting results to {SCALING_OUTPUT_CSV}")
    with open(SCALING_OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nx", "ny", "nz", "atoms", "linear_md_time_sec", "simple_md_time_sec"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Scaling Benchmark Summary ===")
    for r in results:
        print(
            f"nx={r['nx']:4d}, ny={r['ny']:4d}, nz={r['nz']:4d} | "
            f"atoms={r['atoms']:10d} | "
            f"Linear MD: {r['linear_md_time_sec']:10.6f} s | "
            f"Simple MD: {r['simple_md_time_sec']:10.6f} s"
        )


def run_performance_benchmarks():
    results = []

    for (nx, ny, nz), natoms in PERFORMANCE_BENCHMARKS:
        print(f"\n=== Benchmark nx={nx}, ny={ny}, nz={nz} | atoms={natoms} ===")

        # 1. Generate lattice
        print("Running generate_lattice.py...")
        run_command(GENERATE_LATTICE + ["--lattice", str(nx), str(ny), str(nz)])

        # 2. Run C++ code
        print(f"Running C++ MD {' '.join(CPP_COMMAND)}")
        cpp_time = run_command(CPP_COMMAND)
        print(f"C++ elapsed time: {cpp_time:.6f} s")

        # 3. Run Python code
        print("Running Python MD {' '.join(PYTHON_COMMAND)}")
        py_time = run_command(PYTHON_COMMAND)
        print(f"Python elapsed time: {py_time:.6f} s")

        results.append(
            {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "atoms": natoms,
                "cpp_time_sec": cpp_time,
                "python_time_sec": py_time,
            }
        )

    print(f"\nWriting results to {PERFORMANCE_OUTPUT_CSV}")
    with open(PERFORMANCE_OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nx", "ny", "nz", "atoms", "cpp_time_sec", "python_time_sec"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Performance Benchmark Summary ===")
    for r in results:
        print(
            f"nx={r['nx']:4d}, ny={r['ny']:4d}, nz={r['nz']:4d} | "
            f"atoms={r['atoms']:10d} | "
            f"C++: {r['cpp_time_sec']:10.6f} s | "
            f"Python: {r['python_time_sec']:10.6f} s"
        )


if __name__ == "__main__":
    run_scaling_benchmarks()
    # run_performance_benchmarks()
