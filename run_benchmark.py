import subprocess
import time
import csv

# Define the benchmark cases here
BENCHMARKS = [
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

# Output file
OUTPUT_CSV = "benchmark_results.csv"

# Commands
GENERATE_LATTICE = ["python", "generate_lattice.py"]
CPP_COMMAND = ["make", "run"]
PYTHON_COMMAND = ["pixi", "run", "python", "md.py"]


def run_command(cmd, cwd=None):
    """
    Run a shell command and measure elapsed wall-clock time.
    Raises CalledProcessError if the command fails.
    """
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    end = time.perf_counter()
    return end - start


def main():
    results = []

    for (nx, ny, nz), natoms in BENCHMARKS:
        print(f"\n=== Benchmark nx={nx}, ny={ny}, nz={nz} | atoms={natoms} ===")

        # 1. Generate lattice
        print("Running generate_lattice.py...")
        run_command(GENERATE_LATTICE + ["--lattice", str(nx), str(ny), str(nz)])

        # 2. Run C++ code
        print("Running C++ (make run)...")
        cpp_time = run_command(CPP_COMMAND)
        print(f"C++ elapsed time: {cpp_time:.6f} s")

        # 3. Run Python code
        print("Running Python (pixi run python md.py)...")
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

    print(f"\nWriting results to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nx", "ny", "nz", "atoms", "cpp_time_sec", "python_time_sec"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Benchmark Summary ===")
    for r in results:
        print(
            f"nx={r['nx']:4d}, ny={r['ny']:4d}, nz={r['nz']:4d} | "
            f"atoms={r['atoms']:10d} | "
            f"C++: {r['cpp_time_sec']:10.6f} s | "
            f"Python: {r['python_time_sec']:10.6f} s"
        )


if __name__ == "__main__":
    main()
