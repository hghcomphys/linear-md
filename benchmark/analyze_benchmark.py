# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("classic")
mpl.style.use("classic")
mpl.style.use("bmh")

# %%
import cpuinfo

info = cpuinfo.get_cpu_info()["brand_raw"]
# info = "Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz"
print("CPU Model:", info)

# %%

# Load CSV file
df = pd.read_csv("benchmark_results.csv")
print("=== Benchmark Results ===")
print(df)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(df["atoms"], df["cpp_time_sec"], label="C++")
plt.plot(df["atoms"], df["python_time_sec"], label="Python (Numba)")
plt.xlabel("Number of Atoms", fontsize=13)
plt.ylabel("Execution Time (s)", fontsize=13)
plt.title(f"MD Simulation (1000 Time Steps)\n{info}", fontsize=13)
plt.legend(
    ["C++", "Python (Numba)"], title="Implementation", loc="upper left", fontsize=13
)
plt.tight_layout()
plt.savefig("scaling.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5))
df["runs"] = [f"{n // 1000}k" for n in df["atoms"].tolist()]
df.plot(
    kind="bar", x="runs", y=["cpp_time_sec", "python_time_sec"], logy=True, rot=0, ax=ax
)

plt.xlabel("Number of Atoms", fontsize=13)
plt.ylabel("Execution Time (s)", fontsize=13)
plt.title(f"MD Simulation (1000 Time Steps)\n{info}", fontsize=13)
plt.legend(
    ["C++", "Python (Numba)"], title="Implementation", loc="upper left", fontsize=13
)
plt.tight_layout()
plt.savefig("performance.png", dpi=300)
plt.show()
