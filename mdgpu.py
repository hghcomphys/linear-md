import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import math
from typing import NamedTuple

import numpy as np
from numba import jit
from numba import cuda
from numpy.typing import NDArray

# from numba.cuda.cudadrv.devicearray import DeviceNDArray
DeviceNDArray = NDArray

FLOAT = np.float64
INT = np.int32
ORDER = "F"
Array = NDArray[FLOAT]
DeviceArray = DeviceNDArray

K_B = 8.617343e-5  # Boltzmann's constant in natural unit
TIME_UNIT_CONVERSION = 1.018051e1  # from natural unit to fs


class Parameters(NamedTuple):
    num_atoms: INT
    time_step: FLOAT
    temperature: FLOAT
    cutoff_radius: FLOAT
    skin_radius: FLOAT
    neighbor_max_atoms: INT


class Particles(NamedTuple):
    mass: DeviceArray
    position: DeviceArray
    velocity: DeviceArray
    force: DeviceArray
    box: DeviceArray
    position_old: DeviceArray


class CellList(NamedTuple):
    num_atoms_per_cell: DeviceNDArray
    atom_index: DeviceNDArray
    sizes: tuple


class NeighborList(NamedTuple):
    num_neighbors_per_atom: DeviceNDArray
    neighbor_index: DeviceNDArray


def main() -> None:
    ATOMIC_MASS = {"Ar": 40.0}
    atoms, box = read_xyz("argon.xyz")
    params = Parameters(
        num_atoms=INT(len(atoms)),
        time_step=FLOAT(0.5 / TIME_UNIT_CONVERSION),
        temperature=FLOAT(60.0),
        cutoff_radius=FLOAT(9.0),
        skin_radius=FLOAT(1.0),
        neighbor_max_atoms=INT(500),
    )
    mass = np.array([ATOMIC_MASS[a[0]] for a in atoms])
    position = np.array([a[1:4] for a in atoms], dtype=FLOAT, order=ORDER)
    velocity = initialize_velocity(mass, params.temperature)
    force = np.empty((len(atoms), 3), dtype=FLOAT, order=ORDER)
    box = np.array(box, dtype=FLOAT).reshape(9)
    particles = Particles(
        mass=cuda.to_device(mass),
        position=cuda.to_device(position),
        velocity=cuda.to_device(velocity),
        force=cuda.to_device(force),
        box=cuda.to_device(box),
        position_old=cuda.to_device(position.copy()),
    )
    print(f"Number of atoms: {params.num_atoms}")
    print(f"Box matrix H:\n{particles.box.copy_to_host().reshape(3, 3)}")

    # simulate(params, particles, steps=1001)
    cells, neighbors = allocate_neighbors(params, box)
    update_neighbors(particles, params, cells, neighbors)
    # print(cells.copy_to_host())

    print("Done.")


@jit(nopython=True, inline="always")
def index3d(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@cuda.jit(device=True, inline=True)
def index3d_kernel(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return int(ic[0] + nc[0] * (ic[1] + nc[1] * ic[2]))


@cuda.jit(device=True, inline=True)
def index1d_kernel(idx: int, nc: tuple[int, int, int]) -> tuple[int, int, int]:
    ic0 = idx - nc[0] * math.floor(idx / nc[0])
    tmp = math.floor(idx / nc[0])
    ic1 = tmp - nc[1] * math.floor(tmp / nc[1])
    ic2 = math.floor(idx / (nc[0] * nc[1]))
    return ic0, ic1, ic2


@cuda.jit
def reset_neighbor_list(
    params: Parameters,
    neighbors: NeighborList,
) -> None:
    n = cuda.grid(1)
    if n < params.num_atoms:
        neighbors.num_neighbors_per_atom[n] = 0


@cuda.jit
def update_cell_list(
    particles: Particles,
    params: Parameters,
    cells: CellList,
) -> None:
    position, box = particles.position, particles.box
    length = box[0], box[4], box[8]
    nc = cells.sizes
    n = cuda.grid(1)

    if n < params.num_atoms:
        ic0 = math.floor(position[n, 0] * nc[0] / length[0])
        ic1 = math.floor(position[n, 1] * nc[1] / length[1])
        ic2 = math.floor(position[n, 2] * nc[2] / length[2])
        idx = index3d_kernel((ic0, ic1, ic2), nc)
        cells.atom_index[idx, cells.num_atoms_per_cell[idx]] = n
        cuda.atomic.add(cells.num_atoms_per_cell, idx, 1)


@cuda.jit
def zero_kernel(arr: DeviceArray) -> None:
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = 0.0


def zero_device_array(arr: DeviceArray) -> None:
    threads = 32
    blocks = math.ceil(arr.size / threads)
    zero_kernel[blocks, threads](arr)


@cuda.jit(device=True, inline=True)
def apply_pbc(
    rij: FLOAT,
    length: FLOAT,
) -> FLOAT:
    if rij >= 0.5 * length:
        rij -= length
    elif rij <= -0.5 * length:
        rij += length
    return rij


@cuda.jit
def update_neighbor_list(
    particles: Particles,
    cells: CellList,
    neighbors: NeighborList,
    neighbor_cutoff2: FLOAT,
    cell_max_atoms: INT,
    num_cells: INT,
) -> None:

    nc = cells.sizes
    box, position = particles.box, particles.position
    length = box[0], box[4], box[8]

    idx_i, i = cuda.grid(2)
    
    if idx_i < num_cells and i < cell_max_atoms and i < cells.num_atoms_per_cell[idx_i]:

        ni = cells.atom_index[idx_i, i]
        ic0, ic1, ic2 = index1d_kernel(idx_i, nc)

        for kc0 in range(ic0 - 1, ic0 + 2):
            if (kc0 == -1) or (kc0 == nc[0]):
                jc0 = (kc0 + nc[0]) % nc[0]
            else:
                jc0 = kc0

            for kc1 in range(ic1 - 1, ic1 + 2):
                if (kc1 == -1) or (kc1 == nc[1]):
                    jc1 = (kc1 + nc[1]) % nc[1]
                else:
                    jc1 = kc1

                for kc2 in range(ic2 - 1, ic2 + 2):
                    if (kc2 == -1) or (kc2 == nc[2]):
                        jc2 = (kc2 + nc[2]) % nc[2]
                    else:
                        jc2 = kc2

                    idx_j = index3d((jc0, jc1, jc2), nc)
                    for j in range(cells.num_atoms_per_cell[idx_j]):
                        nj = cells.atom_index[idx_j, j]

                        # if ni < nj:
                        dx = apply_pbc(
                            position[nj, 0] - position[ni, 0],
                            FLOAT(length[0]),
                        )
                        dy = apply_pbc(
                            position[nj, 1] - position[ni, 1],
                            length[1],
                        )
                        dz = apply_pbc(
                            position[nj, 2] - position[ni, 2],
                            length[2],
                        )
                        r2 = dx * dx + dy * dy + dz * dz
                        #
                        if r2 < neighbor_cutoff2:
                            print("Neighbor", ni, nj)
                            neighbors.neighbor_index[
                                ni,
                                neighbors.num_neighbors_per_atom[ni],
                            ] = nj
                            cuda.atomic.add(neighbors.num_neighbors_per_atom, ni, 1)
                            # neighbors.num_neighbors_per_atom[ni] += 1


# @cuda.jit
def update_neighbors(
    particles: Particles,
    params: Parameters,
    cells: CellList,
    neighbors: NeighborList,
) -> None:

    neighbor_cutoff2 = (params.cutoff_radius + params.skin_radius) ** 2
    num_cells = math.prod(cells.sizes)
    cell_max_atoms = 54

    zero_device_array(cells.num_atoms_per_cell)
    zero_device_array(neighbors.num_neighbors_per_atom)

    threads = 32
    blocks = math.ceil(params.num_atoms / threads)
    update_cell_list[blocks, threads](particles, params, cells)
    # print(cells.num_atoms_per_cell.copy_to_host())

    # CUDA grid 1
    threads = 32, 32
    blocks = (
        math.ceil(num_cells / threads[0]),
        math.ceil(cell_max_atoms / threads[1]),
        # math.ceil(params.num_atoms / threads[1])
    )
    update_neighbor_list[blocks, threads](
        particles,
        cells,
        neighbors,
        FLOAT(neighbor_cutoff2),
        INT(cell_max_atoms),
        INT(num_cells),
    )
    print(neighbors.num_neighbors_per_atom.copy_to_host())

    # # Update old position
    # for n in range(params.num_atoms):
    #     particles.position_old[n, 0] = position[n, 0]
    #     particles.position_old[n, 1] = position[n, 1]
    #     particles.position_old[n, 2] = position[n, 2]


def allocate_neighbors(
    params: Parameters,
    box: Array,
) -> tuple[CellList, NeighborList]:
    cell_length = params.cutoff_radius
    length = box[0], box[4], box[8]
    cell_sizes = (
        int(length[0] / cell_length),
        int(length[1] / cell_length),
        int(length[2] / cell_length),
    )
    for size in cell_sizes:
        assert size > 2, print(f"{cell_sizes=}")
    # Cell list
    system_volume = math.prod(length)
    cell_volume = cell_length**3
    cell_max_atoms = 3 * int(params.num_atoms / system_volume * cell_volume)
    print("Cell max atoms:", cell_max_atoms)

    cell_atom_index = np.empty(
        shape=(math.prod(cell_sizes), cell_max_atoms),
        dtype=INT,
    )
    cell_num_atoms = np.empty(
        shape=(math.prod(cell_sizes),),
        dtype=INT,
    )
    cells = CellList(
        sizes=cell_sizes,
        num_atoms_per_cell=cuda.to_device(cell_num_atoms),
        atom_index=cuda.to_device(cell_atom_index),
    )
    # Neighbor list
    neighbor_index = np.empty(
        shape=(params.num_atoms, params.neighbor_max_atoms),
        dtype=INT,
    )
    neighbor_num_atoms = np.empty(
        shape=(params.num_atoms,),
        dtype=INT,
    )
    neighbors = NeighborList(
        num_neighbors_per_atom=cuda.to_device(neighbor_num_atoms),
        neighbor_index=cuda.to_device(neighbor_index),
    )
    print(f"Cell length:", params.cutoff_radius)
    print(f"Cell sizes: {cells.sizes}")

    return (
        cells,
        neighbors,
    )


def initialize_velocity(
    mass: Array,
    temperature: float,
    seed: int = 1234,
) -> Array:
    np.random.seed(seed)
    velocity = np.zeros(shape=(len(mass), 3), dtype=FLOAT, order=ORDER)
    for i, m in enumerate(mass):
        sigma = np.sqrt(K_B * temperature / m)
        velocity[i] = np.random.normal(0.0, sigma, size=3)
    vcm = (mass.reshape(-1, 1) * velocity).sum(axis=0) / mass.sum()
    velocity -= vcm
    return velocity


def read_xyz(filename: str) -> tuple:
    atoms, box = [], []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            box = [float(b) for b in f.readline().strip().split()]
            atoms = []
            for _ in range(num_atoms):
                parts = f.readline().split()
                element = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append((element, x, y, z))
    return atoms, box


if __name__ == "__main__":
    main()
