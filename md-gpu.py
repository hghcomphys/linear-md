import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import math
from typing import NamedTuple, TextIO

import numpy as np
from numba import cuda
from numpy.typing import NDArray

FLOAT = np.float64
Array = NDArray[FLOAT]
ORDER = "F"


class SimulationParameters(NamedTuple):
    num_atoms: int
    time_step: float
    box_length: float
    atom_spacing: float
    max_velocity: float
    cutoff: float


num_atoms = 4000
atom_spacing = 2.5
params = SimulationParameters(
    num_atoms=num_atoms,
    time_step=0.0001,
    box_length=math.sqrt(num_atoms) * atom_spacing,
    atom_spacing=atom_spacing,
    max_velocity=10.0,
    cutoff=15.0,
)


class Particles(NamedTuple):
    position: Array
    velocity: Array
    force: Array


class CellList(NamedTuple):
    max_num_atoms: int
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int]


class NeighborList(NamedTuple):
    max_num_neighbors: int
    num_neighbors_per_atom: Array
    neighbor_index: Array


def initialize_position(
    params: SimulationParameters,
) -> Array:
    atom_spacing = params.atom_spacing
    box_length = params.box_length
    num_atoms_per_row = math.ceil(math.sqrt(params.num_atoms))
    position = np.empty(shape=(params.num_atoms, 2), dtype=FLOAT)
    atom_index = 0
    for index_j in range(num_atoms_per_row):
        for index_i in range(num_atoms_per_row):
            if atom_index < position.shape[0]:
                position[atom_index, 0] = (index_i + 0.5) * atom_spacing % box_length
                position[atom_index, 1] = (index_j + 0.5) * atom_spacing % box_length
            atom_index += 1
    return position


def initialize_velocity(
    params: SimulationParameters,
    seed: int = 1234,
) -> Array:
    np.random.seed(seed)
    velocity = np.random.randn(params.num_atoms, 2).astype(FLOAT)
    velocity *= params.max_velocity
    velocity -= velocity.mean(axis=0)
    return velocity


position = initialize_position(params)  # .ravel(order)
velocity = initialize_velocity(params)  # .ravel(order)
position_d = cuda.device_array(position.shape, dtype=FLOAT, order=ORDER)
velocity_d = cuda.device_array(velocity.shape, dtype=FLOAT, order=ORDER)
force_d = cuda.device_array((num_atoms, 2), dtype=FLOAT, order=ORDER)
cuda.to_device(position, to=position_d)
cuda.to_device(velocity, to=velocity_d)
particles = Particles(
    position_d,
    velocity_d,
    force_d,
)
print("Number of atoms:", len(position))


def initialize_neighbors(
    params: SimulationParameters,
) -> tuple[CellList, NeighborList]:
    cell_sizes = (
        math.floor(params.box_length / params.cutoff),
        math.floor(params.box_length / params.cutoff),
    )
    cell_max_atoms = 100
    cell_atom_index = cuda.device_array(
        math.prod(cell_sizes) * cell_max_atoms,
        dtype=np.int32,
    )
    cell_num_atoms = cuda.device_array(
        math.prod(cell_sizes),
        dtype=np.int32,
    )
    cell_list = CellList(
        max_num_atoms=cell_max_atoms,
        num_atoms_per_cell=cell_num_atoms,
        atom_index=cell_atom_index,
        sizes=cell_sizes,
    )

    neighbor_max = 100
    neighbor_index = cuda.device_array(
        params.num_atoms * neighbor_max,
        dtype=np.int32,
    )
    neighbor_num_atoms = cuda.device_array(
        params.num_atoms,
        dtype=np.int32,
    )
    neighbor_list = NeighborList(
        max_num_neighbors=neighbor_max,
        neighbor_index=neighbor_index,
        num_neighbors_per_atom=neighbor_num_atoms,
    )
    return cell_list, neighbor_list


cell_list, neighbor_list = initialize_neighbors(params)
# print(cell_list)


@cuda.jit
def clear_grid(cell_list: CellList) -> None:
    start, stride = cuda.grid(1), cuda.gridsize(1)
    for index in range(start, math.prod(cell_list.sizes), stride):
        cell_list.num_atoms_per_cell[index] = 0


@cuda.jit(device=True, inline=True)
def index3d(ic: tuple[int, int, int], nc: tuple[int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@cuda.jit(device=True, inline=True)
def index2d(ic: tuple[int, int], nc: tuple[int, int]) -> int:
    return ic[0] + nc[0] * ic[1]


@cuda.jit
def update_grid(
    particles: Particles,
    params: SimulationParameters,
    cell_list: CellList,
) -> None:

    start, stride = cuda.grid(1), cuda.gridsize(1)
    for index_i in range(start, params.num_atoms, stride):

        nc = cell_list.sizes
        length = params.box_length
        ic1 = math.floor(particles.position[index_i, 0] * nc[0] / length)
        ic2 = math.floor(particles.position[index_i, 1] * nc[1] / length)

        # must be an atomic add!
        ic = ic1, ic2
        idx2d = index2d(ic, nc)
        idx3d = index3d((*ic, cell_list.num_atoms_per_cell[idx2d]), nc)

        cell_list.atom_index[idx3d] = index_i
        cuda.atomic.add(cell_list.num_atoms_per_cell, idx2d, 1)


threads_per_block = 32
blocks_per_grid = math.ceil(math.prod(cell_list.sizes) / threads_per_block)
clear_grid[blocks_per_grid, threads_per_block](cell_list)
blocks_per_grid = math.ceil(params.num_atoms / threads_per_block)
update_grid[blocks_per_grid, threads_per_block](particles, params, cell_list)
cuda.synchronize()
print(cell_list.num_atoms_per_cell.copy_to_host())
# print(cell_list.atom_index.copy_to_host())


@cuda.jit
def compute_force(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    force = cuda.local.array(shape=(2,), dtype=FLOAT)
    start, stride = cuda.grid(1), cuda.gridsize(1)
    for index_i in range(start, params.num_atoms, stride):
        ri = particles.position[index_i]
        fi = particles.force[index_i]
        # calculate per atom force
        force[0], force[1] = 0.0, 0.0
        for index_j in range(params.num_atoms):
            if index_i != index_j:
                rj = particles.position[index_j]
                pair_interaction(ri, rj, params.box_length, force)
        fi[0], fi[1] = force[0], force[1]


@cuda.jit(device=True)
def pair_interaction(
    position_i: Array,
    position_j: Array,
    box_length: FLOAT,
    force: Array,
) -> None:
    rij_x = apply_pbc(position_i[0] - position_j[0], box_length)
    rij_y = apply_pbc(position_i[1] - position_j[1], box_length)
    r2 = rij_x * rij_x + rij_y * rij_y
    r2i = 1.0 / r2
    r6i = r2i * r2i * r2i
    coef = 24.0 * r6i * (2.0 * r6i - 1.0)
    force[0] += rij_x * r2i * coef
    force[1] += rij_y * r2i * coef


@cuda.jit(device=True, inline=True)
def apply_pbc(
    rij: FLOAT,
    box_length: FLOAT,
) -> FLOAT:
    L = box_length
    if rij >= 0.5 * L:
        rij -= L
    elif rij <= -0.5 * L:
        rij += L
    return rij


@cuda.jit
def verlet_integration_position(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    r, v, f = particles
    dt, L = params.time_step, params.box_length
    start, stride = cuda.grid(1), cuda.gridsize(1)
    for index in range(start, params.num_atoms, stride):
        for dim in range(2):
            r[index, dim] = math.fmod(
                r[index, dim]
                + dt * (v[index, dim] + 0.5 * dt * f[index, dim])
                + 10.0 * L,
                L,
            )
            v[index, dim] += 0.5 * dt * f[index, dim]


@cuda.jit
def verlet_integration_velocity(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    v, f = particles.velocity, particles.force
    dt = params.time_step
    start, stride = cuda.grid(1), cuda.gridsize(1)
    for index in range(start, params.num_atoms, stride):
        for dim in range(2):
            v[index, dim] += 0.5 * dt * f[index, dim]


def save(position: Array, file: TextIO) -> None:
    num_atoms = position.shape[0]
    file.write(f"{num_atoms}\n\n")
    for i in range(num_atoms):
        file.write(f"{'He'}\t{position[i, 0]} {position[i, 1]} {0}\n")
    file.flush()


def get_temperature(velocity: Array) -> FLOAT:
    return 0.5 * (velocity**2).sum() / velocity.shape[0]


def simulate(
    params: SimulationParameters,
    particles: Particles,
    steps: int = 1,
    log_frequency: int = 100,
    filename: str = "configurations.xyz",
) -> None:
    print("A GPU-accelerated Molecular Dynamics Simulations")
    print("System: Lennard-Jones particles in 2D")
    print(f"Number of atoms: {params.num_atoms}")
    print("Step    Time         Temperature")
    print("--------------------------------")
    threads_per_block = 32
    blocks_per_grid = math.ceil(params.num_atoms / threads_per_block)
    compute_force[blocks_per_grid, threads_per_block](
        particles, params
    )  # initialize force
    with open(filename, "w") as file:
        # Simulate
        for step in range(steps):
            if step % log_frequency == 0:
                print(
                    f"{step:<7d}"
                    f" {step * params.time_step:<12.8f}"
                    # f" {get_temperature(particles.velocity.copy_to_host()):<.8f}"
                )
                # save(particles.position.copy_to_host(), file)
            # Next time step (update r, v, and F)
            verlet_integration_position[blocks_per_grid, threads_per_block](
                particles, params
            )
            compute_force[blocks_per_grid, threads_per_block](particles, params)
            verlet_integration_velocity[blocks_per_grid, threads_per_block](
                particles, params
            )
    print("Done.")


# simulate(params, particles, steps=1000)
