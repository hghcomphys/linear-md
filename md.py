import math
from typing import NamedTuple, TextIO

from numba.parfors.parfor import parfor_insert_dels
import numpy as np
from numba import njit
from numpy.typing import NDArray

FLOAT = np.float64
Array = NDArray[FLOAT]
ORDER = "F"

K_B = 8.617343e-5  # Boltzmann's constant in natural unit
TIME_UNIT_CONVERSION = 1.018051e1  # from natural unit to fs


class SimulationParameters(NamedTuple):
    num_atoms: int
    time_step: float
    max_velocity: float
    cutoff_radius: float
    cell_max_atoms: int
    neighbor_max_atoms: int


class Particles(NamedTuple):
    mass: Array
    position: Array
    velocity: Array
    force: Array
    box: Array


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray


def initialize_velocity(
    params: SimulationParameters,
    seed: int = 1234,
) -> Array:
    np.random.seed(seed)
    velocity = np.random.rand(params.num_atoms, 3).astype(FLOAT)
    velocity -= velocity.mean(axis=0)
    # velocity *= params.max_velocity
    return velocity


def initialize_neighbors(
    params: SimulationParameters,
    box: Array,
) -> tuple[CellList, NeighborList]:
    length = box[0], box[4], box[8]
    cell_sizes = (
        math.floor(length[0] / params.cutoff_radius),
        math.floor(length[1] / params.cutoff_radius),
        math.floor(length[2] / params.cutoff_radius),
    )
    for size in cell_sizes:
        assert size > 2, print(f"{cell_sizes=}")
    # Cell list
    cell_atom_index = np.empty(
        shape=(math.prod(cell_sizes), params.cell_max_atoms),
        dtype=np.int32,
    )
    cell_num_atoms = np.empty(
        math.prod(cell_sizes),
        dtype=np.int32,
    )
    cell_list = CellList(
        sizes=cell_sizes,
        num_atoms_per_cell=cell_num_atoms,
        atom_index=cell_atom_index,
    )
    # Neighbor list
    neighbor_index = np.empty(
        shape=(params.num_atoms, params.neighbor_max_atoms),
        dtype=np.int32,
    )
    neighbor_num_atoms = np.empty(
        params.num_atoms,
        dtype=np.int32,
    )
    neighbor_list = NeighborList(
        num_neighbors_per_atom=neighbor_num_atoms,
        neighbor_index=neighbor_index,
    )
    return (
        cell_list,
        neighbor_list,
    )


@njit
def index(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@njit
def apply_pbc(
    rij: FLOAT,
    length: FLOAT,
) -> FLOAT:
    if rij >= 0.5 * length:
        rij -= length
    elif rij <= -0.5 * length:
        rij += length
    return rij


@njit
def update_grid(
    particles: Particles,
    params: SimulationParameters,
    cells: CellList,
    neighbors: NeighborList,
) -> None:
    nc = cells.sizes
    for ic1 in range(nc[0]):
        for ic2 in range(nc[1]):
            for ic3 in range(nc[2]):
                cells.num_atoms_per_cell[index((ic1, ic2, ic3), nc)] = 0

    box = particles.box
    length = box[0], box[4], box[8]
    cutoff2 = params.cutoff_radius * params.cutoff_radius
    for ni in range(0, params.num_atoms):
        ic1 = math.floor(particles.position[ni, 0] * nc[0] / length[0])
        ic2 = math.floor(particles.position[ni, 1] * nc[1] / length[1])
        ic3 = math.floor(particles.position[ni, 2] * nc[2] / length[2])

        idx = index((ic1, ic2, ic3), nc)
        cells.atom_index[idx, cells.num_atoms_per_cell[idx]] = ni
        cells.num_atoms_per_cell[idx] += 1

    for n in range(params.num_atoms):
        neighbors.num_neighbors_per_atom[n] = 0

    for ic1 in range(nc[0]):
        for ic2 in range(nc[1]):
            for ic3 in range(nc[2]):

                idx_i = index((ic1, ic2, ic3), nc)
                for i in range(cells.num_atoms_per_cell[idx_i]):
                    ni = cells.atom_index[idx_i, i]

                    for kc1 in range(ic1 - 1, ic1 + 2):
                        if (kc1 == -1) or (kc1 == nc[0]):
                            jc1 = (kc1 + nc[0]) % nc[0]
                        else:
                            jc1 = kc1

                            for kc2 in range(ic2 - 1, ic2 + 2):
                                if (kc2 == -1) or (kc2 == nc[1]):
                                    jc2 = (kc2 + nc[1]) % nc[1]
                                else:
                                    jc2 = kc2

                                    for kc3 in range(ic3 - 1, ic3 + 2):
                                        if (kc3 == -1) or (kc3 == nc[2]):
                                            jc3 = (kc3 + nc[2]) % nc[2]
                                        else:
                                            jc3 = kc3

                                        idx_j = index((jc1, jc2, jc3), nc)
                                        for j in range(cells.num_atoms_per_cell[idx_j]):
                                            nj = cells.atom_index[idx_j, j]

                                            if ni < nj:
                                                dx = apply_pbc(
                                                    particles.position[nj, 0]
                                                    - particles.position[ni, 0],
                                                    length[0],
                                                )
                                                dy = apply_pbc(
                                                    particles.position[nj, 1]
                                                    - particles.position[ni, 1],
                                                    length[1],
                                                )
                                                dz = apply_pbc(
                                                    particles.position[nj, 2]
                                                    - particles.position[ni, 2],
                                                    length[2],
                                                )
                                                r2 = dx * dx + dy * dy + dz * dz
                                                if r2 < cutoff2:
                                                    neighbors.neighbor_index[
                                                        ni,
                                                        neighbors.num_neighbors_per_atom[
                                                            ni
                                                        ],
                                                    ] = nj
                                                    neighbors.num_neighbors_per_atom[
                                                        ni
                                                    ] += 1
                                                    neighbors.neighbor_index[
                                                        nj,
                                                        neighbors.num_neighbors_per_atom[
                                                            nj
                                                        ],
                                                    ] = ni
                                                    neighbors.num_neighbors_per_atom[
                                                        nj
                                                    ] += 1


@njit
def compute_force(
    particles: Particles,
    params: SimulationParameters,
    neighbors: NeighborList,
    potential_energy: Array,
    cutoff2: float,
) -> None:
    epsilon = 1.032e-2
    sigma = 3.405
    box = particles.box
    force = particles.force
    position = particles.position

    potential_energy[0] = 0.0
    for ni in range(params.num_atoms):
        fi = particles.force[ni]
        fi[0] = fi[1] = fi[2] = 0.0

    for ni in range(params.num_atoms):
        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            if ni < nj:
                length = box[0], box[4], box[8]
                dx = apply_pbc(position[nj, 0] - position[ni, 0], length[0])
                dy = apply_pbc(position[nj, 1] - position[ni, 1], length[1])
                dz = apply_pbc(position[nj, 2] - position[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue
                r2i = (sigma * sigma) / r2
                r6i = r2i * r2i * r2i
                coef = 24.0 * epsilon * r6i * (2.0 * r6i - 1.0)
                force[ni, 0] += dx * r2i * coef
                force[ni, 1] += dy * r2i * coef
                force[ni, 2] += dz * r2i * coef
                force[nj, 0] -= dx * r2i * coef
                force[nj, 1] -= dy * r2i * coef
                force[nj, 2] -= dz * r2i * coef
                potential_energy[0] += 4.0 * epsilon * (r6i * r6i - r6i)

@njit
def verlet_integration_position(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    m, r, v, f, box = particles
    dt = params.time_step
    length = box[0], box[4], box[8]
    for index in range(0, params.num_atoms):
        for dim in range(3):
            r[index, dim] = np.fmod(
                r[index, dim]
                + dt * (v[index, dim] + 0.5 * dt / m[index] * f[index, dim])
                + 10.0 * length[dim],
                length[dim],
            )
            v[index, dim] += 0.5 * dt / m[index] * f[index, dim]


@njit
def verlet_integration_velocity(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    m, _, v, f, _ = particles
    dt = params.time_step
    for index in range(params.num_atoms):
        for dim in range(3):
            v[index, dim] += 0.5 * dt / m[index] * f[index, dim]


def save(position: Array, file: TextIO) -> None:
    num_atoms = position.shape[0]
    file.write(f"{num_atoms}\n\n")
    for i in range(num_atoms):
        file.write(f"{'Ar'}\t{position[i, 0]} {position[i, 1]} {position[i, 2]}\n")
    file.flush()


def get_kinetic_energy(particles: Particles) -> float:
    return 0.5 * (particles.mass.reshape(-1, 1) * particles.velocity**2).sum()


def get_temperature(particles: Particles) -> float:
    natoms = len(particles.velocity)
    return 2.0 / (3 * natoms * K_B) * get_kinetic_energy(particles)


def simulate(
    params: SimulationParameters,
    particles: Particles,
    steps: int = 1,
    log_frequency: int = 100,
    filename: str = "out.xyz",
) -> None:
    cells, neighbors = initialize_neighbors(params, box)
    cutoff2 = params.cutoff_radius * params.cutoff_radius
    update_grid(particles, params, cells, neighbors)
    print(neighbors.num_neighbors_per_atom)
     # print(neighbors.neighbor_index)
    potential_energy = np.empty(1, dtype=FLOAT)
    compute_force(particles, params, neighbors, potential_energy, cutoff2)
    with open(filename, "w") as file:
        # Simulate
        for step in range(steps):
            if step % log_frequency == 0:
                print(
                    f"{step:<7d}"
                    # f" {step * params.time_step:<12.8f}"
                    f" {get_temperature(particles):<.8f}"
                    f" {get_kinetic_energy(particles):<.8f}"
                    f" {potential_energy[0]:<.8f}"
                    f" {neighbors.num_neighbors_per_atom.mean():<.2f}"
                )
                # save(particles.position, file)
            # Next time step (update r, v, and F)
            verlet_integration_position(particles, params)
            update_grid(particles, params, cells, neighbors)
            compute_force(particles, params, neighbors, potential_energy, cutoff2)
            verlet_integration_velocity(particles, params)


def read_xyz(filename: str) -> tuple:
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

    atoms, box = read_xyz("Ar.xyz")
    # atoms = atoms[:5]
    print(f"{box=}")
    params = SimulationParameters(
        num_atoms=len(atoms),
        time_step=0.5 / TIME_UNIT_CONVERSION,
        max_velocity=60.0,
        cutoff_radius=9.0,
        cell_max_atoms=1000,
        neighbor_max_atoms=500,
    )
    position = np.array([a[1:4] for a in atoms])
    velocity = initialize_velocity(params)
    force = np.empty((len(atoms), 3), dtype=FLOAT, order=ORDER)
    box = np.array(box, dtype=FLOAT).reshape(9)
    particles = Particles(
        mass=np.array([{"Ar": 40.0}[a[0]] for a in atoms]),
        position=position.astype(FLOAT, order=ORDER),
        velocity=velocity.astype(FLOAT, order=ORDER),
        force=force,
        box=box,
    )
    print("A Molecular Dynamics Simulations")
    print("System: Lennard-Jones particles")
    print(f"Number of atoms: {params.num_atoms}")
    simulate(params, particles, steps=1)
    print("Done.")
