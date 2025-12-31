import math
from typing import NamedTuple, TextIO

import numpy as np
from numba import njit
from numpy.typing import NDArray

FLOAT = np.float64
Array = NDArray[FLOAT]
ORDER = "C"

K_B = 8.617343e-5  # Boltzmann's constant in natural unit
TIME_UNIT_CONVERSION = 1.018051e1  # from natural unit to fs


class SimulationParameters(NamedTuple):
    num_atoms: int
    time_step: float
    temperature: float
    cutoff_radius: float
    skin_radius: float
    cell_max_atoms: int
    neighbor_max_atoms: int


class Particles(NamedTuple):
    mass: Array
    position: Array
    velocity: Array
    force: Array
    box: Array
    position_old: Array


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray


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


def initialize_neighbors(
    params: SimulationParameters,
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
    cell_atom_index = np.empty(
        shape=(math.prod(cell_sizes), params.cell_max_atoms),
        dtype=np.int32,
    )
    cell_num_atoms = np.empty(
        math.prod(cell_sizes),
        dtype=np.int32,
    )
    cells = CellList(
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
    neighbors = NeighborList(
        num_neighbors_per_atom=neighbor_num_atoms,
        neighbor_index=neighbor_index,
    )
    print(f"Cell length:", params.cutoff_radius)
    print(f"Cell sizes: {cells.sizes}")
    print(f"Cell max atoms:", cells.atom_index.shape[1])
    return (
        cells,
        neighbors,
    )


@njit
def index3d(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
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
    neighbor_cutoff2 = (params.cutoff_radius + params.skin_radius) ** 2
    box, position = particles.box, particles.position
    length = box[0], box[4], box[8]
    nc = cells.sizes

    # Cell list
    for ic0 in range(nc[0]):
        for ic1 in range(nc[1]):
            for ic2 in range(nc[2]):
                cells.num_atoms_per_cell[index3d((ic0, ic1, ic2), nc)] = 0

    for n in range(params.num_atoms):
        ic0 = math.floor(position[n, 0] * nc[0] / length[0])
        ic1 = math.floor(position[n, 1] * nc[1] / length[1])
        ic2 = math.floor(position[n, 2] * nc[2] / length[2])
        idx = index3d((ic0, ic1, ic2), nc)
        cells.atom_index[idx, cells.num_atoms_per_cell[idx]] = n
        cells.num_atoms_per_cell[idx] += 1

    # Neighbor list
    for n in range(params.num_atoms):
        neighbors.num_neighbors_per_atom[n] = 0

    for ic0 in range(nc[0]):
        for ic1 in range(nc[1]):
            for ic2 in range(nc[2]):
                idx_i = index3d((ic0, ic1, ic2), nc)

                for i in range(cells.num_atoms_per_cell[idx_i]):
                    ni = cells.atom_index[idx_i, i]

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

                                    if ni < nj:
                                        dx = apply_pbc(
                                            position[nj, 0] - position[ni, 0],
                                            length[0],
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

                                        if r2 < neighbor_cutoff2:
                                            neighbors.neighbor_index[
                                                ni,
                                                neighbors.num_neighbors_per_atom[ni],
                                            ] = nj
                                            neighbors.num_neighbors_per_atom[ni] += 1
                                            neighbors.neighbor_index[
                                                nj,
                                                neighbors.num_neighbors_per_atom[nj],
                                            ] = ni
                                            neighbors.num_neighbors_per_atom[nj] += 1

    # Update old position
    for n in range(params.num_atoms):
        particles.position_old[n, 0] = position[n, 0]
        particles.position_old[n, 1] = position[n, 1]
        particles.position_old[n, 2] = position[n, 2]


@njit
def check_if_neighbor_need_update(particles: Particles, params: SimulationParameters):
    displacement_cutoff2 = 0.25 * params.skin_radius * params.skin_radius
    r2 = FLOAT(0.0)
    for n in range(params.num_atoms):
        for dim in range(3):
            dx = particles.position[n, dim] - particles.position_old[n, dim]
            r2 += dx * dx
        if r2 > displacement_cutoff2:
            return True
    return False


@njit
def compute_force(
    particles: Particles,
    params: SimulationParameters,
    neighbors: NeighborList,
    potential_energy: Array,
) -> None:
    epsilon = 1.032e-2
    sigma = 3.405
    epsilon = 1.032e-2
    sigma = 3.405
    sigma3 = sigma * sigma * sigma
    sigma6 = sigma3 * sigma3
    sigma12 = sigma6 * sigma6
    e24s6 = 24.0 * epsilon * sigma6
    e48s12 = 48.0 * epsilon * sigma12
    e4s6 = 4.0 * epsilon * sigma6
    e4s12 = 4.0 * epsilon * sigma12

    box = particles.box
    force = particles.force
    position = particles.position
    cutoff2 = params.cutoff_radius * params.cutoff_radius
    length = box[0], box[4], box[8]

    potential_energy[0] = 0.0
    for ni in range(params.num_atoms):
        fi = particles.force[ni]
        fi[0] = fi[1] = fi[2] = 0.0

    for ni in range(params.num_atoms):
        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]
            # for nj in range(params.num_atoms):

            if ni < nj:
                dx = apply_pbc(position[nj, 0] - position[ni, 0], length[0])
                dy = apply_pbc(position[nj, 1] - position[ni, 1], length[1])
                dz = apply_pbc(position[nj, 2] - position[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue
                r2inv = 1.0 / r2
                r4inv = r2inv * r2inv
                r6inv = r2inv * r4inv
                r8inv = r4inv * r4inv
                r12inv = r4inv * r8inv
                r14inv = r6inv * r8inv
                fij = e24s6 * r8inv - e48s12 * r14inv
                force[ni, 0] += dx * fij
                force[ni, 1] += dy * fij
                force[ni, 2] += dz * fij
                force[nj, 0] -= dx * fij
                force[nj, 1] -= dy * fij
                force[nj, 2] -= dz * fij
                potential_energy[0] += e4s12 * r12inv - e4s6 * r6inv


@njit
def verlet_integration_position(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    dt = params.time_step
    m, r, v, f, box, _ = particles
    length = box[0], box[4], box[8]
    for n in range(params.num_atoms):
        for dim in range(3):
            r[n, dim] = np.fmod(
                r[n, dim]
                + dt * (v[n, dim] + 0.5 * dt / m[n] * f[n, dim])
                + 100.0 * length[dim],
                length[dim],
            )
            v[n, dim] += 0.5 * dt / m[n] * f[n, dim]


@njit
def verlet_integration_velocity(
    particles: Particles,
    params: SimulationParameters,
) -> None:
    dt = params.time_step
    m, v, f = particles.mass, particles.velocity, particles.force
    for n in range(params.num_atoms):
        for dim in range(3):
            v[n, dim] += 0.5 * dt / m[n] * f[n, dim]


def save(particles: Particles, file: TextIO) -> None:
    position = particles.position
    natoms = position.shape[0]
    file.write(f"{natoms}\n\n")
    for i in range(natoms):
        file.write(f"{'Ar'}\t{position[i, 0]} {position[i, 1]} {position[i, 2]}\n")
    file.flush()


@njit
def get_kinetic_energy(particles: Particles) -> float:
    natoms = len(particles.velocity)
    m, v = particles.mass, particles.velocity
    ke = 0.0
    for n in range(natoms):
        ke += m[n] * (v[n, 0] * v[n, 0] + v[n, 1] * v[n, 1] + v[n, 2] * v[n, 2])
    return 0.5 * ke


@njit
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
    update_grid(particles, params, cells, neighbors)
    # print(neighbors.num_neighbors_per_atom)
    # print(neighbors.neighbor_index)

    potential_energy = np.empty(1, dtype=FLOAT)
    compute_force(particles, params, neighbors, potential_energy)
    with open(filename, "w") as file:
        # Simulate
        print(
            f"{'Step':8}"
            f" {'Temp':10}"
            f" {'KinE':10}"
            f" {'PotE':10}"
            f" {'TotE':10}"
            f" {'Update':10}"
            f" {'AvgNb':10}"
            f" {'AvgCell':10}"
        )
        num_neighbor_udpates = 1
        for step in range(steps):
            if step % log_frequency == 0:
                ke = get_kinetic_energy(particles)
                pe = potential_energy[0]
                print(
                    f"{step:<8}"
                    # f" {step * params.time_step:<12.8f}"
                    f" {get_temperature(particles):<10.4f}"
                    f" {ke:<10.4f}"
                    f" {pe:<10.4f}"
                    f" {ke + pe:<10.4f}"
                    f" {num_neighbor_udpates:<10}"
                    f" {neighbors.num_neighbors_per_atom.mean():<10.2f}"
                    f" {cells.num_atoms_per_cell.mean():<10.2f}"
                )
                # save(particles.position, file)
            # Next time step (update r, v, and F)
            verlet_integration_position(particles, params)
            if check_if_neighbor_need_update(particles, params):
                update_grid(particles, params, cells, neighbors)
                num_neighbor_udpates += 1
            compute_force(particles, params, neighbors, potential_energy)
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
    params = SimulationParameters(
        num_atoms=len(atoms),
        time_step=0.5 / TIME_UNIT_CONVERSION,
        temperature=60.0,
        cutoff_radius=9.0,
        skin_radius=1.0,
        cell_max_atoms=100,
        neighbor_max_atoms=500,
    )
    mass = np.array([{"Ar": 40.0}[a[0]] for a in atoms])
    position = np.array([a[1:4] for a in atoms], dtype=FLOAT, order=ORDER)
    velocity = initialize_velocity(mass, params.temperature)
    force = np.empty((len(atoms), 3), dtype=FLOAT, order=ORDER)
    box = np.array(box, dtype=FLOAT).reshape(9)
    particles = Particles(
        mass=mass,
        position=position,
        velocity=velocity,
        force=force,
        box=box,
        position_old=position.copy(),
    )
    print(f"Number of atoms: {params.num_atoms}")
    print(f"Box matrix H:\n{particles.box.reshape(3, 3)}")

    # print("A Molecular Dynamics Simulations")
    # print("System: Lennard-Jones particles")
    simulate(params, particles, steps=2001)
    print("Done.")
