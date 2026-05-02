import os

# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import math
from typing import NamedTuple

import numpy as np
from numba import jit, njit
from numba import cuda
from numpy.typing import NDArray

from numba.cuda.cudadrv.devicearray import DeviceNDArray

# DeviceNDArray = NDArray

FLOAT = np.float64
ORDER = "C"
Array = NDArray[FLOAT]
DeviceArray = DeviceNDArray

K_B = 8.617343e-5  # Boltzmann's constant in natural unit
TIME_UNIT_CONVERSION = 1.018051e1  # from natural unit to fs


class Parameters(NamedTuple):
    num_atoms: int
    time_step: float
    temperature: float
    cutoff_radius: float
    skin_radius: float
    neighbor_max_atoms: int


class Particles(NamedTuple):
    mass: DeviceArray
    position: DeviceArray
    velocity: DeviceArray
    force: DeviceArray
    box: DeviceArray
    position_old: DeviceArray


class NeighborUpdateInputs(NamedTuple):
    position: Array
    box: Array


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray


def main() -> None:

    ATOMIC_MASS = {"Ar": 40.0}
    atoms, box = read_xyz("argon.xyz")
    params = Parameters(
        num_atoms=len(atoms),
        time_step=0.5 / TIME_UNIT_CONVERSION,
        temperature=60.0,
        cutoff_radius=9.0,
        skin_radius=1.0,
        neighbor_max_atoms=500,
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
    simulate(params, particles, steps=1001)

    print("Done.")


def simulate(
    params: Parameters,
    particles: Particles,
    steps: int = 1,
    log_frequency: int = 100,
    filename: str = "out.xyz",
) -> None:

    box = particles.box
    cells, neighbors = initialize_neighbors(params, box)
    neighbor_update_inputs_host = NeighborUpdateInputs(
        position=particles.position.copy_to_host(),
        box=particles.box.copy_to_host(),
        # position_old=particles.position_old.copy_to_host(),
    )
    update_neighbors(
        neighbor_update_inputs_host,
        params,
        cells,
        neighbors,
    )

    threads = 32
    blocks = math.ceil(params.num_atoms / threads)
    print(f"CUDA grid: {blocks=}, {threads}")

    potential_energy = cuda.device_array(1, dtype=FLOAT)
    compute_force_kernel[blocks, threads](
        particles, params, neighbors, potential_energy
    )

    with open(filename, "w") as file:
        # Simulate
        print(
            f"{'Step':8}"
            f" {'Temp':10}"
            f" {'KinE':10}"
            # f" {'PotE':10}"
            # f" {'TotE':10}"
            f" {'UpdateNb':10}"
            f" {'AvgNb':10}"
            f" {'AvgCell':10}"
        )
        num_neighbor_updates = 1
        check_neighbor_result_host = np.zeros(1, dtype=np.float32)
        check_neighbor_result = cuda.to_device(check_neighbor_result_host)

        for step in range(steps):
            if step % log_frequency == 0:
                # pe = potential_energy.copy_to_host()[0]
                velocity = particles.velocity.copy_to_host()
                mass = particles.mass.copy_to_host()
                ke = get_kinetic_energy(velocity, mass)
                print(
                    f"{step:<8}"
                    # f" {step * params.time_step:<12.8f}"
                    f" {get_temperature(velocity, mass):<10.4f}"
                    f" {ke:<10.4f}"
                    # f" {pe:<10.4f}"
                    # f" {ke + pe:<10.4f}"
                    f" {num_neighbor_updates:<10}"
                    f" {neighbors.num_neighbors_per_atom.mean():<10.0f}"
                    f" {cells.num_atoms_per_cell.mean():<10.0f}"
                )
                # save(particles.position, file)

            # Next time step (update r, v, and F)
            verlet_integration_position_kernel[blocks, threads](particles, params)

            check_if_neighbor_need_update_kernel[blocks, threads](
                particles, params, check_neighbor_result
            )
            check_neighbor_result.copy_to_host(check_neighbor_result_host)
            if check_neighbor_result_host[0] > 0:
                particles.position.copy_to_host(neighbor_update_inputs_host.position)
                particles.box.copy_to_host(neighbor_update_inputs_host.box)
                update_neighbors(neighbor_update_inputs_host, params, cells, neighbors)
                check_neighbor_result_host[0] = 0
                cuda.to_device(check_neighbor_result_host, to=check_neighbor_result)
                particles.position_old[:] = particles.position  # copying on the device
                num_neighbor_updates += 1

            compute_force_kernel[blocks, threads](
                particles, params, neighbors, potential_energy
            )
            verlet_integration_velocity_kernel[blocks, threads](particles, params)



@cuda.jit
def check_if_neighbor_need_update_kernel(
    particles: Particles,
    params: Parameters,
    result: DeviceArray,
):
    displacement_cutoff2 = 0.25 * params.skin_radius * params.skin_radius
    r2 = FLOAT(0.0)
    n = cuda.grid(1)
    if n < params.num_atoms:
        for dim in range(3):
            dx = particles.position[n, dim] - particles.position_old[n, dim]
            r2 += dx * dx
        if r2 > displacement_cutoff2:
            cuda.atomic.max(result, 0, 1)


@cuda.jit
def compute_force_kernel(
    particles: Particles,
    params: Parameters,
    neighbors: NeighborList,
    potential_energy: Array,
) -> None:
    ni = cuda.grid(1)

    if ni < params.num_atoms:
        epsilon = FLOAT(1.032e-2)
        sigma = FLOAT(3.405)
        epsilon = FLOAT(1.032e-2)
        sigma = FLOAT(3.405)
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
        fi = particles.force[ni]
        fi[0] = fi[1] = fi[2] = 0.0

        for j in range(neighbors.num_neighbors_per_atom[ni]):
            nj = neighbors.neighbor_index[ni, j]

            # if ni < nj:
            dx = apply_pbc_device(position[nj, 0] - position[ni, 0], length[0])
            dy = apply_pbc_device(position[nj, 1] - position[ni, 1], length[1])
            dz = apply_pbc_device(position[nj, 2] - position[ni, 2], length[2])
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
            # pe = 0.5 * (e4s12 * r12inv - e4s6 * r6inv)
            # cuda.atomic.add(potential_energy, 0, pe)


@cuda.jit
def verlet_integration_position_kernel(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    m, r, v, f, box, _ = particles
    length = box[0], box[4], box[8]
    n = cuda.grid(1)

    if n < params.num_atoms:
        for dim in range(3):
            r[n, dim] = math.fmod(
                r[n, dim]
                + dt * (v[n, dim] + 0.5 * dt / m[n] * f[n, dim])
                + 100.0 * length[dim],
                length[dim],
            )
            v[n, dim] += 0.5 * dt / m[n] * f[n, dim]


@cuda.jit
def verlet_integration_velocity_kernel(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    m, v, f = particles.mass, particles.velocity, particles.force
    n = cuda.grid(1)

    if n < params.num_atoms:
        for dim in range(3):
            v[n, dim] += 0.5 * dt / m[n] * f[n, dim]


def initialize_neighbors(
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

    return (
        cells,
        neighbors,
    )


@njit
def update_neighbors(
    particles: NeighborUpdateInputs,
    params: Parameters,
    cells: CellList,
    neighbors: NeighborList,
) -> None:

    neighbor_cutoff2 = (params.cutoff_radius + params.skin_radius) ** 2
    position, box = particles.position, particles.box
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


@jit(nopython=True, inline="always")
def index3d(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@njit
def get_kinetic_energy(velocity: Array, mass: Array) -> FLOAT:
    m, v = mass, velocity
    natoms = len(velocity)
    ke = FLOAT(0.0)
    for n in range(natoms):
        ke += m[n] * (v[n, 0] * v[n, 0] + v[n, 1] * v[n, 1] + v[n, 2] * v[n, 2])
    return FLOAT(0.5 * ke)


@njit
def get_temperature(velocity: Array, mass: Array) -> FLOAT:
    natoms = len(velocity)
    return FLOAT(2.0 / (3 * natoms * K_B) * get_kinetic_energy(velocity, mass))


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


@cuda.jit(device=True, inline=True)
def apply_pbc_device(
    rij: FLOAT,
    length: FLOAT,
) -> FLOAT:
    if rij >= 0.5 * length:
        rij -= length
    elif rij <= -0.5 * length:
        rij += length
    return rij


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
