#!/usr/bin/env python
# coding: utf-8

import math
from typing import NamedTuple, TextIO

import numpy as np
from numba import njit
from numpy.typing import NDArray
from ase.io import read
from ase.build import bulk
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)


FLOAT = np.float64
Array = NDArray[FLOAT]
ORDER = "C"

K_B = 8.617343e-5  # eV/K 
FS_TO_NATURAL = 1.0 / 1.018051e1  


atoms_ase = read("argon.xyz")
with open("argon.xyz") as f:
    f.readline()
    box = np.fromstring(f.readline(), sep=" ").reshape(3, 3)
atoms_ase.set_cell(box)
atoms_ase.set_pbc(True)


ATOMIC_MASS = {"Ar": 40.0}  # amu
# atoms_ase = bulk("Ar", "fcc", a=5.385, cubic=True).repeat((8, 8, 8))
MaxwellBoltzmannDistribution(
    atoms_ase,
    temperature_K=60.0,
    rng=np.random.default_rng(1234),
)
Stationary(atoms_ase)


class Parameters(NamedTuple):
    num_atoms: int
    time_step: float
    temperature: float
    cutoff_radius: float
    skin_radius: float
    neighbor_max_atoms: int


params = Parameters(
    num_atoms=len(atoms_ase),
    time_step=0.5 * FS_TO_NATURAL,
    temperature=60.0, # Kelvin
    cutoff_radius=9.0,  # Angstrom
    skin_radius=1.0,  # Angstrom
    neighbor_max_atoms=500,
)
params


class Particles(NamedTuple):
    mass: Array
    position: Array
    velocity: Array
    force: Array
    box: Array
    position_old: Array


mass = atoms_ase.get_masses()
position = np.asarray(atoms_ase.get_positions(), dtype=FLOAT, order=ORDER)
velocity = np.asarray(atoms_ase.get_velocities(), dtype=FLOAT, order=ORDER)
force = np.empty((len(atoms_ase), 3), dtype=FLOAT, order=ORDER)
box =  np.asarray(atoms_ase.cell.array, dtype=FLOAT)
particles = Particles(
    mass=mass,
    position=position,
    velocity=velocity,
    force=force,
    box=box,
    position_old=position.copy(),
)
# particles


class CellList(NamedTuple):
    num_atoms_per_cell: NDArray
    atom_index: NDArray
    sizes: tuple[int, int, int]


def initialize_cell_list(
    params: Parameters,
    box: Array,
):
    cell_length = params.cutoff_radius
    length = box[0][0], box[1][1], box[2][2]
    cell_sizes = (
        int(length[0] / cell_length),
        int(length[1] / cell_length),
        int(length[2] / cell_length),
    )
    for size in cell_sizes:
        assert size > 2, print(f"{cell_sizes=}")

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

    return CellList(
        sizes=cell_sizes,
        num_atoms_per_cell=cell_num_atoms,
        atom_index=cell_atom_index,
    )


class NeighborList(NamedTuple):
    num_neighbors_per_atom: NDArray
    neighbor_index: NDArray


def initialize_neighbor_list(
    params: Parameters,
):
    neighbor_index = np.empty(
        shape=(params.num_atoms, params.neighbor_max_atoms),
        dtype=np.int32,
    )
    neighbor_num_atoms = np.empty(
        params.num_atoms,
        dtype=np.int32,
    )

    return NeighborList(
        num_neighbors_per_atom=neighbor_num_atoms,
        neighbor_index=neighbor_index,
    )


def initialize_neighbors(
    params: Parameters,
    box: Array,
) -> tuple[CellList, NeighborList]:
    cells = initialize_cell_list(params, box)
    neighbors = initialize_neighbor_list(params)

    print(f"Cell length:", params.cutoff_radius)
    print(f"Cell sizes: {cells.sizes}")

    return cells, neighbors


cells, neighbors = initialize_neighbors(params, particles.box)
# cells, neighbors


@njit(inline="always")
def index3d(ic: tuple[int, int, int], nc: tuple[int, int, int]) -> int:
    return ic[0] + nc[0] * (ic[1] + nc[1] * ic[2])


@njit(inline="always")
def apply_pbc(
    rij: FLOAT,
    length: FLOAT,
) -> FLOAT:
    if rij >= FLOAT(0.5) * length:
        rij -= length
    elif rij <= FLOAT(-0.5) * length:
        rij += length
    return rij


@njit
def update_neighbors(
    particles: Particles,
    params: Parameters,
    cells: CellList,
    neighbors: NeighborList,
) -> None:
    neighbor_cutoff2 = (params.cutoff_radius + params.skin_radius) ** 2
    box, position = particles.box, particles.position
    length = box[0][0], box[1][1], box[2][2]
    nc = cells.sizes

    # --- cell list ---
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

    # --- neighbor list ---
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
                                        dx = apply_pbc(position[nj, 0] - position[ni, 0], length[0])
                                        dy = apply_pbc(position[nj, 1] - position[ni, 1], length[1])
                                        dz = apply_pbc(position[nj, 2] - position[ni, 2], length[2])
                                        r2 = dx * dx + dy * dy + dz * dz

                                        if r2 < neighbor_cutoff2:
                                            neighbors.neighbor_index[ni, neighbors.num_neighbors_per_atom[ni]] = nj
                                            neighbors.num_neighbors_per_atom[ni] += 1
                                            neighbors.neighbor_index[nj,neighbors.num_neighbors_per_atom[nj]] = ni
                                            neighbors.num_neighbors_per_atom[nj] += 1
    # --- update old position ---
    for n in range(params.num_atoms):
        particles.position_old[n, 0] = position[n, 0]
        particles.position_old[n, 1] = position[n, 1]
        particles.position_old[n, 2] = position[n, 2]


update_neighbors(particles, params, cells, neighbors)
# cells, neighbors


@njit
def check_if_neighbor_need_update(particles: Particles, params: Parameters):
    displacement_cutoff2 = 0.25 * params.skin_radius * params.skin_radius
    r2 = FLOAT(0.0)
    for n in range(params.num_atoms):
        for dim in range(3):
            dx = particles.position[n, dim] - particles.position_old[n, dim]
            r2 += dx * dx
        if r2 > displacement_cutoff2:
            return True
    return False


check_if_neighbor_need_update(particles, params)


@njit
def compute_force(
    particles: Particles,
    params: Parameters,
    neighbors: NeighborList,
    potential_energy: Array,
) -> None:

    epsilon = FLOAT(1.032e-2)
    sigma = FLOAT(3.405)
    sigma3 = sigma * sigma * sigma
    sigma6 = sigma3 * sigma3
    sigma12 = sigma6 * sigma6
    e24s6 = FLOAT(24.0) * epsilon * sigma6
    e48s12 = FLOAT(48.0) * epsilon * sigma12
    e4s6 = FLOAT(4.0) * epsilon * sigma6
    e4s12 = FLOAT(4.0) * epsilon * sigma12

    box = particles.box
    force = particles.force
    position = particles.position
    cutoff2 = params.cutoff_radius * params.cutoff_radius
    length = box[0][0], box[1][1], box[2][2]

    potential_energy[0] = FLOAT(0.0)
    for ni in range(params.num_atoms):
        fi = particles.force[ni]
        fi[0] = fi[1] = fi[2] = FLOAT(0.0)

    for ni in range(params.num_atoms):
        for nj in range(params.num_atoms):
        # for j in range(neighbors.num_neighbors_per_atom[ni]):
        #     nj = neighbors.neighbor_index[ni, j]

            if ni < nj:
                dx = apply_pbc(position[nj, 0] - position[ni, 0], length[0])
                dy = apply_pbc(position[nj, 1] - position[ni, 1], length[1])
                dz = apply_pbc(position[nj, 2] - position[ni, 2], length[2])
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > cutoff2:
                    continue
                r2inv = FLOAT(1.0) / r2
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


potential_energy = np.empty(1, dtype=FLOAT)
compute_force(particles, params, neighbors, potential_energy)


@njit
def verlet_integration_position(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    m, r, v, f, box, _ = particles
    length = box[0][0], box[1][1], box[2][2]
    for n in range(params.num_atoms):
        for dim in range(3):
            r[n, dim] = np.fmod(
                r[n, dim]
                + dt * (v[n, dim] + FLOAT(0.5) * dt / m[n] * f[n, dim])
                + FLOAT(100.0) * length[dim],
                length[dim],
            )
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]


@njit
def verlet_integration_velocity(
    particles: Particles,
    params: Parameters,
) -> None:
    dt = params.time_step
    m, v, f = particles.mass, particles.velocity, particles.force
    for n in range(params.num_atoms):
        for dim in range(3):
            v[n, dim] += FLOAT(0.5) * dt / m[n] * f[n, dim]


def simulate(
    params: Parameters,
    particles: Particles,
    steps: int = 1,
    log_frequency: int = 100,
    filename: str = "out.xyz",
) -> None:
    box = particles.box
    cells, neighbors = initialize_neighbors(params, box)
    update_neighbors(particles, params, cells, neighbors)

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
            f" {'UpdateNb':10}"
            f" {'AvgNb':10}"
            f" {'AvgCell':10}"
        )
        num_neighbor_updates = 1
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
                    f" {num_neighbor_updates:<10}"
                    f" {neighbors.num_neighbors_per_atom.mean():<10.2f}"
                    f" {cells.num_atoms_per_cell.mean():<10.2f}"
                )
                # save(particles.position, file)
            # Next time step (update r, v, and F)
            verlet_integration_position(particles, params)
            if check_if_neighbor_need_update(particles, params):
                update_neighbors(particles, params, cells, neighbors)
                num_neighbor_updates += 1
            compute_force(particles, params, neighbors, potential_energy)
            verlet_integration_velocity(particles, params)


@njit
def get_kinetic_energy(particles: Particles) -> FLOAT:
    natoms = len(particles.velocity)
    m, v = particles.mass, particles.velocity
    ke = FLOAT(0.0)
    for n in range(natoms):
        ke += m[n] * (v[n, 0] * v[n, 0] + v[n, 1] * v[n, 1] + v[n, 2] * v[n, 2])
    return FLOAT(0.5) * ke


@njit
def get_temperature(particles: Particles) -> FLOAT:
    natoms = len(particles.velocity)
    return FLOAT(2.0) / (FLOAT(3) * natoms * K_B) * get_kinetic_energy(particles)


def save(particles: Particles, file: TextIO) -> None:
    position = particles.position
    natoms = position.shape[0]
    file.write(f"{natoms}\n\n")
    for i in range(natoms):
        file.write(f"{'Ar'}\t{position[i, 0]} {position[i, 1]} {position[i, 2]}\n")
    file.flush()


print(f"Number of atoms: {params.num_atoms}")
simulate(params, particles, steps=1001)
print("Done.")

