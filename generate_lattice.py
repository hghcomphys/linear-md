from ase.build import bulk

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate a lattice with given dimensions"
    )

    parser.add_argument(
        "--lattice",
        metavar=("NX", "NY", "NZ"),
        type=int,
        nargs=3,
        default=(10, 10, 10),
        help="Lattice dimensions nx ny nz (default: 10 10 10)"
    )

    args = parser.parse_args()
    nx, ny, nz = args.lattice

    print(f"Generating lattice: nx={nx}, ny={ny}, nz={nz}")
    atoms = bulk("Ar", "fcc", a=5.385, cubic=True)
    atoms = atoms.repeat((nx, ny, nz))
    print(f"Number of atoms:", len(atoms))

    cell = atoms.get_cell()
    Lx, Ly, Lz = cell.lengths()

    with open("argon.xyz", "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f'{Lx} 0 0 0 {Ly} 0 0 0 {Lz}\n')
        for atom in atoms:
            x, y, z = atom.position
            f.write(f"{atom.symbol} {x:.6f} {y:.6f} {z:.6f}\n")


if __name__ == "__main__":
    main()


