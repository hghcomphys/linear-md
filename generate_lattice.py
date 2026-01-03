from ase.build import bulk

atoms = bulk("Ar", "fcc", a=5.385, cubic=True)
atoms = atoms.repeat((10, 10, 10))
print(f"Number of atoms:", len(atoms))

cell = atoms.get_cell()
Lx, Ly, Lz = cell.lengths()

with open("argon.xyz", "w") as f:
    f.write(f"{len(atoms)}\n")
    f.write(f'{Lx} 0 0 0 {Ly} 0 0 0 {Lz}\n')
    for atom in atoms:
        x, y, z = atom.position
        f.write(f"{atom.symbol} {x:.6f} {y:.6f} {z:.6f}\n")

