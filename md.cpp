#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#define INDEX(ic, nc) ((ic)[0] + (nc)[0] * ((ic)[1] + (nc)[1] * (ic)[2]))
#define ITERATE_OVER_CELLS(ic, nc)                      \
    for ((ic)[0] = 0; (ic)[0] < (nc)[0]; (ic)[0]++)     \
        for ((ic)[1] = 0; (ic)[1] < (nc)[1]; (ic)[1]++) \
            for ((ic)[2] = 0; (ic)[2] < (nc)[2]; (ic)[2]++)
#define ITERATE_OVER_ATOMS(n, vmax) \
    for ((n) = 0; (n) < (vmax); ++(n))
#define ITERATE_OVER_DIMS(d) \
    for ((d) = 0; (d) < 3; ++(d))

#define K_B 8.617343e-5                  // Boltzmann's constant in natural unit
#define TIME_UNIT_CONVERSION 1.018051e+1 // from natural unit to fs
#define CELL_MAX_ATOMS 500

const double r_cut = 9.0;
const int cell_update_frequency = 10;
const int output_frequency = 100;

struct MDParameters
{
    int numberOfSteps;
    double timeStep;
    double temperature;
};

struct Atom
{
    double mass;
    double position[3];
    double velocity[3];
    double force[3];
};

struct Cell
{
    int numberOfAtomsPerCell;
    std::vector<int> atomIndex;
};

struct MDSystem
{
    int numberOfAtoms;
    std::vector<Atom> atoms;

    int cellSizes[3];
    int numberOfCellUpdates;
    std::vector<Cell> grid;

    double box[9];
    double potentialEnergy;
};

std::vector<std::string> getTokens(std::ifstream &input)
{
    std::string line, token;
    std::getline(input, line);
    std::istringstream iss(line);
    std::vector<std::string> tokens;

    while (iss >> token)
        tokens.push_back(token);

    return tokens;
}

double getDouble(std::string &token)
{
    float value = 0;
    try
    {
        value = std::stod(token);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception:" << e.what() << std::endl;
        exit(1);
    }
    return value;
}

int getInt(std::string &token)
{
    int value = 0;
    try
    {
        value = std::stoi(token);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard exception:" << e.what() << std::endl;
        exit(1);
    }
    return value;
}

void readXyz(MDSystem &sys, const std::string &filename)
{
    int n, d, d1, d2;

    std::ifstream input(filename);
    if (!input.is_open())
    {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(1);
    }

    // line 1
    std::vector<std::string> tokens = getTokens(input);
    if (tokens.size() != 1)
    {
        std::cerr << "The first line of xyz.in should have one item." << std::endl;
        exit(1);
    }
    sys.numberOfAtoms = getInt(tokens[0]);
    std::cout << "Number of atoms = " << sys.numberOfAtoms << std::endl;

    // allocate memory
    sys.atoms.resize(sys.numberOfAtoms);

    // line 2
    tokens = getTokens(input);
    if (tokens.size() != 9)
    {
        std::cerr << "The second line of " << filename << " should have 9 items." << std::endl;
        exit(1);
    }

    std::cout << "Box matrix H = " << std::endl;
    ITERATE_OVER_DIMS(d1)
    {
        ITERATE_OVER_DIMS(d2)
        {
            sys.box[d1 * 3 + d2] = getDouble(tokens[d1 * 3 + d2]);
            std::cout << sys.box[d1 * 3 + d2] << " ";
        }
        std::cout << std::endl;
    }

    // starting from line 3
    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        tokens = getTokens(input);
        if (tokens.size() < 5)
        {
            std::cerr << "The 3rd line and later of xyz.in "
                         "should have 5 items."
                      << std::endl;
            exit(1);
        }
        ITERATE_OVER_DIMS(d)
        {
            sys.atoms[n].position[d] = getDouble(tokens[d + 1]);
        }
        sys.atoms[n].mass = getDouble(tokens[4]);
    }

    input.close();
}

void readRun(MDParameters &params, const std::string &filename)
{
    std::ifstream input(filename);
    if (!input.is_open())
    {
        std::cerr << "Failed to open run.in." << std::endl;
        exit(1);
    }

    while (input.peek() != EOF)
    {
        std::vector<std::string> tokens = getTokens(input);
        if (tokens.size() > 0)
        {
            if (tokens[0] == "time_step")
            {
                params.timeStep = getDouble(tokens[1]);
                if (params.timeStep < 0)
                {
                    std::cout << "timeStep should >= 0." << std::endl;
                    exit(1);
                }
                std::cout << "timeStep = " << params.timeStep << " fs." << std::endl;
                params.timeStep /= TIME_UNIT_CONVERSION; // from fs to natural unit
            }
            else if (tokens[0] == "run")
            {
                params.numberOfSteps = getInt(tokens[1]);
                if (params.numberOfSteps < 1)
                {
                    std::cout << "numberOfSteps should >= 1." << std::endl;
                    exit(1);
                }
                std::cout << "numberOfSteps = " << params.numberOfSteps << std::endl;
            }
            else if (tokens[0] == "velocity")
            {
                params.temperature = getDouble(tokens[1]);
                if (params.temperature < 0)
                {
                    std::cout << "temperature >= 0." << std::endl;
                    exit(1);
                }
                std::cout << "temperature = " << params.temperature << " K." << std::endl;
            }
        }
        else if (tokens[0][0] != '#')
        {
            std::cout << tokens[0] << " is not a valid keyword." << std::endl;
            exit(1);
        }
    }
}

double ComputeKineticEnergy(const MDSystem &sys)
{
    int n, d;
    double kineticEnergy = 0.0;

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        double v2 = 0.0;
        ITERATE_OVER_DIMS(d)
        {
            v2 += sys.atoms[n].velocity[d] * sys.atoms[n].velocity[d];
        }
        kineticEnergy += sys.atoms[n].mass * v2;
    }
    return 0.5 * kineticEnergy;
}

void scaleVelocity(MDSystem &sys, const double T0)
{
    int n, d;
    const double temperature =
        ComputeKineticEnergy(sys) * 2.0 / (3.0 * K_B * sys.numberOfAtoms);
    double scaleFactor = sqrt(T0 / temperature);
    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIMS(d)
        {
            sys.atoms[n].velocity[d] *= scaleFactor;
        }
    }
}

void initializeVelocity(MDSystem &sys, const double T0)
{
#ifndef DEBUG
    srand(42);
#endif

    int n, d;
    double centerOfMassVelocity[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        totalMass += sys.atoms[n].mass;
        ITERATE_OVER_DIMS(d)
        {
            sys.atoms[n].velocity[d] = -1.0 + (rand() * 2.0) / RAND_MAX;
            centerOfMassVelocity[d] += sys.atoms[n].mass * sys.atoms[n].velocity[d];
        }
    }
    ITERATE_OVER_DIMS(d)
    {
        centerOfMassVelocity[d] /= totalMass;
    }

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIMS(d)
        {
            sys.atoms[n].velocity[d] -= centerOfMassVelocity[d];
        }
    }
    scaleVelocity(sys, T0);
}

void initializeCells(MDSystem &sys)
{
    int d;
    double l;

    ITERATE_OVER_DIMS(d)
    {
        l = sys.box[d * 3 + d];
        sys.cellSizes[d] = (int)(l / r_cut);
        if (sys.cellSizes[d] < 3)
        {
            std::cerr << "nc[" << d << "]=" << sys.cellSizes[d] << " under PBC, this must be greater than 2!" << std::endl;
            exit(1);
        }
    }

    int numberOfCells = 1;
    ITERATE_OVER_DIMS(d)
    {
        numberOfCells *= sys.cellSizes[d];
    }
    sys.grid.resize(numberOfCells);

    for (Cell &cell : sys.grid)
        cell.atomIndex.resize(CELL_MAX_ATOMS);

    sys.numberOfCellUpdates = 0;
}

void updateCells(MDSystem &sys)
{
    double l;
    int n, d, ic[3];

    ITERATE_OVER_CELLS(ic, sys.cellSizes)
    {
        sys.grid[INDEX(ic, sys.cellSizes)].numberOfAtomsPerCell = 0;
    }

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIMS(d)
        {
            l = sys.box[d * 3 + d];
            ic[d] = (int)(sys.atoms[n].position[d] * sys.cellSizes[d] / l);
        }

        Cell &cell = sys.grid[INDEX(ic, sys.cellSizes)];
        if (cell.numberOfAtomsPerCell > CELL_MAX_ATOMS)
        {
            std::cerr << "Max number of atoms per cell exceeded: " << CELL_MAX_ATOMS << std::endl;
            exit(1);
        }
        cell.atomIndex[cell.numberOfAtomsPerCell++] = n;
    }
    sys.numberOfCellUpdates += 1;
}

void computeForce(MDSystem &sys)
{
    int n, d, ni, nj;
    int ic[3], kc[3], jc[3];
    double l[3], r[3], r2;

    const double epsilon = 1.032e-2;
    const double sigma = 3.405;
    const double cutoffSquare = r_cut * r_cut;
    const double sigma3 = sigma * sigma * sigma;
    const double sigma6 = sigma3 * sigma3;
    const double sigma12 = sigma6 * sigma6;
    const double e24s6 = 24.0 * epsilon * sigma6;
    const double e48s12 = 48.0 * epsilon * sigma12;
    const double e4s6 = 4.0 * epsilon * sigma6;
    const double e4s12 = 4.0 * epsilon * sigma12;

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIMS(d)
        {
            sys.atoms[n].force[d] = 0.0;
        }
    }
    sys.potentialEnergy = 0.0;

    ITERATE_OVER_DIMS(d)
    {
        l[d] = sys.box[d * 3 + d];
    }

    ITERATE_OVER_CELLS(ic, sys.cellSizes)
    {
        // Iterate over atoms in cell ic
        Cell &cell_ic = sys.grid[INDEX(ic, sys.cellSizes)];
        for (int i = 0; i < cell_ic.numberOfAtomsPerCell; ++i)
        {
            ni = cell_ic.atomIndex[i];

            // Iterate over neighbor cells of cell ic
            for (kc[0] = ic[0] - 1; kc[0] <= ic[0] + 1; kc[0]++)
            {
                if ((kc[0] == -1) || (kc[0] == sys.cellSizes[0]))
                    jc[0] = (kc[0] + sys.cellSizes[0]) % sys.cellSizes[0];
                else
                    jc[0] = kc[0];

                for (kc[1] = ic[1] - 1; kc[1] <= ic[1] + 1; kc[1]++)
                {
                    if ((kc[1] == -1) || (kc[1] == sys.cellSizes[1]))
                        jc[1] = (kc[1] + sys.cellSizes[1]) % sys.cellSizes[1];
                    else
                        jc[1] = kc[1];

                    for (kc[2] = ic[2] - 1; kc[2] <= ic[2] + 1; kc[2]++)
                    {
                        if ((kc[2] == -1) || (kc[2] == sys.cellSizes[2]))
                            jc[2] = (kc[2] + sys.cellSizes[2]) % sys.cellSizes[2];
                        else
                            jc[2] = kc[2];

                        // Iterate over all atoms in cell jc
                        Cell &cell_jc = sys.grid[INDEX(jc, sys.cellSizes)];
                        for (int j = 0; j < cell_jc.numberOfAtomsPerCell; ++j)
                        {
                            nj = cell_jc.atomIndex[j];

                            if (ni < nj)
                            {
                                r2 = 0;
                                ITERATE_OVER_DIMS(d)
                                {
                                    r[d] = sys.atoms[nj].position[d] - sys.atoms[ni].position[d];
                                    // Apply PBC
                                    if (r[d] > l[d] * 0.5)
                                        r[d] -= l[d];
                                    else if (r[d] < -l[d] * 0.5)
                                        r[d] += l[d];

                                    r2 += r[d] * r[d];
                                }
                                if (r2 <= cutoffSquare)
                                {
                                    const double r2inv = 1.0 / r2;
                                    const double r4inv = r2inv * r2inv;
                                    const double r6inv = r2inv * r4inv;
                                    const double r8inv = r4inv * r4inv;
                                    const double r12inv = r4inv * r8inv;
                                    const double r14inv = r6inv * r8inv;
                                    const double fij = e24s6 * r8inv - e48s12 * r14inv;

                                    ITERATE_OVER_DIMS(d)
                                    {
                                        sys.atoms[ni].force[d] += fij * r[d];
                                        sys.atoms[nj].force[d] -= fij * r[d];
                                    }
                                    sys.potentialEnergy += e4s12 * r12inv - e4s6 * r6inv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void integrateVelocityVerlet1(MDSystem &sys, const MDParameters &params)
{
    int n, d;
    double x;
    double l[3];

    ITERATE_OVER_DIMS(d)
    {
        l[d] = sys.box[d * 3 + d];
    }

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIMS(d)
        {
            x = sys.atoms[n].position[d] + params.timeStep * (sys.atoms[n].velocity[d] + params.timeStep * 0.5 / sys.atoms[n].mass * sys.atoms[n].force[d]);
            if ((x < 0.) || (x >= l[d]))
                x = fmod(x + 100. * l[d], l[d]);

            sys.atoms[n].position[d] = x;
            sys.atoms[n].velocity[d] += params.timeStep * 0.5 / sys.atoms[n].mass * sys.atoms[n].force[d];
        }
    }
}

void integrateVelocityVerlet2(MDSystem &sys, const MDParameters &params)
{
    int n, d;

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIMS(d)
        {
            sys.atoms[n].velocity[d] += params.timeStep * 0.5 / sys.atoms[n].mass * sys.atoms[n].force[d];
        }
    }
}

void saveXyz(const MDSystem &sys)
{
    std::ofstream file("out.xyz", std::ios::app);
    if (!file)
        return;

    file << sys.numberOfAtoms << "\n";
    file << "XYZ configuration\n";
    for (const Atom &atom : sys.atoms)
    {
        file << "Ar" << " "
             << atom.position[0] << " "
             << atom.position[1] << " "
             << atom.position[2] << " "
             << "\n";
    }
    file.close();
}

// int main(int argc, char **argv)
int main()
{
    MDSystem sys;
    MDParameters params;

    readRun(params, "run.in");
    readXyz(sys, "Ar.xyz");
    initializeVelocity(sys, params.temperature);
    initializeCells(sys);
    updateCells(sys);
    computeForce(sys);

    std::cout
        << "Step Temperature KineticEnergy  PotentialEnergy TotalEnergy CellUpdates" << std::endl;
    for (int step = 0; step < params.numberOfSteps; ++step)
    {
        integrateVelocityVerlet1(sys, params);
        if (step % cell_update_frequency == 0)
            updateCells(sys);
        computeForce(sys);
        integrateVelocityVerlet2(sys, params);

        if (step % output_frequency == 0)
        {
            const double pe = sys.potentialEnergy;
            const double ke = ComputeKineticEnergy(sys);
            const double T = ke / (3.0 / 2.0 * K_B * sys.numberOfAtoms);
            std::cout << step << " " << T << " " << ke << " " << pe << " " << ke + pe << " " << sys.numberOfCellUpdates << std::endl;
            // saveXyz(sys);
        }
    }

    return 0;
}
