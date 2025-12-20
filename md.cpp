
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#define SQR(x) ((x) * (x))
#define INDEX(ic, nc) ((ic)[0] + (nc)[0] * ((ic)[1] + (nc)[1] * (ic)[2]))
#define ITERATE_OVER_CELLS(iv, mv)                      \
    for ((iv)[0] = 0; (iv)[0] < (mv)[0]; (iv)[0]++)     \
        for ((iv)[1] = 0; (iv)[1] < (mv)[1]; (iv)[1]++) \
            for ((iv)[2] = 0; (iv)[2] < (mv)[2]; (iv)[2]++)
#define ITERATE_OVER_ATOMS(iv, mv) \
    for ((iv) = 0; (iv) < (mv); ++(iv))
#define ITERATE_OVER_DIM(iv) \
    for ((iv) = 0; (iv) < 3; ++(iv))

const double r_cut = 2.5;
const double K_B = 8.617343e-5;                  // Boltzmann's constant in natural unit
const double TIME_UNIT_CONVERSION = 1.018051e+1; // from natural unit to fs
const int OUTPUT_FREQUENCY = 100;
const int MAX_ATOMS_PER_CELL = 500;

struct SimulationParameters
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
    int count;
    std::vector<int> atomIndex;
};

struct System
{
    int numberOfAtoms;
    std::vector<Atom> atoms;

    int cellSizes[3];
    int numberOfCellUpdates;
    std::vector<Cell> grid;

    double box[9];
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

void readXyz(System &sys, const std::string &filename)
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
    ITERATE_OVER_DIM(d1)
    {
        ITERATE_OVER_DIM(d2)
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
        ITERATE_OVER_DIM(d)
        {
            sys.atoms[n].position[d] = getDouble(tokens[d + 1]);
        }
        sys.atoms[n].mass = getDouble(tokens[4]);
    }

    input.close();
}

void readRun(SimulationParameters &params, const std::string &filename)
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

double ComputeKineticEnergy(const System &sys)
{
    int n, d;
    double kineticEnergy = 0.0;

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        double v2 = 0.0;
        ITERATE_OVER_DIM(d)
        {
            v2 += sys.atoms[n].position[d] * sys.atoms[n].position[d];
        }
        kineticEnergy += sys.atoms[n].mass * v2;
    }
    return 0.5 * kineticEnergy;
}

void scaleVelocity(System &sys, const double T0)
{
    int n, d;
    const double temperature =
        ComputeKineticEnergy(sys) * 2.0 / (3.0 * K_B * sys.numberOfAtoms);
    double scaleFactor = sqrt(T0 / temperature);
    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIM(d)
        {
            sys.atoms[n].velocity[d] *= scaleFactor;
        }
    }
}

void initializeVelocity(System &sys, const double T0)
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
        ITERATE_OVER_DIM(d)
        {
            sys.atoms[n].velocity[d] = -1.0 + (rand() * 2.0) / RAND_MAX;
            centerOfMassVelocity[d] += sys.atoms[n].mass * sys.atoms[n].velocity[d];
        }
    }
    ITERATE_OVER_DIM(d)
    {
        centerOfMassVelocity[d] /= totalMass;
    }

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIM(d)
        {
            sys.atoms[n].velocity[d] -= centerOfMassVelocity[d];
        }
    }
    scaleVelocity(sys, T0);
}

// bool checkIfNeedUpdate(const System &sys)
// {
//     bool needUpdate = false;
//     for (int n = 0; n < sys.numberOfAtoms; ++n)
//     {
//         double dx = sys.x[n] - sys.x0[n];
//         double dy = sys.y[n] - sys.y0[n];
//         double dz = sys.z[n] - sys.z0[n];
//         if (dx * dx + dy * dy + dz * dz > 0.25)
//         {
//             needUpdate = true;
//             break;
//         }
//     }
//     return needUpdate;
// }

// void applyPbcOne(double &sx)
// {
//     if (sx < 0.0)
//     {
//         sx += 1.0;
//     }
//     else if (sx > 1.0)
//     {
//         sx -= 1.0;
//     }
// }

// void findNeighbor(System &sys)
// {
//     if (checkIfNeedUpdate(sys))
//     {
//         sys.numberOfUpdates++;
//         applyPbc(sys);
//         if (atom.neighbor_flag == 1)
//             findNeighborON1(atom);
//         else if (atom.neighbor_flag == 2)
//             findNeighborON2(atom);
//         updateXyz0(atom);
//     }
// }

void initializeCells(System &sys)
{
    int d;
    double l;

    ITERATE_OVER_DIM(d)
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
    ITERATE_OVER_DIM(d)
    {
        numberOfCells *= sys.cellSizes[d];
    }
    sys.grid.resize(numberOfCells);

    for (Cell &cell : sys.grid)
        cell.atomIndex.resize(MAX_ATOMS_PER_CELL);
}

void updateCells(System &sys)
{
    double l;
    int n, d, ic[3];

    ITERATE_OVER_CELLS(ic, sys.cellSizes)
    {
        sys.grid[INDEX(ic, sys.cellSizes)].count = 0;
    }

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIM(d)
        {
            l = sys.box[d * 3 + d];
            ic[d] = (int)(sys.atoms[n].position[d] * sys.cellSizes[d] / l);
        }

        Cell &cell = sys.grid[INDEX(ic, sys.cellSizes)];
        if (cell.count > MAX_ATOMS_PER_CELL)
        {
            std::cerr << "Max number of atoms per cell exceeded: " << MAX_ATOMS_PER_CELL << std::endl;
            exit(1);
        }
        cell.atomIndex[cell.count++] = n;
    }
}

void computeForce(System &sys)
{
    int d, ic[3], kc[3], jc[3];
    double mag, rSqd, r[3];
    double rInv2, rInv6, rInv12;
    double l[3];

    ITERATE_OVER_DIM(d)
    {
        l[d] = sys.box[d * 3 + d];
    }

    ITERATE_OVER_CELLS(ic, sys.cellSizes)
    {
        Cell &cell_ic = sys.grid[INDEX(ic, sys.cellSizes)];

        // loop over atoms in cell ic
        for (int i = 0; i < cell_ic.count; ++i)
        {
            int index_i = cell_ic.atomIndex[i];
            ITERATE_OVER_DIM(d)
            {
                sys.atoms[index_i].force[d] = 0.0;
            }

            // iterate over neighbor cells of cell ic
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

                        // iterate over all atoms in cell jc
                        Cell &cell_jc = sys.grid[INDEX(jc, sys.cellSizes)];
                        for (int j = 0; j < cell_jc.count; ++j)
                        {
                            int index_j = cell_jc.atomIndex[j];
                            if (index_i != index_j)
                            {
                                rSqd = 0;
                                ITERATE_OVER_DIM(d)
                                {
                                    r[d] = sys.atoms[index_i].position[d] - sys.atoms[index_j].position[d];
                                    if (r[d] > l[d] * 0.5)
                                        r[d] -= l[d];
                                    else if (r[d] < -l[d] * 0.5)
                                        r[d] += l[d];
                                    rSqd += SQR(r[d]);
                                }
                                if (rSqd <= SQR(r_cut))
                                {
                                    rInv2 = 1. / rSqd;
                                    rInv6 = rInv2 * SQR(rInv2);
                                    rInv12 = SQR(rInv6);
                                    mag = rInv2 * (48. * rInv12 - 24. * rInv6);
                                    ITERATE_OVER_DIM(d)
                                    {
                                        std::cout << r[d] * mag << std::endl;
                                        sys.atoms[index_i].force[d] += r[d] * mag;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void integrateVelocityVerlet1(System &sys, const SimulationParameters &params)
{
    int n, d;
    double x;
    double l[3];

    ITERATE_OVER_DIM(d)
    {
        l[d] = sys.box[d * 3 + d];
    }

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIM(d)
        {
            x = sys.atoms[n].position[d] + params.timeStep * (sys.atoms[n].velocity[d] + params.timeStep * 0.5 / sys.atoms[n].mass * sys.atoms[n].force[d]);
            if ((x < 0.) || (x >= l[d]))
                x = fmod(x + 100. * l[d], l[d]);

            sys.atoms[n].position[d] = x;
            sys.atoms[n].velocity[d] += params.timeStep * 0.5 / sys.atoms[n].mass * sys.atoms[n].force[d];
        }
    }
}

void integrateVelocityVerlet2(System &sys, const SimulationParameters &params)
{
    int n, d;

    ITERATE_OVER_ATOMS(n, sys.numberOfAtoms)
    {
        ITERATE_OVER_DIM(d)
        {
            sys.atoms[n].velocity[d] += params.timeStep * 0.5 / sys.atoms[n].mass * sys.atoms[n].force[d];
        }
    }
}

// int main(int argc, char **argv)
int main()
{
    System sys;
    SimulationParameters params;

    readRun(params, "run.in");
    readXyz(sys, "Ar.xyz");
    initializeVelocity(sys, params.temperature);
    initializeCells(sys);
    updateCells(sys);
    computeForce(sys);

    std::cout
        << "Step Temperature KineticEnergy" << std::endl;
    for (int step = 0; step < params.numberOfSteps; ++step)
    {
        integrateVelocityVerlet1(sys, params);
        updateCells(sys);
        computeForce(sys);
        integrateVelocityVerlet2(sys, params);

        if (step % OUTPUT_FREQUENCY == 0)
        {
            const double kineticEnergy = ComputeKineticEnergy(sys);
            const double T = kineticEnergy / (3.0 / 2.0 * K_B * sys.numberOfAtoms);
            std::cout << step << " " << T << " " << kineticEnergy << std::endl;
        }
    }

    return 0;
}