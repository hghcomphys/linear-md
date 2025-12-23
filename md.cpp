/*
 * Experimental linear molecular dynamics code implemented using
 * cell-list and neighbor-list methods.
 *
 * References:
 *  1. https://github.com/a-amouei/fmd
 *  2. https://github.com/brucefan1983/Molecular-Dynamics-Simulation
 *
 * Modified by:
 *   Hossein Ghorbanfekr (2025)
 *   https://github.com/hghcomphys
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#define K_B 8.617343e-5                  // Boltzmann's constant in natural unit
#define TIME_UNIT_CONVERSION 1.018051e+1 // from natural unit to fs
#define INDEX(ic, nc) ((ic)[0] + (nc)[0] * ((ic)[1] + (nc)[1] * (ic)[2]))

const int atomMaxNeighbors = 500;
const double cutoffRadius = 9.0;
const double skinRadius = 1.0;

struct System;
struct Parameters;
void readXyz(System &sys, const std::string &filename);
void readRun(Parameters &params, const std::string &filename);
void scaleVelocity(System &sys, const double T0);
void initializeVelocity(System &sys, const double T0);
void initializeNeighbors(System &sys, double cellLength);
void updateNeighbors(System &sys);
void computeForce(System &sys);
void integrateVerletOne(System &sys, const Parameters &params);
void integrateVerletTwo(System &sys, const Parameters &params);
void saveXyz(const System &sys);
void updatePositionOld(System &sys);
bool checkIfNeighborsNeedUpdate(const System &sys, double threshold);
double getDouble(std::string &token);
double getKineticEnergy(const System &sys);
std::vector<std::string> getTokens(std::ifstream &input);
inline void applyPBC(double &r, const double l);

struct Atom
{
    double mass;
    double position[3];
    double velocity[3];
    double force[3];

    double positionOld[3];
};

struct Cell
{
    int numberOfAtomsPerCell;
    std::vector<int> atomIndex;
};

struct Parameters
{
    int numberOfSteps;
    double timeStep;
    double temperature;
};

struct System
{
    int numberOfAtoms;
    std::vector<Atom> atoms;

    int cellSizes[3];
    int cellUpdates;
    int cellMaxAtoms;
    std::vector<Cell> grid;

    std::vector<int> neighborIndex;
    std::vector<int> neighborCount;

    double box[9];
    double potentialEnergy;
};

// int main(int argc, char **argv)
int main()
{
    System sys;
    Parameters params;

    readRun(params, "run.in");
    readXyz(sys, "Ar.xyz");
    initializeVelocity(sys, params.temperature);
    initializeNeighbors(sys, cutoffRadius);

    updateNeighbors(sys);
    computeForce(sys);

    std::cout
        << "Step Temperature KineticEnergy  PotentialEnergy"
        << "TotalEnergy NeighborUpdates AverageNeighbors"
        << std::endl;
    for (int step = 0; step < params.numberOfSteps; ++step)
    {
        integrateVerletOne(sys, params);
        if (checkIfNeighborsNeedUpdate(sys, 0.25 * skinRadius * skinRadius))
            updateNeighbors(sys);
        computeForce(sys);
        integrateVerletTwo(sys, params);

        if (step % 100 == 0)
        {
            // saveXyz(sys);
            const double pe = sys.potentialEnergy;
            const double ke = getKineticEnergy(sys);
            const double T = ke / (3.0 / 2.0 * K_B * sys.numberOfAtoms);

            double averageNeighbors = 0.0;
            for (int count : sys.neighborCount)
                averageNeighbors += (double)(count);
            averageNeighbors /= sys.numberOfAtoms;

            std::cout << step << " "
                      << T << " "
                      << ke << " "
                      << pe << " "
                      << ke + pe << " "
                      << sys.cellUpdates << " "
                      << averageNeighbors << " "
                      << std::endl;
        }
    }

    return 0;
}

void computeForce(System &sys)
{
    double r[3];

    const double l[3] = {sys.box[0], sys.box[4], sys.box[8]};
    const double epsilon = 1.032e-2;
    const double sigma = 3.405;
    const double cutoffSquare = cutoffRadius * cutoffRadius;
    const double sigma3 = sigma * sigma * sigma;
    const double sigma6 = sigma3 * sigma3;
    const double sigma12 = sigma6 * sigma6;
    const double e24s6 = 24.0 * epsilon * sigma6;
    const double e48s12 = 48.0 * epsilon * sigma12;
    const double e4s6 = 4.0 * epsilon * sigma6;
    const double e4s12 = 4.0 * epsilon * sigma12;

    for (int n = 0; n < sys.numberOfAtoms; ++n)
        for (int d = 0; d < 3; ++d)
            sys.atoms[n].force[d] = 0.0;
    sys.potentialEnergy = 0.0;

    for (int ni = 0; ni < sys.numberOfAtoms; ni++)
    {
        for (int j = 0; j < sys.neighborCount[ni]; j++)
        {
            const int nj = sys.neighborIndex[ni * atomMaxNeighbors + j];

            if (ni < nj)
            {
                double r2 = 0.0;
                for (int d = 0; d < 3; ++d)
                {
                    r[d] = sys.atoms[nj].position[d] - sys.atoms[ni].position[d];
                    applyPBC(r[d], l[d]);
                    r2 += r[d] * r[d];
                }
                if (r2 > cutoffSquare)
                    continue;

                const double r2inv = 1.0 / r2;
                const double r4inv = r2inv * r2inv;
                const double r6inv = r2inv * r4inv;
                const double r8inv = r4inv * r4inv;
                const double r12inv = r4inv * r8inv;
                const double r14inv = r6inv * r8inv;
                const double fij = e24s6 * r8inv - e48s12 * r14inv;

                for (int d = 0; d < 3; ++d)
                {
                    sys.atoms[ni].force[d] += fij * r[d];
                    sys.atoms[nj].force[d] -= fij * r[d];
                }
                sys.potentialEnergy += e4s12 * r12inv - e4s6 * r6inv;
            }
        }
    }
}

void updateNeighborList(System &sys)
{
    double r[3];
    int ni, nj, ic[3], jc[3], kc[3];

    const double l[3] = {sys.box[0], sys.box[4], sys.box[8]};
    const int *nc = sys.cellSizes;
    const double neighborCutoff = cutoffRadius + skinRadius;
    const double squaredNeighborCutoff = neighborCutoff * neighborCutoff;
    std::fill(sys.neighborCount.begin(), sys.neighborCount.end(), 0);

    // Iterate over cells
    for (ic[0] = 0; ic[0] < nc[0]; ic[0]++)
        for (ic[1] = 0; ic[1] < nc[1]; ic[1]++)
            for (ic[2] = 0; ic[2] < nc[2]; ic[2]++)
            {
                // Iterate over atoms in cell ic
                const auto &cell_ic = sys.grid[INDEX(ic, sys.cellSizes)];

                for (int i = 0; i < cell_ic.numberOfAtomsPerCell; ++i)
                {
                    ni = cell_ic.atomIndex[i];
                    const auto &atom_ni = sys.atoms[ni];

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
                                const auto &cell_jc = sys.grid[INDEX(jc, sys.cellSizes)];
                                for (int j = 0; j < cell_jc.numberOfAtomsPerCell; ++j)
                                {
                                    nj = cell_jc.atomIndex[j];

                                    if (ni < nj) // bug in original code
                                    {
                                        const auto &atom_nj = sys.atoms[nj];

                                        double r2 = 0.0;
                                        for (int d = 0; d < 3; ++d)
                                        {
                                            r[d] = atom_nj.position[d] - atom_ni.position[d];
                                            applyPBC(r[d], l[d]);
                                            r2 += r[d] * r[d];
                                        }
                                        if (r2 < squaredNeighborCutoff)
                                        {
                                            sys.neighborIndex[ni * atomMaxNeighbors + sys.neighborCount[ni]++] = nj;
                                            sys.neighborIndex[nj * atomMaxNeighbors + sys.neighborCount[nj]++] = ni;

                                            if ((sys.neighborCount[ni] > atomMaxNeighbors) || (sys.neighborCount[nj] > atomMaxNeighbors))
                                            {
                                                std::cout << "Error: max number of neighbors exceeds!" << std::endl;
                                                exit(1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

    updatePositionOld(sys);
}

void updateCellList(System &sys)
{
    double ic[3];
    const double l[3] = {sys.box[0], sys.box[4], sys.box[8]};

    for (auto &cell : sys.grid)
        cell.numberOfAtomsPerCell = 0;

    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        for (int d = 0; d < 3; ++d)
            ic[d] = (int)(sys.atoms[n].position[d] * sys.cellSizes[d] / l[d]);

        auto &cell = sys.grid[INDEX(ic, sys.cellSizes)];
        if (cell.numberOfAtomsPerCell > sys.cellMaxAtoms)
        {
            std::cerr << "Max number of atoms per cell exceeded: " << sys.cellMaxAtoms << std::endl;
            exit(1);
        }
        cell.atomIndex[cell.numberOfAtomsPerCell++] = n;
    }
}

void updateNeighbors(System &sys)
{
    updateCellList(sys);
    updateNeighborList(sys);

    // Update cells
    sys.cellUpdates += 1;
}

void integrateVerletOne(System &sys, const Parameters &params)
{
    double pos;

    const double l[3] = {sys.box[0], sys.box[4], sys.box[8]};
    const double dt = params.timeStep;

    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        auto &atom = sys.atoms[n];

        for (int d = 0; d < 3; ++d)
        {
            pos = atom.position[d] + dt * (atom.velocity[d] + dt * 0.5 / atom.mass * atom.force[d]);
            if ((pos < 0.) || (pos >= l[d]))
                pos = fmod(pos + 100. * l[d], l[d]);

            atom.position[d] = pos;
            atom.velocity[d] += dt * 0.5 / atom.mass * atom.force[d];
        }
    }
}

void integrateVerletTwo(System &sys, const Parameters &params)
{
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        auto &atom = sys.atoms[n];

        for (int d = 0; d < 3; ++d)
            atom.velocity[d] += params.timeStep * 0.5 / atom.mass * atom.force[d];
    }
}

void updatePositionOld(System &sys)
{
    for (int n = 0; n < sys.numberOfAtoms; ++n)
        for (int d = 0; d < 3; ++d)
            sys.atoms[n].positionOld[d] = sys.atoms[n].position[d];
}

bool checkIfNeighborsNeedUpdate(const System &sys, double threshold)
{
    bool needUpdate = false;

    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        double r2 = 0.0;
        for (int d = 0; d < 3; ++d)
            r2 += sys.atoms[n].position[d] - sys.atoms[n].positionOld[d];
        if (r2 > threshold)
        {
            needUpdate = true;
            break;
        }
    }
    return needUpdate;
}
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
    for (int d1 = 0; d1 < 3; ++d1)
    {
        for (int d2 = 0; d2 < 3; ++d2)
        {
            sys.box[d1 * 3 + d2] = getDouble(tokens[d1 * 3 + d2]);
            std::cout << sys.box[d1 * 3 + d2] << " ";
        }
        std::cout << std::endl;
    }

    // starting from line 3
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        tokens = getTokens(input);
        if (tokens.size() < 5)
        {
            std::cerr << "The 3rd line and later of xyz.in "
                         "should have 5 items."
                      << std::endl;
            exit(1);
        }
        for (int d = 0; d < 3; ++d)
            sys.atoms[n].position[d] = getDouble(tokens[d + 1]);
        sys.atoms[n].mass = getDouble(tokens[4]);
    }

    input.close();
}

void readRun(Parameters &params, const std::string &filename)
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
            else if (tokens[0] == "temperature")
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

double getKineticEnergy(const System &sys)
{
    double v2;
    double kineticEnergy = 0.0;

    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        v2 = 0.0;
        for (int d = 0; d < 3; ++d)
            v2 += sys.atoms[n].velocity[d] * sys.atoms[n].velocity[d];
        kineticEnergy += sys.atoms[n].mass * v2;
    }
    return 0.5 * kineticEnergy;
}

void scaleVelocity(System &sys, const double T0)
{
    const double temperature =
        getKineticEnergy(sys) * 2.0 / (3.0 * K_B * sys.numberOfAtoms);
    double scaleFactor = sqrt(T0 / temperature);

    for (int n = 0; n < sys.numberOfAtoms; ++n)
        for (int d = 0; d < 3; ++d)
            sys.atoms[n].velocity[d] *= scaleFactor;
}

void initializeVelocity(System &sys, const double T0)
{
    double totalMass = 0.0;
    double centerOfMassVelocity[3] = {0.0, 0.0, 0.0};

#ifndef DEBUG
    srand(42);
#endif

    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        totalMass += sys.atoms[n].mass;

        for (int d = 0; d < 3; ++d)
        {
            sys.atoms[n].velocity[d] = -1.0 + (rand() * 2.0) / RAND_MAX;
            centerOfMassVelocity[d] += sys.atoms[n].mass * sys.atoms[n].velocity[d];
        }
    }
    for (int d = 0; d < 3; ++d)
        centerOfMassVelocity[d] /= totalMass;

    for (int n = 0; n < sys.numberOfAtoms; ++n)
        for (int d = 0; d < 3; ++d)
            sys.atoms[n].velocity[d] -= centerOfMassVelocity[d];

    scaleVelocity(sys, T0);
}

void initializeNeighbors(System &sys, double cellLength)
{
    const double l[3] = {sys.box[0], sys.box[4], sys.box[8]};

    for (int d = 0; d < 3; ++d)
    {
        sys.cellSizes[d] = (int)(l[d] / cellLength);
        if (sys.cellSizes[d] < 3)
        {
            std::cerr << "nc[" << d << "]=" << sys.cellSizes[d] << " under PBC, this must be greater than 2!" << std::endl;
            exit(1);
        }
    }
    std::cout << "Cell length = " << cellLength << std::endl;
    std::cout << "Cell sizes = "
              << sys.cellSizes[0] << " "
              << sys.cellSizes[1] << " "
              << sys.cellSizes[2] << " "
              << std::endl;

    sys.grid.resize(sys.cellSizes[0] * sys.cellSizes[1] * sys.cellSizes[2]);

    const double systemVolume = l[0] * l[1] * l[2];
    const double cellVolume = cellLength * cellLength * cellLength;
    sys.cellMaxAtoms = 3 * (int)(sys.numberOfAtoms / systemVolume * cellVolume);
    std::cout << "Cell max atoms = " << sys.cellMaxAtoms << std::endl;

    for (auto &cell : sys.grid)
        cell.atomIndex.resize(sys.cellMaxAtoms);

    sys.neighborCount.resize(sys.numberOfAtoms);
    sys.neighborIndex.resize(sys.numberOfAtoms * atomMaxNeighbors);
    updatePositionOld(sys);

    sys.cellUpdates = 0;
}

inline void applyPBC(double &r, const double l)
{
    if (r > l * 0.5)
        r -= l;
    else if (r < -l * 0.5)
        r += l;
}
void saveXyz(const System &sys)
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
