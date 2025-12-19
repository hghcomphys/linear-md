
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

const int Ns = 100;                              // output frequency
const double K_B = 8.617343e-5;                  // Boltzmann's constant in natural unit
const double TIME_UNIT_CONVERSION = 1.018051e+1; // from natural unit to fs

struct SimulationParameters
{
    int numberOfSteps;
    double timeStep;
    double temperature;
};

struct System
{
    int numberOfAtoms;
    int numberOfUpdates;

    double box[9];
    double potentialEnergy;

    std::vector<double> mass;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;

    std::vector<double> x0, y0, z0;
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
    sys.mass.resize(sys.numberOfAtoms, 0.0);
    sys.x.resize(sys.numberOfAtoms, 0.0);
    sys.y.resize(sys.numberOfAtoms, 0.0);
    sys.z.resize(sys.numberOfAtoms, 0.0);
    sys.vx.resize(sys.numberOfAtoms, 0.0);
    sys.vy.resize(sys.numberOfAtoms, 0.0);
    sys.vz.resize(sys.numberOfAtoms, 0.0);
    sys.fx.resize(sys.numberOfAtoms, 0.0);
    sys.fy.resize(sys.numberOfAtoms, 0.0);
    sys.fz.resize(sys.numberOfAtoms, 0.0);
    sys.x0.resize(sys.numberOfAtoms, 0.0);
    sys.y0.resize(sys.numberOfAtoms, 0.0);
    sys.z0.resize(sys.numberOfAtoms, 0.0);

    // line 2
    tokens = getTokens(input);
    if (tokens.size() != 9)
    {
        std::cerr << "The second line of xyz.in should have 9 items." << std::endl;
        exit(1);
    }

    for (int d1 = 0; d1 < 3; ++d1)
    {
        for (int d2 = 0; d2 < 3; ++d2)
        {
            sys.box[d2 * 3 + d1] = getDouble(tokens[d1 * 3 + d2]);
        }
    }

    std::cout << "Box matrix H = " << std::endl;
    for (int d1 = 0; d1 < 3; ++d1)
    {
        for (int d2 = 0; d2 < 3; ++d2)
        {
            std::cout << sys.box[d1 * 3 + d2] << " ";
        }
        std::cout << std::endl;
    }

    // starting from line 3
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        tokens = getTokens(input);
        if (tokens.size() != 5)
        {
            std::cerr << "The 3rd line and later of xyz.in "
                         "should have 5 items."
                      << std::endl;
            exit(1);
        }
        // atom types not used
        sys.x[n] = getDouble(tokens[1]);
        sys.y[n] = getDouble(tokens[2]);
        sys.z[n] = getDouble(tokens[3]);
        sys.mass[n] = getDouble(tokens[4]);
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
    double kineticEnergy = 0.0;
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        double v2 = sys.vx[n] * sys.vx[n] + sys.vy[n] * sys.vy[n] + sys.vz[n] * sys.vz[n];
        kineticEnergy += sys.mass[n] * v2;
    }
    return kineticEnergy * 0.5;
}

void scaleVelocity(System &sys, const double T0)
{
    const double temperature =
        ComputeKineticEnergy(sys) * 2.0 / (3.0 * K_B * sys.numberOfAtoms);
    double scaleFactor = sqrt(T0 / temperature);
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        sys.vx[n] *= scaleFactor;
        sys.vy[n] *= scaleFactor;
        sys.vz[n] *= scaleFactor;
    }
}

void initializeVelocity(System &sys, const double T0)
{
#ifndef DEBUG
    srand(42);
#endif
    double centerOfMassVelocity[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        totalMass += sys.mass[n];
        sys.vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
        sys.vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
        sys.vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
        centerOfMassVelocity[0] += sys.mass[n] * sys.vx[n];
        centerOfMassVelocity[1] += sys.mass[n] * sys.vy[n];
        centerOfMassVelocity[2] += sys.mass[n] * sys.vz[n];
    }
    centerOfMassVelocity[0] /= totalMass;
    centerOfMassVelocity[1] /= totalMass;
    centerOfMassVelocity[2] /= totalMass;
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        sys.vx[n] -= centerOfMassVelocity[0];
        sys.vy[n] -= centerOfMassVelocity[1];
        sys.vz[n] -= centerOfMassVelocity[2];
    }
    scaleVelocity(sys, T0);
}

bool checkIfNeedUpdate(const System &sys)
{
    bool needUpdate = false;
    for (int n = 0; n < sys.numberOfAtoms; ++n)
    {
        double dx = sys.x[n] - sys.x0[n];
        double dy = sys.y[n] - sys.y0[n];
        double dz = sys.z[n] - sys.z0[n];
        if (dx * dx + dy * dy + dz * dz > 0.25)
        {
            needUpdate = true;
            break;
        }
    }
    return needUpdate;
}

void applyPbcOne(double &sx)
{
    if (sx < 0.0)
    {
        sx += 1.0;
    }
    else if (sx > 1.0)
    {
        sx -= 1.0;
    }
}

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

// int main(int argc, char **argv)
int main()
{
    System sys;
    SimulationParameters params;

    readRun(params, "run.in");
    readXyz(sys, "Ar.xyz");
    initializeVelocity(sys, params.temperature);

    std::cout << "Step Temperature KineticEnergy PotentialEnergy" << std::endl;
    for (int step = 0; step < params.numberOfSteps; ++step)
    {
        // if (atom.neighbor_flag != 0)
        //     findNeighbor(atom);
        // integrate(true, timeStep, atom);  // step 1 in the book
        // findForce(atom);                  // step 2 in the book
        // integrate(false, timeStep, atom); // step 3 in the book
        if (step % Ns == 0)
        {
            const double kineticEnergy = ComputeKineticEnergy(sys);
            const double T = kineticEnergy / (1.5 * K_B * sys.numberOfAtoms);
            std::cout << step << " " << T << " " << kineticEnergy << " " << sys.potentialEnergy << std::endl;
        }
    }

    return 0;
}