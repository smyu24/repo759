#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <cstring>
#include <omp.h>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Size of the board
// Function to calculate acceleration due to gravity
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N)
{
    // TODO:
    // init all of variables
    double dx, dy, dz, inv_r3;
    std::memset(acc, 0, N * 3 * sizeof(double));

// number of rows for N is given
// acc is also given
#pragma omp parallel for schedule(runtime) private(dx, dy, dz, inv_r3)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i != j)
            {
                dx = pos[j][0] - pos[i][0];
                dy = pos[j][1] - pos[i][1];
                dz = pos[j][2] - pos[i][2];
                inv_r3 = pow((pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(softening, 2)), -1.5);
                // inv_r3 = pow((dx * dx + dy * dy + dz * dz + softening * softening), -1.5);

                acc[i][0] += G * (dx * inv_r3) * mass[j];
                acc[i][1] += G * (dy * inv_r3) * mass[j];
                acc[i][2] += G * (dz * inv_r3) * mass[j];
            }
        }
    }

    // no return; modified value of acc within scope of function

    /* Python Ver.
    """
    Calculate the acceleration on each particle due to Newton's Law
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """

    N = pos.shape[0]
    a = np.zeros((N, 3))

    for i in range(N):
        for j in range(N):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2) ** (-1.5)
                a[i, 0] += G * (dx * inv_r3) * mass[j, 0]
                a[i, 1] += G * (dy * inv_r3) * mass[j, 0]
                a[i, 2] += G * (dz * inv_r3) * mass[j, 0]

    return a
    */
}

// For debug: save positions to a CSV file
void savePositionsToCSV(const double pos[][3], int N, int step, const std::string &filename)
{
    std::ofstream file;

    // Open the file in append mode
    file.open(filename, std::ios_base::app);

    if (file.is_open())
    {
        file << step << ",[";
        for (int i = 0; i < N; i++)
        {
            if (i != N - 1)
                file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "],";
            else
                file << "[" << pos[i][0] << "," << pos[i][1] << "," << pos[i][2] << "]";
        }
        file << "]\n"; // Newline for separation between steps
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file!" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    // Check if correct number of arguments are provided
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <number_of_particles> <simulation_end_time>" << std::endl;
        return 1;
    }

    // Read N and tEnd from command line
    int N = std::stoi(argv[1]);       // Number of particles
    double tEnd = std::stod(argv[2]); // Time at which simulation ends
    int num_threads = std::stoi(argv[3]);
    int schedule_type = std::stoi(argv[4]);
    // 1:"static", 2:"dynamic", 3:"guided"

    omp_set_num_threads(num_threads);
    if (schedule_type == 1) {
        omp_set_schedule(omp_sched_static, 0);
    } else if (schedule_type == 2) {
        omp_set_schedule(omp_sched_dynamic, 0);
    } else if (schedule_type == 3) {
        omp_set_schedule(omp_sched_guided, 0);
    } else {
        return 1;
    }

    // File to save positions
    std::string filename = "positions.csv";

    // Clear the file before starting simulation (optional)
    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::trunc);
    file.close();

    // Allocate dynamic arrays based on N
    double *mass = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    // Create a random number engine
    std::mt19937 generator(std::random_device{}());

    // Create random distributions
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Simulation parameters
    double t = 0.0;

    // Set initial masses and random positions/velocities
    for (int i = 0; i < N; i++)
    {
        mass[i] = uniform_dist(generator);

        pos[i][0] = normal_dist(generator);
        pos[i][1] = normal_dist(generator);
        pos[i][2] = normal_dist(generator);

        vel[i][0] = normal_dist(generator);
        vel[i][1] = normal_dist(generator);
        vel[i][2] = normal_dist(generator);
    }

    // Convert to Center-of-Mass frame
    double velCM[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;
    for (int i = 0; i < N; i++)
    {
        velCM[0] += vel[i][0] * mass[i];
        velCM[1] += vel[i][1] * mass[i];
        velCM[2] += vel[i][2] * mass[i];
        totalMass += mass[i];
    }

    velCM[0] /= totalMass;
    velCM[1] /= totalMass;
    velCM[2] /= totalMass;

    for (int i = 0; i < N; i++)
    {
        vel[i][0] -= velCM[0];
        vel[i][1] -= velCM[1];
        vel[i][2] -= velCM[2];
    }

    // Initial accelerations
    getAcc(pos, mass, acc, N);

    // Number of timesteps
    int Nt = int(tEnd / dt);

    // Main simulation loop
    for (int step = 0; step < Nt; step++)
    {

        // TODO: (1/2) kick
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++)
        {
            vel[i][0] += acc[i][0] * dt / 2.0;
            vel[i][1] += acc[i][1] * dt / 2.0;
            vel[i][2] += acc[i][2] * dt / 2.0;
        }

        // TODO: Drift
        // TODO: Ensure particles stay within the board limits
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++)
        {
            pos[i][0] += vel[i][0] * dt;
            pos[i][1] += vel[i][1] * dt;
            pos[i][2] += vel[i][2] * dt;

            for (int d = 0; d < 3; d++)
            {
                if (pos[i][d] > board_size)
                    pos[i][d] = board_size;
                if (pos[i][d] < -board_size)
                    pos[i][d] = -board_size;
            }
        }

        // Update accelerations
        getAcc(pos, mass, acc, N);

        // TODO: (1/2) kick
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++)
        {
            vel[i][0] += acc[i][0] * dt / 2.0;
            vel[i][1] += acc[i][1] * dt / 2.0;
            vel[i][2] += acc[i][2] * dt / 2.0;
        }

        // Update time
        t += dt;

        // For debug: save positions to CSV at each step
        // savePositionsToCSV(pos, N, step, filename);
    }

    // Clean up dynamically allocated memory
    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << "time: " << duration_sec.count() << "ms\n";

    return 0;
}
