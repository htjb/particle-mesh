#include <iostream>
#include <random>
#include <vector>

using namespace std;

struct Particle {
    double x, y;
    double vx, vy;
};


int main() {

    cout << "Running 2D Particle Mesh Code." << endl;
    
    const double L = 1.0; // box size
    const double N = 1000; // number of particles
    const int Ngrid = 64;
    const double mass = 10; 

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> particle_initalisation_distribution(0.0, L);

    vector<Particle> particles(N);
    for (int i = 0; i < N; ++i){
        particles[i].x = particle_initalisation_distribution(gen);
        particles[i].y = particle_initalisation_distribution(gen);
        particles[i].vx = 0;
        particles[i].vy = 0;
    }

    cout << "Initialised Particles." << endl;

    const double dx = L/Ngrid;

    vector<vector<double> > density(Ngrid, vector<double>(Ngrid, 0.0));

    for (const auto& p : particles){
        int di = p.x/dx;
        int dj = p.y/dx;
        di = di % Ngrid;
        dj = dj % Ngrid;
        density[di][dj] += mass;
    }
    
    cout << "Intialised Density." << endl;

}