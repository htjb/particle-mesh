#include <iostream>
#include <random>
#include <vector>
#include <fftw3.h>

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
    const double PI = 3.14;
    const double G = 1; // not doing real physics yet...

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

    int fftshape = Ngrid * Ngrid;

    double* in = fftw_alloc_real(fftshape);
    fftw_complex* out = fftw_alloc_complex(Ngrid * (Ngrid/2 + 1));

    // map 2D density to 1D in array for fft
    for (int i = 0; i < Ngrid; ++i) {
        for (int j = 0; j < Ngrid; ++j) {
            in[i * Ngrid + j] = density[i][j];
        }
    }

    vector<double> kx(Ngrid);
    vector<double> ky(Ngrid/2 + 1);
    vector<vector<double>> k_squared(Ngrid, vector<double>(Ngrid/2 + 1));
    for (int i = 0; i < Ngrid; ++i) {
        if (i <= Ngrid/2){
            kx[i] = (2 * PI / L) * i;
        } else{
            kx[i] = (2 * PI / L) * (i - Ngrid);
        }
    }

    for (int j = 0; j <= Ngrid/2; ++j) {
        ky[j] = (2 * PI / L)  * j;
    }

    // Compute k^2 at each FFT grid point
    for (int i = 0; i < Ngrid; ++i) {
        for (int j = 0; j <= Ngrid/2; ++j) {
            k_squared[i][j] = kx[i]*kx[i] + ky[j]*ky[j];
        }
    }

    fftw_plan plan = fftw_plan_dft_r2c_2d(Ngrid, Ngrid, in, out, FFTW_ESTIMATE);

    // Execute FFT
    fftw_execute(plan);

    // Now 'out' contains Fourier coefficients of your density field

    // ... (apply Poisson solver in Fourier space here) ...
    vector<vector<fftw_complex> > phi_k(Ngrid, vector<fftw_complex>(Ngrid/2 + 1));

    for (int i = 0; i < Ngrid; ++i) {
        for (int j = 0; j <= Ngrid/2; ++j) {
            double k2 = k_squared[i][j];
            if (k2 != 0) {
                double factor = -4 * PI * G / k2;
                phi_k[i][j][0] = factor * out[i * (Ngrid/2 + 1) + j][0]; // real part
                phi_k[i][j][1] = factor * out[i * (Ngrid/2 + 1) + j][1]; // imag part
            } else {
                phi_k[i][j][0] = 0.0; // avoid divide by zero at k=0
                phi_k[i][j][1] = 0.0;
            }
        }
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    double* phi_real = fftw_alloc_real(Ngrid * Ngrid);
    fftw_plan plan_back = fftw_plan_dft_c2r_2d(Ngrid, Ngrid, out, phi_real, FFTW_ESTIMATE);

    fftw_execute(plan_back);
    for (int i = 0; i < Ngrid * Ngrid; ++i) {
        phi_real[i] /= (Ngrid * Ngrid); // normalise the fft result i think...
    }

    fftw_destroy_plan(plan_back);
    // some weird pointer arithmatic
    vector<double> phi_vec(phi_real, phi_real + Ngrid * Ngrid);
    fftw_free(phi_real);

    cout << "Potential calcualted" << endl;
}