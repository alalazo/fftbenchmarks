#include <fftbenchmarks_utils.h>

#include <fftw3-mpi.h>

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    auto cli_description = po::options_description(
            "Benchmarks a 3D complex FFT using FFTW"
    );

    auto vRepetitions = vector<unsigned long>();
    auto vNx = vector<ptrdiff_t>();
    auto vNy = vector<ptrdiff_t>();
    auto vNz = vector<ptrdiff_t>();
    auto transpose_output = false;

    // Add all the different program options
    cli_description.add_options()
            ("help", "gives this help message")
            ("transpose", po::bool_switch(&transpose_output),
             "transpose the output of every transformation back to its original layout")
            ("repetitions,r",
             po::value<vector<unsigned long int>>(&vRepetitions)->multitoken()->default_value({100}, "100"),
             "number of FFTs performed during this run")
            ("nx", po::value<vector<ptrdiff_t>>(&vNx)->multitoken()->default_value({128}, "128"),
             "grid dimension along the x-axis")
            ("ny", po::value<vector<ptrdiff_t>>(&vNy)->multitoken()->default_value({128}, "128"),
             "grid dimension along the y-axis")
            ("nz", po::value<vector<ptrdiff_t>>(&vNz)->multitoken()->default_value({128}, "128"),
             "grid dimension along the z-axis");

    // Parse the command line and push the values to local variables
    auto vm = po::variables_map();
    po::store(po::parse_command_line(argc, argv, cli_description), vm);
    po::notify(vm);

    auto MpiMasterWrite = MpiWrite(0);

    if (vm.count("help")) {
        MpiMasterWrite(cli_description);
        MPI_Finalize();
        return 1;
    }

    if (transpose_output) {
        MpiMasterWrite("Final transposition [activated]\n");
    } else {
        MpiMasterWrite("Final transposition [deactivated]\n");
    }
    MpiMasterWrite(results_header("FFTW complex transform over a 3D region"));
    for (auto n: zip(vNx, vNy, vNz, vRepetitions)) {
        auto Nx = n[0];
        auto Ny = n[1];
        auto Nz = n[2];
        auto repetitions = n[3];

        // Global size of the 3D grid
        fftw_complex *data;
        ptrdiff_t alloc_local, local_n0, local_0_start;

        // Get local data size and allocate
        alloc_local = fftw_mpi_local_size_3d(Nx, Ny, Nz, MPI_COMM_WORLD, &local_n0, &local_0_start);
        data = fftw_alloc_complex(size_t(alloc_local));
        fill(&data[0][0], &data[0][0] + 2 * alloc_local, 0.0);

        // Create plan_forward for in-place forward DFT
        fftw_plan plan_forward;
        fftw_plan plan_backward;
        if (transpose_output) {
            plan_forward = fftw_mpi_plan_dft_3d(Nx, Ny, Nz, data, data, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
            plan_backward = fftw_mpi_plan_dft_3d(Nx, Ny, Nz, data, data, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
        } else {
            plan_forward = fftw_mpi_plan_dft_3d(
                    Nx, Ny, Nz, data, data, MPI_COMM_WORLD, FFTW_FORWARD,
                    FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT
            );
            plan_backward = fftw_mpi_plan_dft_3d(
                    Nx, Ny, Nz, data, data, MPI_COMM_WORLD, FFTW_BACKWARD,
                    FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN
            );
        }

        // Initialize data to some function my_function(x,y)
        for (auto i = 0; i < local_n0; ++i) {
            for (auto j = 0; j < Ny; ++j) {
                for (auto k = 0; k < Nz; ++k) {
                    auto idx = (i * Ny + j) * Nz + k;
                    data[idx][0] = i + j + k;
                    data[idx][1] = i * j * k;
                }
            }
        }

        auto scale = 1.0 / (Nx * Ny * Nz);
        // To avoid accounting for padding I need to execute a dry run before
        // measuring the l2 norm of the vector of doubles
        fftw_execute(plan_forward);
        fftw_execute(plan_backward);
        for_each(&data[0][0], &data[0][0] + 2 * alloc_local, [scale](auto &item) { item *= scale; });
        auto norm_before = l2square(&data[0][0], &data[0][0] + 2 * alloc_local);

        // Compute transforms, in-place, as many times as desired
        auto start_time = chrono::system_clock::now();
        for (auto ii = 0; ii < repetitions; ++ii) {
            fftw_execute(plan_forward);
            fftw_execute(plan_backward);
            for_each(&data[0][0], &data[0][0] + 2 * alloc_local, [scale](auto &item) { item *= scale; });
        }
        auto end_time = chrono::system_clock::now();
        auto elapsed = chrono::duration<double>(end_time - start_time);
        auto norm_after = l2square(&data[0][0], &data[0][0] + 2 * alloc_local);

        MpiMasterWrite(results_line(Nx, Ny, Nz, repetitions, elapsed, relative_error(norm_after, norm_before)));

        fftw_destroy_plan(plan_forward);
        fftw_destroy_plan(plan_backward);
        free(data);
    }
    fftw_mpi_cleanup();
    MPI_Finalize();
}
