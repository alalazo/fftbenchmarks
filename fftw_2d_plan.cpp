#include <fftw3-mpi.h>

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    auto cli_description = po::options_description(
            "Benchmarks a 2D complex FFT using FFTW"
    );

    auto repetitions = 0ul;
    auto Nx = ptrdiff_t(0);
    auto Ny = ptrdiff_t(0);

    // Add all the different program options
    cli_description.add_options()
            ("help", "gives this help message")
            ("repetitions,r", po::value<unsigned long int>(&repetitions)->default_value(100),
             "number of FFTs performed during this run")
            ("nx", po::value<ptrdiff_t>(&Nx)->default_value(128), "grid dimension along the x-axis")
            ("ny", po::value<ptrdiff_t>(&Ny)->default_value(128), "grid dimension along the y-axis");

    // Parse the command line and push the values to local variables
    auto vm = po::variables_map();
    po::store(po::parse_command_line(argc, argv, cli_description), vm);
    po::notify(vm);

    auto rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto MpiWrite = [rank](int target_rank) {
        return [rank, target_rank](auto message) {
            if (rank == target_rank) {
                cout << message;
            }
        };
    };
    auto MpiMasterWrite = MpiWrite(0);

    if (vm.count("help")) {
        MpiMasterWrite(cli_description);
        MPI_Finalize();
        return 1;
    }

    // Global size of the 2D grid
    fftw_complex *data;
    ptrdiff_t alloc_local, local_n0, local_0_start, i, j;

    // Get local data size and allocate
    alloc_local = fftw_mpi_local_size_2d(Nx, Ny, MPI_COMM_WORLD, &local_n0, &local_0_start);
    data = fftw_alloc_complex(size_t(alloc_local));
    fill(&data[0][0], &data[0][0] + 2 * alloc_local, 0.0);

    // Create plan_forward for in-place forward DFT
    fftw_plan plan_forward;
    plan_forward = fftw_mpi_plan_dft_2d(Nx, Ny, data, data, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_backward;
    plan_backward = fftw_mpi_plan_dft_2d(Nx, Ny, data, data, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);

    // Initialize data to some function my_function(x,y)
    for (i = 0; i < local_n0; ++i) {
        for (j = 0; j < Ny; ++j) {
            data[i * Ny + j][0] = local_0_start + i;
            data[i * Ny + j][1] = j;
        }
    }

    auto norm_before = accumulate(&data[0][0], &data[0][0] + 2 * alloc_local, 0.0,
                                  [](const double &sum, const double &x) {
                                      return sum + x * x;
                                  });

    // Compute transforms, in-place, as many times as desired
    auto start_time = chrono::system_clock::now();
    auto scale = 1.0 / (Nx * Ny);
    for (auto ii = 0ul; ii < repetitions; ++ii) {
        fftw_execute(plan_forward);
        fftw_execute(plan_backward);
        for_each(&data[0][0], &data[0][0] + 2 * alloc_local, [scale](auto &item) { item *= scale; });
    }
    auto end_time = chrono::system_clock::now();
    auto elapsed = chrono::duration<double>(end_time - start_time);
    stringstream msg;
    msg.str(string());
    msg << "Elapsed time: " << elapsed.count() << " sec.\n\n";
    MpiMasterWrite(msg.str());

    auto norm_after = accumulate(&data[0][0], &data[0][0] + 2 * alloc_local, 0.0,
                                 [](const double &sum, const double &x) {
                                     return sum + x * x;
                                 });

    msg.str(string());
    msg << "Initial square norm: " << norm_before << " ";
    msg << "Final square norm: " << norm_after << "\n";
    msg << "Relative error: " << (norm_after - norm_before) / norm_before << "\n";
    MpiMasterWrite(msg.str());

    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    free(data);
    fftw_mpi_cleanup();
    MPI_Finalize();
}
