#include <fftbenchmarks_utils.h>

#include <accfft_gpu.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>

#include <cstdlib>

using namespace std;
namespace po = boost::program_options;


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    accfft_init();

    auto cli_description = po::options_description(
            "Benchmarks a 3D complex FFT using AccFFT"
    );

    auto vRepetitions = vector<unsigned long>();
    auto vNx = vector<int>();
    auto vNy = vector<int>();
    auto vNz = vector<int>();

    // Add all the different program options
    cli_description.add_options()
            ("help", "gives this help message")
            ("repetitions,r",
             po::value<vector<unsigned long int>>(&vRepetitions)->multitoken()->default_value({100}, "100"),
             "number of FFTs performed during this run")
            ("nx", po::value<vector<int>>(&vNx)->multitoken()->default_value({128}, "128"),
             "grid dimension along the x-axis")
            ("ny", po::value<vector<int>>(&vNy)->multitoken()->default_value({128}, "128"),
             "grid dimension along the y-axis")
            ("nz", po::value<vector<int>>(&vNz)->multitoken()->default_value({128}, "128"),
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

    // AccFFT needs a Cartesian communicator, setting to zero
    // seems to let the library decide automatically on the size
    auto c_dims = array<int, 2>({0, 0});
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims.data(), &c_comm);

    MpiMasterWrite(results_header("AccFFT-GPU complex transform over a 3D region"));

    for (auto n: zip(vNx, vNy, vNz, vRepetitions)) {
        auto Nx = n[0];
        auto Ny = n[1];
        auto Nz = n[2];
        auto repetitions = n[3];

        // Global size of the 2D grid
        Complex *data = nullptr;
        Complex *data_cpu = nullptr;
        ptrdiff_t alloc_local;

        // Get local data size and allocate
        auto isize = array<int, 3>();
        auto osize = array<int, 3>();
        auto istart = array<int, 3>();
        auto ostart = array<int, 3>();

        alloc_local = accfft_local_size_dft_c2c_gpu(
                n.data(), isize.data(), istart.data(),
                osize.data(), ostart.data(), c_comm
        );
        data_cpu = reinterpret_cast<Complex *>(malloc(alloc_local));
        cudaMalloc((void **) &data, alloc_local);

        // Create a plan, differently from FFTW the direction of the FFT is set
        // at execution time
        auto plan = accfft_plan_dft_3d_c2c_gpu(n.data(), data, data, c_comm, ACCFFT_MEASURE);

        // Initialize data to some function my_function(x,y)
        for (int i = 0; i < isize[0]; i++) {
            for (int j = 0; j < isize[1]; j++) {
                for (int k = 0; k < isize[2]; k++) {
                    auto ptr = i * isize[1] * n[2] + j * n[2] + k;
                    data_cpu[ptr][0] = i + j + k; // Real Component
                    data_cpu[ptr][1] = i * j * k; // Imag Component
                }
            }
        }

        auto norm_before = l2square(&data_cpu[0][0], &data_cpu[0][0] + 2 * isize[0] * isize[1] * n[2]);

        // Copy data to device
        cudaMemcpy(data, data_cpu, alloc_local, cudaMemcpyHostToDevice);
        MPI_Barrier(MPI_COMM_WORLD);

        // Set the handle to cuBlas
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Compute transforms, in-place, as many times as desired
        auto start_time = chrono::system_clock::now();
        auto scale = 1.0 / (Nx * Ny * Nz);
        for (auto ii = 0ul; ii < repetitions; ++ii) {
            accfft_execute_c2c_gpu(plan, ACCFFT_FORWARD, data, data);
            accfft_execute_c2c_gpu(plan, ACCFFT_BACKWARD, data, data);
            cublasDscal(handle, 2 * isize[0] * isize[1] * n[2], &scale, &data[0][0], 1);
        }
        auto end_time = chrono::system_clock::now();
        auto elapsed = chrono::duration<double>(end_time - start_time);

        // Copy data back to host
        cudaMemcpy(data_cpu, data, alloc_local, cudaMemcpyDeviceToHost);
        MPI_Barrier(MPI_COMM_WORLD);
        auto norm_after = l2square(&data_cpu[0][0], &data_cpu[0][0] + 2 * isize[0] * isize[1] * n[2]);

        MpiMasterWrite(results_line(Nx, Ny, Nz, repetitions, elapsed, relative_error(norm_after, norm_before)));

        cublasDestroy(handle);
        free(data_cpu);
        cudaFree(data);
        accfft_destroy_plan(plan);
    }
    accfft_cleanup_gpu();
    MPI_Finalize();
}
