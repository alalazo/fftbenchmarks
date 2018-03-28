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

    auto repetitions = 0ul;
    auto Nx = 0;
    auto Ny = 0;
    auto Nz = 0;

    // Add all the different program options
    cli_description.add_options()
            ("help", "gives this help message")
            ("repetitions,r", po::value<unsigned long int>(&repetitions)->default_value(100),
             "number of FFTs performed during this run")
            ("nx", po::value<int>(&Nx)->default_value(128), "grid dimension along the x-axis")
            ("ny", po::value<int>(&Ny)->default_value(128), "grid dimension along the y-axis")
            ("nz", po::value<int>(&Nz)->default_value(128), "grid dimension along the y-axis");

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

    // AccFFT needs a Cartesian communicator, setting to zero
    // seems to let the library decide automatically on the size
    auto c_dims = array<int, 2>({0, 0});
    MPI_Comm c_comm;
    accfft_create_comm(MPI_COMM_WORLD, c_dims.data(), &c_comm);

    // Global size of the 2D grid
    Complex * data = nullptr;
    Complex * data_cpu = nullptr;
    ptrdiff_t alloc_local;

    // Get local data size and allocate
    auto isize = array<int, 3>();
    auto osize = array<int, 3>();
    auto istart = array<int, 3>();
    auto ostart = array<int, 3>();
    // Apparently there are issues with Nz == 1
    auto n = array<int, 3>({Nx, Ny, Nz});

    alloc_local = accfft_local_size_dft_c2c_gpu(
            n.data(), isize.data(), istart.data(),
            osize.data(), ostart.data(), c_comm
    );
    data_cpu = reinterpret_cast<Complex *>(malloc(alloc_local));
    cudaMalloc((void**) &data, alloc_local);

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

    auto norm_before = accumulate(&data_cpu[0][0], &data_cpu[0][0] + 2 * isize[0] * isize[1] * n[2], 0.0,
                                  [](const double &sum, const double &x) {
                                      return sum + x * x;
                                  });
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
        //for_each(&data[0][0], &data[0][0] + 2 * isize[0] * isize[1] * n[2], [scale](auto &item) { item *= scale; });
    }
    auto end_time = chrono::system_clock::now();
    auto elapsed = chrono::duration<double>(end_time - start_time);

    // Copy data back to host
    cudaMemcpy(data_cpu, data, alloc_local, cudaMemcpyDeviceToHost);
    MPI_Barrier(MPI_COMM_WORLD);

    stringstream msg;
    msg.str(string());
    msg << "Elapsed time: " << elapsed.count() << " sec.\n\n";
    MpiMasterWrite(msg.str());

    auto norm_after = accumulate(&data_cpu[0][0], &data_cpu[0][0] + 2 * isize[0] * isize[1] * n[2], 0.0,
                                 [](const double &sum, const double &x) {
                                     return sum + x * x;
                                 });

    msg.str(string());
    msg << "Initial square norm: " << norm_before << " ";
    msg << "Final square norm: " << norm_after << "\n";
    msg << "Relative error: " << (norm_after - norm_before) / norm_before << "\n";
    MpiMasterWrite(msg.str());

    cublasDestroy(handle);
    free(data_cpu);
    cudaFree(data);
    accfft_destroy_plan(plan);
    accfft_cleanup_gpu();
    MPI_Finalize();
}
