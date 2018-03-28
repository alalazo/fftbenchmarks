#ifndef FFTBENCHMARKS_FFTBENCHMARKS_UTILS_H_20180328
#define FFTBENCHMARKS_FFTBENCHMARKS_UTILS_H_20180328

#include <mpi.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>


auto MpiWrite = [](int target_rank) {
    auto rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return [rank, target_rank](auto message) {
        if (rank == target_rank) {
            std::cout << message;
        }
    };
};


template<class T>
double l2square(T begin, T end) {
    auto norm_after = 0.0;
    auto lnorm_after = std::accumulate(
            begin, end, 0.0,
            [](const double &sum, const double &x) {
                return sum + x * x;
            });
    MPI_Allreduce(&lnorm_after, &norm_after, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return norm_after;
}

template<class T>
T relative_error(T before, T after) {
    return (after - before) / before;
}

template<class T>
std::string results_header(T title) {
    std::stringstream msg;
    auto size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    msg << title << " [" << size << " MPI tasks]\n";
    msg << std::setw(6) << "Nx" << std::setw(6) << "Ny" << std::setw(6) << "Nz";
    msg << std::setw(8) << "Repeat" << std::setw(18) << "Elapsed (sec.)";
    msg << std::setw(18) << "Avg. FFT (sec.)" << std::setw(18) << "Relative err." << std::endl;
    return msg.str();
}

template<class T, class R, class E>
std::string results_line(T Nx, T Ny, T Nz, R repetitions, E elapsed, double rel_error) {
    std::stringstream msg;
    msg << std::setw(6) << Nx << std::setw(6) << Ny << std::setw(6) << Nz;
    msg << std::setw(8) << repetitions << std::setw(18) << std::scientific << elapsed.count();
    msg << std::setw(18) << elapsed.count() / repetitions << std::setw(18) << rel_error << "\n";
    return msg.str();
}

template<class T>
std::vector<std::array<T, 3>> zip(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z) {

    auto length = std::min(std::min(x.size(), y.size()), z.size());
    auto result = std::vector<std::array<T, 3>>();

    for (auto ii = 0ul; ii < length; ++ii) {
        result.push_back({x[ii], y[ii], z[ii]});
    }

    return result;
}
#endif //FFTBENCHMARKS_FFTBENCHMARKS_UTILS_H_20180328
