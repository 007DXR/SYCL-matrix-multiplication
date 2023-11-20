//==============================================================
// Matrix Multiplication: SYCL Matrix Multiplication Common
//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <getopt.h>
#include <ctime>
#include <chrono>

using namespace sycl;

// # matrix multiplication kernel implementation in mm_dpcpp_*.cpp
void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N0, size_t N1, size_t N2, size_t M);

// # floating point error verification function
bool almost_equal(float a, float b)
{
    float tolerance = 1e-6;
    float diff = fabs(a - b);
    a = fabs(a);
    b = fabs(b);
    float bigger = (b > a) ? b : a;
    if (diff <= bigger * tolerance)
        return true;
    return false;
}

int main(int argc, char *argv[])
{

    size_t N0;
    size_t N1;
    size_t N2;
    size_t M;
    size_t D;
    int VERIFY = 0;
    int PRINT_OUTPUT_MATRIX = 0;
    int max_d = 0;

    // # command line arguments
    int arg;
    // compute duration except for input time cost
    auto computed_time = 0;

    while ((arg = getopt(argc, argv, "d:m:vp")) != -1)
    {
        //         std::cout<<"ARC "<<arg<<std::endl;
        switch (arg)
        {
        case 'd':
            D = std::atoi(optarg);
            break;

        case 'm':
            M = std::atoi(optarg);
            break;
        case 'v':
            VERIFY = 1;
            break;
        case 'p':
            PRINT_OUTPUT_MATRIX = 1;
            break;
        case 'h':
            std::cout << std::endl;
            std::cout << "Usage   : ./a.out -n0 <DIMENSION0> -n1 <DIMENSION1> -m <WORK_GROUP_SIZE> -v -p\n\n";
            std::cout << "          [-n] size for matrix, eg: 1024\n";
            std::cout << "          [-m] size of work_group, eg: 8/16\n";
            std::cout << "          [-v] verify output with linear computation on cpu\n";
            std::cout << "          [-p] print output matrix\n";
            std::cout << "Example : ./a.out -n 1024 -m 16 -v -p\n\n";
            std::exit(0);
        }
    }
    queue q(property::queue::enable_profiling{});
    std::cout << "Offload Device        : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size   : " << q.get_device().get_info<info::device::max_work_group_size>() << "\n";

    std::vector<int> dimensions(D);

    for (int d = 0; d < D; ++d)
    {
        std::cout << "Dimension " << d << ": ";
        std::cin >> dimensions[d];
        max_d = max(max_d, dimensions[d]);
    }

    std::cout << "Matrix 0 :" << std::endl;
    // # Initialize matrix0 with values
    N0 = dimensions[0];
    N1 = dimensions[1];
    std::vector<float> matrix_a(N0 * max_d);
    for (int i = 0; i < N0; i++)
        for (int j = 0; j < N1; j++)
        {
            std::cin >> matrix_a[i * N1 + j];
        }

    for (int d = 2; d < D; ++d)
    {
        N2 = dimensions[d];

        std::vector<float> matrix_b(N1 * N2);
        std::vector<float> matrix_c(N0 * N2);
        std::vector<float> matrix_d(N0 * N2);
        std::cout << "Matrix " << d - 1 << " :" << std::endl;
        // # Initialize matrices with values

        for (int i = 0; i < N1; i++)
            for (int j = 0; j < N2; j++)
            {
                std::cin >> matrix_b[i * N2 + j];
            }
        // # get start time
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        // # Call matrix multiplication kernel implementation
        mm_kernel(q, matrix_a, matrix_b, matrix_c, N0, N1, N2, M);
        auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
        //         only the time cost of matrix multiplication is added to the conputed_time
        computed_time += duration;
        // # Compute local and compare with offload computation if -v in cmd-line
        if (VERIFY)
        {
            int fail = 0;
            for (int i = 0; i < N0; i++)
            {
                for (int j = 0; j < N2; j++)
                {
                    for (int k = 0; k < N1; k++)
                    {
                        matrix_d[i * N2 + j] += matrix_a[i * N1 + k] * matrix_b[k * N2 + j];
                    }
                    if (!almost_equal(matrix_c[i * N2 + j], matrix_d[i * N2 + j]))
                        fail = 1;
                }
            }
            if (fail == 1)
            {
                std::cout << "FAIL\n";
                return 0;
            }
            else
            {
                std::cout << "PASS\n";
            }
        }
        for (int i = 0; i < N0; i++)
        {
            for (int j = 0; j < N2; j++)
            {
                //                 std::cout << matrix_c[i * N2 + j] << " " << lab/dxr -d 4 -m 16 -v -pmatrix_d[i * N2 + j] << " ";
                matrix_a[i * N2 + j] = matrix_c[i * N2 + j];
            }
            //             std::cout << "\n";
        }
        N1 = N2;
    }

    // # Print Output if -p in cmd-line
    std::cout << "Output Matrix: " << std::endl;
    if (PRINT_OUTPUT_MATRIX)
    {
        for (int i = 0; i < N0; i++)
        {
            for (int j = 0; j < N2; j++)
            {
                std::cout << matrix_a[i * N2 + j] << " ";
            }
            std::cout << "\n";
        }
    }
    else
    {
        std::cout << " [0][0] = " << matrix_a[0] << "\n";
    }
    // # print kernel compute duration from host

    std::cout << "Compute Duration      : " << computed_time / 1e+9 << " seconds\n";
    return 0;
}
